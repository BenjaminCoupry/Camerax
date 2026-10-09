import jax

import camerax.vectors as vectors
import camerax.camera as camera

def get_polynomial_coefficients(distortion_parameters):
    """Build the coefficients of the radial distortion polynomial.

    Parameters
    ----------
    distortion_parameters : array_like, (3, )
        Radial distortion coefficients ``(k1, k2, k3)``.

    Returns
    -------
    polynomial_coefficients : jax.Array, (8, )
        Coefficients of ``x + k1 * x**3 + k2 * x**5 + k3 * x**7``, from the
        highest to the lowest degree (the order expected by
        `jax.numpy.polyval`).
    """
    polynomial_coefficients = jax.numpy.flip(jax.numpy.asarray([0, 1, 0, distortion_parameters[0], 0, distortion_parameters[1], 0, distortion_parameters[2]]))
    return polynomial_coefficients

def get_pixel_to_pinhole(K, width, height, distortion_parameters, steps=5000):
    """Build the undistortion from distorted pixels to pinhole pixels.

    Parameters
    ----------
    K : array_like, (3, 3)
        Intrinsic matrix.
    width : int
        Image width, in pixels.
    height : int
        Image height, in pixels.
    distortion_parameters : array_like, (3, )
        Radial distortion coefficients ``(k1, k2, k3)``.
    steps : int, optional
        Number of radii at which the inverse is tabulated. Default is 5000.

    Returns
    -------
    pixel_to_pinhole : callable, R^2 -> R^2
        Maps a distorted pixel to the pixel of the ideal pinhole camera.
        Also accepts arrays of shape ``(..., 2)``.

    Notes
    -----
    The inverse of the distortion polynomial is not available in closed
    form. It is tabulated by solving the polynomial for ``steps`` radii
    regularly spaced in ``[0, max(width, height)]``, measured from the
    principal point, then linearly interpolated. Radii beyond this range
    are clamped to the last tabulated value, and the smallest real positive
    root is kept, which assumes a monotonic distortion over the image.
    The polynomial is solved with `jax.numpy.roots`, which runs on CPU only.
    """
    scale = camera.get_scale(K, width, height)
    principal_point = camera.get_principal_point(K)
    polynomial_coefficients = get_polynomial_coefficients(distortion_parameters)
    root_finder = lambda distorted_radius : scale * vectors.smallest_real_positive(jax.numpy.roots(polynomial_coefficients.at[-1].set(-distorted_radius/scale), strip_zeros=False))
    sample_range = jax.numpy.linspace(0, jax.numpy.maximum(width, height), steps)
    sample_values = jax.vmap(root_finder)(sample_range)
    def pixel_to_pinhole(pixel):
        distorted_radius, phi = jax.numpy.unstack(vectors.cartesian_to_polar(pixel, principal_point), axis=-1)
        radius = jax.numpy.interp(distorted_radius, sample_range, sample_values)
        pinhole_pixel = vectors.polar_to_cartesian(jax.numpy.stack([radius, phi],axis=-1), principal_point)
        return pinhole_pixel
    return pixel_to_pinhole

def get_pinhole_to_pixel(K, width, height, distortion_parameters):
    """Build the distortion from pinhole pixels to distorted pixels.

    Parameters
    ----------
    K : array_like, (3, 3)
        Intrinsic matrix.
    width : int
        Image width, in pixels.
    height : int
        Image height, in pixels.
    distortion_parameters : array_like, (3, )
        Radial distortion coefficients ``(k1, k2, k3)``.

    Returns
    -------
    pinhole_to_pixel : callable, R^2 -> R^2
        Maps a pixel of the ideal pinhole camera to the distorted pixel.
        Also accepts arrays of shape ``(..., 2)``.

    Notes
    -----
    With ``x = r / scale`` the radius from the principal point divided by
    `get_scale`, the distorted radius is
    ``scale * (x + k1 * x**3 + k2 * x**5 + k3 * x**7)``. The azimuth around
    the principal point is unchanged.
    """
    scale = camera.get_scale(K, width, height)
    principal_point = camera.get_principal_point(K)
    polynomial_coefficients = get_polynomial_coefficients(distortion_parameters)
    def pinhole_to_pixel(pinhole_pixel):
        radius, phi = jax.numpy.unstack(vectors.cartesian_to_polar(pinhole_pixel, principal_point), axis=-1)
        distorted_radius = scale * jax.numpy.polyval(polynomial_coefficients, radius/scale)
        pixel = vectors.polar_to_cartesian(jax.numpy.stack([distorted_radius, phi],axis=-1), principal_point)
        return pixel
    return pinhole_to_pixel
