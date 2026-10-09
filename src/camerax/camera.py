import jax

import camerax.vectors as vectors

def get_scale(K, width, height):
    """Compute the focal length expressed for the largest image side.

    Parameters
    ----------
    K : array_like, (3, 3)
        Intrinsic matrix.
    width : int
        Image width, in pixels.
    height : int
        Image height, in pixels.

    Returns
    -------
    scale : jax.Array, (, )
        Focal length ``K[0, 0]`` rescaled to the largest image side, in
        pixels.

    Notes
    -----
    This follows the Meshroom convention: the focal length is normalized by
    the image width, then multiplied by ``max(width, height)``. The radial
    distortion coefficients are defined relative to this scale.
    """
    f_x = K[0,0]
    scale = f_x/width*jax.numpy.maximum(width, height)
    return scale

def get_principal_point(K):
    """Extract the principal point from an intrinsic matrix.

    Parameters
    ----------
    K : array_like, (3, 3)
        Intrinsic matrix.

    Returns
    -------
    principal_point : jax.Array, (2, )
        Principal point ``(x0, y0)``, in pixels.
    """
    principal_point = K[:2,2]
    return principal_point

def get_K_matrix(focal_length, x0, y0, fy=None):
    """Build an intrinsic matrix.

    Parameters
    ----------
    focal_length : float
        Focal length along the x axis, in pixels.
    x0 : float
        Horizontal coordinate of the principal point, in pixels.
    y0 : float
        Vertical coordinate of the principal point, in pixels.
    fy : float, optional
        Focal length along the y axis, in pixels. Default is
        ``focal_length`` (square pixels).

    Returns
    -------
    K : jax.Array, (3, 3)
        Intrinsic matrix, without skew.

    Notes
    -----
    The pixel convention, i.e. the position of the pixel centers, is
    entirely set by ``x0`` and ``y0``. The package does not assume one.
    """
    K = jax.numpy.asarray([[focal_length, 0, x0],
                    [0, focal_length if fy is None else fy, y0],
                    [0, 0, 1]])
    return K

def get_projection(K, pinhole_to_pixel=None):
    """Build a projection from camera space to pixel coordinates.

    Parameters
    ----------
    K : array_like, (3, 3)
        Intrinsic matrix.
    pinhole_to_pixel : callable, R^2 -> R^2, optional
        Distortion applied to the ideal pinhole pixel, for instance the
        output of `get_pinhole_to_pixel`. Default is no distortion.

    Returns
    -------
    projection : callable, R^3 -> (R^2, bool)
        Maps a point expressed in the camera frame to its pixel and a
        validity flag, which is ``True`` when the point lies in front of the
        camera (positive depth).

    Notes
    -----
    The camera looks along the positive z axis. The pixel of a point behind
    the camera is still computed, and is not meaningful.
    """
    def projection(point):
        homogeneous_pixel = jax.numpy.matmul(K, point)
        pinhole_pixel, valid = vectors.from_homogeneous(homogeneous_pixel)
        pixel = pinhole_to_pixel(pinhole_pixel) if pinhole_to_pixel is not None else pinhole_pixel
        return pixel, valid
    return projection

def get_deprojection(K, pixel_to_pinhole=None):
    """Build a deprojection from pixel coordinates to a camera ray.

    Parameters
    ----------
    K : array_like, (3, 3)
        Intrinsic matrix.
    pixel_to_pinhole : callable, R^2 -> R^2, optional
        Undistortion applied to the pixel, for instance the output of
        `get_pixel_to_pinhole`. Default is no undistortion.

    Returns
    -------
    deprojection : callable, R^2 -> (R^3, R^3)
        Maps a pixel to the origin and the unit direction of its ray,
        expressed in the camera frame.

    Notes
    -----
    The ray origin is the camera center, i.e. the origin of the camera
    frame.
    """
    ray_origin = jax.numpy.zeros(3)
    inv_K = jax.numpy.linalg.inv(K)
    def deprojection(pixel):
        pinhole_pixel = pixel_to_pinhole(pixel) if pixel_to_pinhole is not None else pixel
        homogeneous_pixel = vectors.to_homogeneous(pinhole_pixel)
        _, ray_direction = vectors.norm_vector(jax.numpy.matmul(inv_K, homogeneous_pixel))
        return ray_origin, ray_direction
    return deprojection
