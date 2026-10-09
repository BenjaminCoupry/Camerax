import jax

def norm_vector(v, epsilon = 1e-8):
    """Compute the norm and the direction of vectors.

    Parameters
    ----------
    v : array_like, (..., n)
        Input vectors.
    epsilon : float, optional
        Regularization added under the square root, which keeps the norm
        differentiable and non-zero at the origin.

    Returns
    -------
    norm : jax.Array, (...)
        Regularized norm of each vector.
    direction : jax.Array, (..., n)
        Each vector divided by its regularized norm.
    """
    s = jax.numpy.square(v)
    norm = jax.numpy.sqrt(jax.numpy.sum(s, axis=-1) + epsilon)
    direction = v / jax.numpy.expand_dims(norm, axis=-1)
    return norm, direction

def to_homogeneous(v):
    """Append a unit coordinate to vectors.

    Parameters
    ----------
    v : array_like, (..., n)
        Input vectors.

    Returns
    -------
    homogeneous : jax.Array, (..., n + 1)
        Vectors with a last coordinate equal to 1.
    """
    append_term = jax.numpy.ones(jax.numpy.shape(v)[:-1]+(1,))
    homogeneous = jax.numpy.append(v,append_term,axis=-1)
    return homogeneous

def from_homogeneous(v):
    """Divide homogeneous vectors by their last coordinate.

    Parameters
    ----------
    v : array_like, (..., n + 1)
        Homogeneous vectors.

    Returns
    -------
    p : jax.Array, (..., n)
        Vectors divided by the last coordinate, which is then dropped.
    valid : jax.Array, (...)
        ``True`` where the last coordinate is positive. For a point
        projected by a camera, this means it lies in front of it.
    """
    p = v[..., :-1] / v[..., -1:]
    valid = jax.numpy.sign(v[..., -1]) > 0
    return p, valid

def apply_transform(transform, points):
    """Apply a homogeneous transform to points.

    Parameters
    ----------
    transform : array_like, (n + 1, n + 1)
        Homogeneous transform matrix.
    points : array_like, (..., n)
        Points to transform.

    Returns
    -------
    transformed : jax.Array, (..., n)
        Transformed points.
    """
    homogeneous = to_homogeneous(points)
    transformed = jax.numpy.einsum('uk, ...k -> ...u', transform, homogeneous)[...,:-1]
    return transformed

def smallest_real_positive(v, epsilon = 1e-6):
    """Select the smallest non-negative real value among complex roots.

    Parameters
    ----------
    v : array_like, (..., n)
        Complex values, typically the roots of a polynomial.
    epsilon : float, optional
        A value is considered real when the magnitude of its imaginary part
        is below ``epsilon``.

    Returns
    -------
    smallest : jax.Array, (...)
        Smallest real part among the real, non-negative values. Equal to
        ``inf`` when there is none.
    """
    real_mask = jax.numpy.abs(jax.numpy.imag(v)) < epsilon
    positive_mask = jax.numpy.real(v) >= 0
    valid_v = jax.numpy.where(jax.numpy.logical_and(positive_mask, real_mask), jax.numpy.real(v), jax.numpy.inf)
    smallest = jax.numpy.min(valid_v, axis=-1)
    return smallest

def cartesian_to_polar(cartesian, center, epsilon = 1e-6):
    """Convert planar cartesian coordinates to polar coordinates.

    Parameters
    ----------
    cartesian : array_like, (..., 2)
        Cartesian coordinates ``(x, y)``.
    center : array_like, (2, )
        Origin of the polar coordinates.
    epsilon : float, optional
        Regularization that keeps ``rho`` and ``phi`` differentiable at the
        center.

    Returns
    -------
    polar : jax.Array, (..., 2)
        Polar coordinates ``(rho, phi)`` relative to ``center``, with
        ``phi`` in ``[-pi, pi]``.

    Notes
    -----
    The regularization biases ``rho`` and ``phi`` close to the center, and
    is negligible far from it. The center itself maps to ``rho = sqrt(epsilon)``.
    """
    x, y = jax.numpy.unstack(cartesian - center, axis=-1)
    rho = jax.numpy.sqrt(jax.numpy.square(x) + jax.numpy.square(y) + epsilon)
    phi = jax.numpy.arctan2(y, x + epsilon)
    polar = jax.numpy.stack([rho, phi], axis=-1)
    return polar

def polar_to_cartesian(polar, center):
    """Convert planar polar coordinates to cartesian coordinates.

    Parameters
    ----------
    polar : array_like, (..., 2)
        Polar coordinates ``(rho, phi)``.
    center : array_like, (2, )
        Origin of the polar coordinates.

    Returns
    -------
    cartesian : jax.Array, (..., 2)
        Cartesian coordinates ``(x, y)``.
    """
    rho, phi = jax.numpy.unstack(polar, axis=-1)
    x = rho * jax.numpy.cos(phi)
    y = rho * jax.numpy.sin(phi)
    cartesian = jax.numpy.stack([x, y],axis=-1) + center
    return cartesian
