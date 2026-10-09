import jax
import functools

import camerax.vectors as vectors

def get_camera_to_world(R, c):
    """Build the rigid transform from the camera frame to the world frame.

    Parameters
    ----------
    R : array_like, (3, 3)
        Camera-to-world rotation. Its columns are the camera axes expressed
        in the world frame.
    c : array_like, (3, )
        Camera center, expressed in the world frame.

    Returns
    -------
    camera_to_world : callable, R^3 -> R^3
        Maps points expressed in the camera frame to the world frame, as
        ``R @ x + c``. Also accepts arrays of shape ``(..., 3)``.

    Notes
    -----
    This follows the Meshroom convention, where the stored rotation is the
    camera-to-world one. Pass the transpose of an OpenCV-style rotation.
    """
    transform = jax.numpy.block([[R, jax.numpy.expand_dims(c, axis=-1)], [jax.numpy.zeros((1,3)), jax.numpy.ones((1,1))]])
    camera_to_world = functools.partial(vectors.apply_transform, transform)
    return camera_to_world

def get_world_to_camera(R, c):
    """Build the rigid transform from the world frame to the camera frame.

    Parameters
    ----------
    R : array_like, (3, 3)
        Camera-to-world rotation, as in `get_camera_to_world`.
    c : array_like, (3, )
        Camera center, expressed in the world frame.

    Returns
    -------
    world_to_camera : callable, R^3 -> R^3
        Maps points expressed in the world frame to the camera frame, as
        ``R.T @ (x - c)``. Also accepts arrays of shape ``(..., 3)``. It is
        the inverse of `get_camera_to_world` for the same ``R`` and ``c``.
    """
    R_inv = jax.numpy.swapaxes(R, -1, -2)
    t_inv = -jax.numpy.matmul(R_inv, c)
    transform = jax.numpy.block([[R_inv, jax.numpy.expand_dims(t_inv, axis=-1)], [jax.numpy.zeros((1,3)), jax.numpy.ones((1,1))]])
    world_to_camera = functools.partial(vectors.apply_transform, transform)
    return world_to_camera
