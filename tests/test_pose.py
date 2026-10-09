import jax
import pytest

from camerax.pose import (
    get_camera_to_world,
    get_world_to_camera,
)


ATOL = 1.5e-5
N_TEST = 15


def random_pose(key):
    """Generate a random camera-to-world rotation and camera center."""
    key_R, key_c = jax.random.split(key)
    R = jax.random.orthogonal(key_R, 3)
    R = R.at[:, 0].multiply(jax.numpy.linalg.det(R))
    c = jax.random.normal(key_c, (3,))
    return R, c

@pytest.mark.parametrize("seed", range(N_TEST))
def test_camera_to_world_origin_is_camera_center(seed):
    R, c = random_pose(jax.random.key(seed))
    camera_to_world = get_camera_to_world(R, c)

    assert jax.numpy.allclose(
        camera_to_world(jax.numpy.zeros(3)),
        c,
        atol=ATOL,
    )

@pytest.mark.parametrize("seed", range(N_TEST))
def test_camera_to_world_round_trip(seed):
    key_pose, key_points = jax.random.split(jax.random.key(seed))
    R, c = random_pose(key_pose)
    points = jax.random.normal(key_points, (8, 3))
    camera_to_world = get_camera_to_world(R, c)
    world_to_camera = get_world_to_camera(R, c)

    assert jax.numpy.allclose(
        world_to_camera(camera_to_world(points)),
        points,
        atol=ATOL,
    )

@pytest.mark.parametrize("seed", range(N_TEST))
def test_world_to_camera_round_trip(seed):
    key_pose, key_points = jax.random.split(jax.random.key(seed))
    R, c = random_pose(key_pose)
    points = jax.random.normal(key_points, (8, 3))
    camera_to_world = get_camera_to_world(R, c)
    world_to_camera = get_world_to_camera(R, c)

    assert jax.numpy.allclose(
        camera_to_world(world_to_camera(points)),
        points,
        atol=ATOL,
    )
