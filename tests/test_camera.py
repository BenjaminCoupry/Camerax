import jax
import pytest

from camerax.camera import (
    get_K_matrix,
    get_projection,
    get_deprojection,
)


ATOL = 1e-2
N_TEST = 15
WIDTH = 1920
HEIGHT = 1080


def random_K(key):
    """Generate a random intrinsic matrix with a near-centered principal point."""
    key_f, key_fy, key_x, key_y = jax.random.split(key, 4)
    focal_length = jax.random.uniform(key_f, minval=800.0, maxval=2500.0)
    fy = focal_length * jax.random.uniform(key_fy, minval=0.95, maxval=1.05)
    x0 = WIDTH / 2 + jax.random.uniform(key_x, minval=-50.0, maxval=50.0)
    y0 = HEIGHT / 2 + jax.random.uniform(key_y, minval=-50.0, maxval=50.0)
    return get_K_matrix(focal_length, x0, y0, fy=fy)

@pytest.mark.parametrize("seed", range(N_TEST))
def test_deprojection_then_projection_round_trip(seed):
    key_K, key_pixel, key_depth = jax.random.split(jax.random.key(seed), 3)
    K = random_K(key_K)
    pixel = jax.random.uniform(
        key_pixel,
        (2,),
        minval=0.0,
        maxval=jax.numpy.asarray([WIDTH, HEIGHT]),
    )
    depth = jax.random.uniform(key_depth, minval=0.5, maxval=10.0)
    projection = get_projection(K)
    deprojection = get_deprojection(K)

    ray_origin, ray_direction = deprojection(pixel)
    point = ray_origin + depth * ray_direction
    projected, valid = projection(point)

    assert valid
    assert jax.numpy.allclose(projected, pixel, atol=ATOL)

@pytest.mark.parametrize("seed", range(N_TEST))
def test_projection_then_deprojection_round_trip(seed):
    key_K, key_point = jax.random.split(jax.random.key(seed))
    K = random_K(key_K)
    point = jax.random.normal(key_point, (3,))
    point = point.at[2].set(jax.numpy.abs(point[2]) + 0.5)
    projection = get_projection(K)
    deprojection = get_deprojection(K)

    pixel, valid = projection(point)
    ray_origin, ray_direction = deprojection(pixel)

    assert valid
    assert jax.numpy.allclose(ray_origin, 0.0, atol=ATOL)
    assert jax.numpy.allclose(
        ray_direction,
        point / jax.numpy.linalg.vector_norm(point),
        atol=ATOL,
    )

@pytest.mark.parametrize("seed", range(N_TEST))
def test_projection_flags_points_behind_the_camera(seed):
    key_K, key_point = jax.random.split(jax.random.key(seed))
    K = random_K(key_K)
    point = jax.random.normal(key_point, (3,))
    projection = get_projection(K)

    _, valid_front = projection(point.at[2].set(jax.numpy.abs(point[2]) + 0.5))
    _, valid_behind = projection(point.at[2].set(-jax.numpy.abs(point[2]) - 0.5))

    assert valid_front
    assert not valid_behind
