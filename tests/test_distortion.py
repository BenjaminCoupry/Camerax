import jax
import pytest

from camerax.camera import get_K_matrix
from camerax.distortion import (
    get_pixel_to_pinhole,
    get_pinhole_to_pixel,
)


ATOL = 1e-2
N_TEST = 15
WIDTH = 1920
HEIGHT = 1080
K = get_K_matrix(1500.0, WIDTH / 2, HEIGHT / 2)


def random_distortion(key):
    """Generate random radial distortion coefficients, monotonic over the image."""
    return jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)

def random_pixels(key, n=8):
    """Generate random pixels inside the image."""
    return jax.random.uniform(
        key,
        (n, 2),
        minval=0.0,
        maxval=jax.numpy.asarray([WIDTH, HEIGHT]),
    )

@pytest.mark.parametrize("seed", range(N_TEST))
def test_distortion_then_undistortion_round_trip(seed):
    key_k, key_pixels = jax.random.split(jax.random.key(seed))
    distortion_parameters = random_distortion(key_k)
    pinhole_pixels = random_pixels(key_pixels)
    pinhole_to_pixel = get_pinhole_to_pixel(K, WIDTH, HEIGHT, distortion_parameters)
    pixel_to_pinhole = get_pixel_to_pinhole(K, WIDTH, HEIGHT, distortion_parameters)

    assert jax.numpy.allclose(
        pixel_to_pinhole(pinhole_to_pixel(pinhole_pixels)),
        pinhole_pixels,
        atol=ATOL,
    )

@pytest.mark.parametrize("seed", range(N_TEST))
def test_undistortion_then_distortion_round_trip(seed):
    key_k, key_pixels = jax.random.split(jax.random.key(seed))
    distortion_parameters = random_distortion(key_k)
    pixels = random_pixels(key_pixels)
    pinhole_to_pixel = get_pinhole_to_pixel(K, WIDTH, HEIGHT, distortion_parameters)
    pixel_to_pinhole = get_pixel_to_pinhole(K, WIDTH, HEIGHT, distortion_parameters)

    assert jax.numpy.allclose(
        pinhole_to_pixel(pixel_to_pinhole(pixels)),
        pixels,
        atol=ATOL,
    )

def test_distortion_is_differentiable_at_the_principal_point():
    distortion_parameters = jax.numpy.asarray([0.02, -0.01, 0.005])
    principal_point = K[:2, 2]
    pinhole_to_pixel = get_pinhole_to_pixel(K, WIDTH, HEIGHT, distortion_parameters)
    pixel_to_pinhole = get_pixel_to_pinhole(K, WIDTH, HEIGHT, distortion_parameters)

    assert jax.numpy.all(jax.numpy.isfinite(jax.jacfwd(pinhole_to_pixel)(principal_point)))
    assert jax.numpy.all(jax.numpy.isfinite(jax.jacfwd(pixel_to_pinhole)(principal_point)))
