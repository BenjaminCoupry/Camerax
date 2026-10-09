import jax
import pytest

from camerax.vectors import (
    to_homogeneous,
    from_homogeneous,
    cartesian_to_polar,
    polar_to_cartesian,
)


ATOL = 1.5e-5
N_TEST = 15


@pytest.mark.parametrize("seed", range(N_TEST))
def test_homogeneous_round_trip(seed):
    v = jax.random.normal(jax.random.key(seed), (8, 3))
    p, valid = from_homogeneous(to_homogeneous(v))

    assert jax.numpy.allclose(p, v, atol=ATOL)
    assert jax.numpy.all(valid)

@pytest.mark.parametrize("seed", range(N_TEST))
def test_polar_round_trip(seed):
    key_cartesian, key_center = jax.random.split(jax.random.key(seed))
    cartesian = 10.0 * jax.random.normal(key_cartesian, (8, 2))
    center = jax.random.normal(key_center, (2,))
    polar = cartesian_to_polar(cartesian, center)

    assert jax.numpy.allclose(
        polar_to_cartesian(polar, center),
        cartesian,
        atol=1e-3,
    )

def test_polar_is_differentiable_at_the_center():
    center = jax.numpy.asarray([100.0, 50.0])
    round_trip = lambda p: polar_to_cartesian(cartesian_to_polar(p, center), center)

    assert jax.numpy.all(jax.numpy.isfinite(jax.jacfwd(round_trip)(center)))
