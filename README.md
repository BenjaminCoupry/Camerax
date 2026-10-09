# Camerax

A small JAX-based Python package for pinhole cameras with radial distortion, following the Meshroom conventions.

The package provides the projection of camera-space points to pixels and the deprojection of pixels to camera rays, the radial distortion polynomial and its numerical inverse, and the rigid transforms between the camera and the world frames. The projections, deprojections, distortions and transforms are built by `get_*` factories that return a function acting on a single point, to be wrapped in `jax.vmap` for batches.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone <repository-url>
cd Camerax
python -m pip install -e .
```

The test suite needs `pytest`, pulled in by the `test` extra:

```bash
python -m pip install -e ".[test]"
```

## Examples

### Projecting world points to distorted pixels

```python
import jax
from camerax import (
    get_K_matrix,
    get_world_to_camera,
    get_projection,
    get_pinhole_to_pixel,
)

width, height = 1920, 1080
K = get_K_matrix(1500.0, width / 2, height / 2)
distortion_parameters = jax.numpy.asarray([0.08, 0.06, -0.3])  # k1, k2, k3

R = jax.numpy.eye(3)  # camera-to-world rotation
c = jax.numpy.asarray([0.0, 0.0, -2.0])  # camera center, in the world frame

world_to_camera = get_world_to_camera(R, c)
pinhole_to_pixel = get_pinhole_to_pixel(K, width, height, distortion_parameters)
projection = get_projection(K, pinhole_to_pixel)

points = jax.random.uniform(jax.random.key(0), (100, 3), minval=-0.5, maxval=0.5)
pixels, valid = jax.vmap(projection)(world_to_camera(points))
```

### Casting rays through distorted pixels

```python
import jax
from camerax import (
    get_K_matrix,
    get_camera_to_world,
    get_deprojection,
    get_pixel_to_pinhole,
)

width, height = 1920, 1080
K = get_K_matrix(1500.0, width / 2, height / 2)
distortion_parameters = jax.numpy.asarray([0.08, 0.06, -0.3])

R = jax.numpy.eye(3)
c = jax.numpy.asarray([0.0, 0.0, -2.0])

camera_to_world = get_camera_to_world(R, c)
pixel_to_pinhole = get_pixel_to_pinhole(K, width, height, distortion_parameters)
deprojection = get_deprojection(K, pixel_to_pinhole)

columns, rows = jax.numpy.meshgrid(jax.numpy.arange(width), jax.numpy.arange(height))
pixels = jax.numpy.stack([columns, rows], axis=-1).reshape(-1, 2)
ray_origins, ray_directions = jax.vmap(deprojection)(pixels)

ray_origins = camera_to_world(ray_origins)  # the camera center, in the world frame
ray_directions = ray_directions @ R.T  # directions rotate, they do not translate
```

## Conventions

**Frames.** The camera looks along the positive `z` axis. `R` is the camera-to-world rotation, whose columns are the camera axes expressed in the world frame, and `c` is the camera center in the world frame. This is the Meshroom convention, and the transform built by `get_world_to_camera(R, c)` is the exact inverse of the one built by `get_camera_to_world(R, c)`. The transpose of an OpenCV-style world-to-camera rotation is the corresponding `R`.

**Pixels.** The intrinsic matrix `K` has no skew. The position of the pixel centers is set entirely by the principal point `(x0, y0)`; the package does not assume one.

**Validity.** The projection built by `get_projection` returns the pixel together with a flag, `True` when the point lies in front of the camera. The pixel of a point behind the camera is computed, and is not meaningful.

**Rays.** The deprojection built by `get_deprojection` returns the camera center and the unit direction of the ray, in the camera frame.

## Radial distortion

Let `r` be the distance of a pinhole pixel to the principal point, and `x = r / scale` its normalization, where `scale` is the focal length expressed for the largest image side, `K[0, 0] / width * max(width, height)`. The distorted radius is

```text
r_d = scale * (x + k1 * x^3 + k2 * x^5 + k3 * x^7)
```

and the azimuth around the principal point is unchanged. The function built by `get_pinhole_to_pixel` applies this polynomial.

The function built by `get_pixel_to_pinhole` inverts it numerically: the polynomial is solved for a table of radii in `[0, max(width, height)]`, and the table is linearly interpolated. The inverse is therefore built once, when the function is created, and assumes a distortion that is monotonic over the image. Pixels beyond the tabulated range are clamped.

## References

- Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
- Brown, D. C. (1966). Decentering distortion of lenses. *Photogrammetric Engineering*, 32(3), 444-462.
- Griwodz, C., Gasparini, S., Calvet, L., Gurdjos, P., Castan, F., Maujean, B., De Lillo, G., & Lanthony, Y. (2021). AliceVision Meshroom: An open-source 3D reconstruction pipeline. In *Proceedings of the 12th ACM Multimedia Systems Conference* (pp. 241-247).
