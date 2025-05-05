"""Extrema finding and optimization.

Extrema in the response scale space are keypoints in the SURF algorithm.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int

from .gradient import gradient_at_center_pixel_2d
from .hessian import hessian_at_center_pixel_2d
from .types import Coord3D, ImageScaleSpace, ScaleSpace, ThresholdedScaleSpace


def find_extrema(x: ScaleSpace, window_size: int = 3) -> ThresholdedScaleSpace:
    """Find candidate local extremal points in the input scale space by non-maximal suppression.

    Args:
        x (ScaleSpace): scale space of images.
        window_size (int): size of window to use. For SURF, a window size of 3 is used.

    Returns:
        ThresholdedScaleSpace: binary mask where 1's are detected extrema.
    """
    n_dim = len(x.shape)
    window_dim = tuple(
        [
            window_size,
        ]
        * n_dim
    )
    window_stride = tuple(
        [
            1,
        ]
        * n_dim
    )
    loc_max = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, window_dim, window_stride, padding="valid"
    )
    loc_max = jnp.pad(loc_max, 1, mode="constant", constant_values=jnp.inf)
    return x == loc_max


def threshold_responses(
    ss: ScaleSpace, threshold: float, z_score: bool = False
) -> ThresholdedScaleSpace:
    """Find candidate extremal points by thresholding the responses.

    Threshold can be set either absolutely, or as a multiple of the standard deviation of all values (`z_score=True`).

    Args:
        ss (ScaleSpace): the scale space.
        threshold (float): threshold value.
        z_score (bool, optional): whether to z-score the input scale space and set the threshold as values >`threshold*sigma` or treat the threshold as an absolute value (`z_score=False`). Defaults to False.

    Returns:
        ThresholdedScaleSpace: binary mask where 1's are detected extrema.
    """
    if z_score:
        return (ss - jnp.mean(ss)) / jnp.std(ss) > threshold
    else:
        return ss > threshold


def find_threshold_extrema(
    ss: ScaleSpace,
    threshold: float,
    z_score: bool = False,
    window_size: int = 3,
) -> ThresholdedScaleSpace:
    """Find candidate extremal points by identifying local maxima and thresholding.

    Args:
        ss (ScaleSpace): the scale space.
        threshold (float): the threshold value.
        z_score (bool, optional): whether to z-score the input scale space and set the threshold as values >`threshold*sigma` or treat the threshold as an absolute value (`z_score=False`). Defaults to False.
        window_size (int, optional): window size to use for non-maximal suppression. Defaults to 3.

    Returns:
        ThresholdedScaleSpace: binary mask where 1's are detected extrema.
    """
    return jnp.logical_and(
        find_extrema(ss, window_size),
        threshold_responses(ss, threshold, z_score),
    )


def interpolate_extrema_2d(
    ss: ImageScaleSpace, num_iter: int, coords: Int[Array, "n 3"]
) -> Int[Array, "m 3"]:
    def body_fun(i: int, crds: Int[Array, "n 3"]) -> Int[Array, "n 3"]:
        return jax.vmap(Partial(_interpolate_extrema_one_step, ss), 0, 0)(crds)

    coordu = jax.lax.fori_loop(0, num_iter, body_fun, coords)
    return coordu[jax.vmap(Partial(_check_final, ss), 0, 0)(coordu)]


def _interpolate_extrema_one_step(
    ss: ImageScaleSpace, coord: Coord3D
) -> Coord3D:
    return jax.lax.select(
        jnp.all(coord == 1), jnp.ones_like(coord), _do_update(ss, coord)
    )


def _do_update(ss: ImageScaleSpace, coord: Coord3D) -> Coord3D:
    cube = jax.lax.dynamic_slice(ss, coord - 1, (3, 3, 3))
    update = _extremum_update(cube).flatten()
    coordu = coord + jnp.around(update, decimals=0).astype(int)
    coordu = jnp.where(
        _coord_is_valid(ss, coordu), coordu, jnp.ones_like(coordu)
    )
    return jnp.where(jnp.any(jnp.abs(update) > 0.5), coordu, coord)


def _coord_is_valid(ss: ImageScaleSpace, coord: Coord3D):
    z, r, c = coord[0], coord[1], coord[2]
    mz, mr, mc = ss.shape
    okay_z = jnp.logical_and(z >= 0, z < mz)
    okay_r = jnp.logical_and(r >= 0, r < mr)
    okay_c = jnp.logical_and(c >= 0, c < mc)
    return jnp.logical_and(jnp.logical_and(okay_r, okay_c), okay_z)


def _extremum_update(cube: Float[Array, "3 3"]) -> Float[Array, " 3"]:
    d2h_dx2 = hessian_at_center_pixel_2d(cube)
    dh_dx = gradient_at_center_pixel_2d(cube)
    return -jnp.linalg.lstsq(d2h_dx2, dh_dx)[0]


def _check_final(ss: ImageScaleSpace, coord: Coord3D) -> Coord3D:
    cube = jax.lax.dynamic_slice(ss, coord - 1, (3, 3, 3))
    update = _extremum_update(cube).flatten()
    update_converged = jnp.all(jnp.abs(update) < 0.5)
    not_default = jnp.max(coord) > 1
    return jnp.logical_and(update_converged, not_default)
