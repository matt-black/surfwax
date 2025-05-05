from functools import partial
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int

from ._util import strided_batch
from .extrema import find_threshold_extrema
from .haar import haar_response_2d
from .hessian import hessian_determinant_2d
from .types import Coord2D, Image, ImageOrVolume, IntegralImage, Volume

__all__ = ["upright_surf_2d"]


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _detect_keypoints_local_2d(
    im: IntegralImage,
    prev_scale: Volume,
    lobe_sizes: Int[Array, " n"],
    crop_size: int,
    normalize_responses: bool = True,
    threshold: float = 6,
    z_score: bool = True,
    max_pts: int = 10,
) -> Tuple[ImageOrVolume, Int[Array, "n 3"]]:
    # compute full-size response scale space
    ss = jax.vmap(
        Partial(hessian_determinant_2d, im, normalize=normalize_responses), 0, 0
    )(lobe_sizes[:, None])
    ss = jnp.concatenate([prev_scale, ss], axis=0)
    prev_scale = ss[-lobe_sizes.shape[0] + 1 :, ...].copy()
    imsze_at_filt = [im.shape[0] - 2 * crop_size, im.shape[1] - 2 * crop_size]
    slice = Partial(
        jax.lax.dynamic_slice,
        start_indices=[crop_size, crop_size],
        slice_sizes=imsze_at_filt,
    )
    ss = jax.vmap(slice, 0, 0)(ss)
    extt = find_threshold_extrema(ss, threshold, z_score, ss.shape[0])
    coords = jnp.stack(jnp.nonzero(extt, size=max_pts, fill_value=0), axis=-1)
    coords = coords.at[:, 1:].add(crop_size)
    return prev_scale, coords


def upright_surf_2d(
    im: IntegralImage,
    lobe_sizes: Sequence[int],
    octave_size: int,
    # localization
    normalize_responses: bool = True,
    extrema_threshold: float = 6.0,
    extrema_zscore: bool = True,
    # descriptor args
    desc_haar_scale_nstds: float = 2.0,
    desc_window_scale_nstds: float = 20.0,
    desc_subwins_per_dim: int = 4,
    desc_npts_per_dim: int = 5,
    max_pts_in_octave: int = 10,
) -> Float[Array, "m 67"]:
    """Find and compute descriptors for U-SURF keypoints of the input image.

    Args:
        im (Image): input image.
        lobe_sizes (Sequence[int]): lobe size of the filter to use at each scale.
        octave_size (int):
        normalize_responses (bool): normalize the filter responses to the size of the filter.
        extrema_threshold (float): threshold strength for calling points as candidate extrema.
        extrema_zscore (bool): whether the threshold is absolute (`False`) or is specified as a multiple of the standard deviation of the z-scored response values.
        desc_haar_scale_nstds (float, optional): scale of haar filters to use when calculating descriptors. Defaults to 2.0.
        desc_window_scale_nstds (float, optional): number of std. dev's of the detected scale to compute descriptor responses over. Defaults to 20.0.
        desc_subwins_per_dim (int, optional): number of subwindows to divide the descriptor region into in each dimension. Defaults to 4.
        desc_npts_per_dim (int, optional): number of points in each subwindow to query haar values at when forming descriptor. Defaults to 5.
        max_pts_in_octave (int, optional): maximum number of points that will be detected in a single octave. Defaults to 10.

    Returns:
        Array: U-SURF keypoint localizations in scale and space, and their descriptor vectors.
            rows in output array correspond to a single keypoint
            cols are [scale, y, x, ...vec...]
    """
    if octave_size % 2 == 0:
        raise ValueError("octave_size must be an odd number")
    hoct_rd = octave_size // 2
    if len(lobe_sizes) % (hoct_rd + 1) == 0:
        raise ValueError("# of lobe sizes must evenly divide octave_size//2+1")
    first_halfoct, rest_lobes = lobe_sizes[:hoct_rd], lobe_sizes[hoct_rd:]
    # pre-compute the first 1/2 of the first octave
    hess_fun = Partial(hessian_determinant_2d, im, normalize=True)
    prev_scales = jax.vmap(hess_fun, 0, 0)(
        jnp.asarray(first_halfoct, dtype=jnp.int32)
    )
    # generate the function for each scale
    surf_fun = Partial(
        _usurf_local_2d,
        im,
        haar_filt_mult=desc_haar_scale_nstds,
        window_scale_mult=desc_window_scale_nstds,
        normalize_responses=normalize_responses,
        threshold=extrema_threshold,
        z_score=extrema_zscore,
        n_subwindow_per_dim=desc_subwins_per_dim,
        n_pts_per_dim=desc_npts_per_dim,
        max_pts=max_pts_in_octave,
    )
    v = []
    for lobes in strided_batch(rest_lobes, hoct_rd + 1):
        mid_lobe = lobes[(hoct_rd + 1) // 2]
        crop_size = mid_lobe * 2
        prev_scales, vecs = surf_fun(
            prev_scales,
            jnp.asarray(lobes, dtype=jnp.int32),
            crop_size,
            mid_lobe,
        )
        vecs = vecs.at[:, 0].multiply(mid_lobe * (1.3 / 2))
        v.append(vecs)
    return jnp.concatenate(v, axis=0)


def _usurf_local_2d(
    im: IntegralImage,
    prev_scale: Volume,
    lobe_sizes: Int[Array, " n"],
    crop_size: int,
    mid_lobe: int,
    haar_filt_mult: float = 2.0,
    window_scale_mult: float = 20.0,
    normalize_responses: bool = True,
    threshold: float = 6.0,
    z_score: bool = True,
    n_subwindow_per_dim: int = 4,
    n_pts_per_dim: int = 5,
    max_pts: int = 10,
) -> Tuple[Image, Float[Array, "m 67"]]:
    sigma = mid_lobe * (1.2 / 3)
    prev_scale, coords = _detect_keypoints_local_2d(
        im,
        prev_scale,
        lobe_sizes,
        crop_size,
        normalize_responses,
        threshold,
        z_score,
        max_pts,
    )
    haar_filt_size = jnp.around(sigma * haar_filt_mult, decimals=0).astype(
        jnp.int32
    )
    window_size = _descriptor_window_size_2d(
        sigma, window_scale_mult, n_subwindow_per_dim
    )
    chunk_size = window_size // n_subwindow_per_dim

    def describe(coord: Coord2D):
        swin = _split_descriptor_window_2d(
            _extract_descriptor_window_2d(im, window_size, coord), chunk_size
        )
        descs = jax.vmap(
            Partial(
                _subwindow_descriptor_2d,
                haar_filt_size=haar_filt_size,
                n_pts_per_dim=n_pts_per_dim,
            ),
            0,
            0,
        )(swin)
        return descs.flatten()[: 4 * n_subwindow_per_dim**2]

    vecs = jnp.stack(
        [describe(coords[i, 1:]) for i in range(coords.shape[0])], axis=0
    )
    out = jnp.concatenate([coords, vecs], axis=1)
    return prev_scale, out[out[:, 0] > 0, :]


def _descriptor_window_size_2d(
    sigma: float,
    window_scale_mult: float = 20.0,
    n_subwindow_per_dim: int = 4,
) -> int:
    window_size = (
        jnp.around(window_scale_mult * sigma, decimals=0).astype(jnp.int32)
        // n_subwindow_per_dim
        * n_subwindow_per_dim
    )
    return window_size


def _extract_descriptor_window_2d(
    im: IntegralImage,
    window_size: int,
    coord: Coord2D,
    orientation: float = 0,
) -> Float[Array, "{window_size} {window_size}"]:
    top_left = jnp.around(coord - window_size // 2, decimals=0).astype(
        jnp.int32
    )
    top_left = jnp.where(top_left > 0, top_left, 0)
    bot_right = top_left + window_size
    return im[top_left[0] : bot_right[0], top_left[1] : bot_right[1]]


def _split_descriptor_window_2d(im: Image, chunk_size: int) -> Volume:
    steps = jnp.arange(0, im.shape[-1], chunk_size)
    coords = jnp.stack(jnp.meshgrid(steps, steps), axis=-1).reshape(-1, 2)
    map_fun = Partial(
        jax.lax.dynamic_slice, im, slice_sizes=[chunk_size, chunk_size]
    )
    return jax.vmap(map_fun, 0, 0)(coords)


def _subwindow_descriptor_2d(
    im: IntegralImage, haar_filt_size: int, n_pts_per_dim: int = 5
) -> Float[Array, " 4"]:
    haar_resp = haar_response_2d(im, haar_filt_size)
    mid = im.shape[0] // 2
    stp = im.shape[0] // (n_pts_per_dim + 2)
    pts = jnp.arange(-n_pts_per_dim // 2, n_pts_per_dim // 2 + 1) * stp + mid
    coords = jnp.stack(jnp.meshgrid(pts, pts), axis=-1).reshape(-1, 2)

    def map_fun(coord: Coord2D) -> Float[Array, " 4"]:
        dx = haar_resp[0, coord[0], coord[1]]
        dy = haar_resp[1, coord[0], coord[1]]
        return jnp.asarray([dx, dy, jnp.abs(dx), jnp.abs(dy)])

    return jnp.sum(jax.vmap(map_fun, 0, 0)(coords), axis=0)
