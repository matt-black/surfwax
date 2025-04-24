"""2D SURF

Speeded Up Robust Features (SURF) on 2D arrays (images).
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int

from .boxfilter import haarx2, haary2, integral_image
from .extrema import find_threshold_extrema
from .hessian import response_scale_space_2d
from .types import Coord2D, Image, IntegralImage, Volume


@dataclass
class SURFKeypoint2D:
    sigma: float
    location: Coord2D
    orientation: Optional[float]
    vector: Float[Array, "64"]


def _maximum_filt_size(shape: Sequence[int]):
    min_dim = min(shape)
    curr_filt_size = 9
    while curr_filt_size < min_dim:
        curr_filt_size += 6
    return curr_filt_size


def detect_keypoints(
    im: IntegralImage,
    lobe_sizes: Optional[Sequence[int]],
    normalize_responses: bool = True,
    crop_responses: bool = True,
    threshold: float = 6,
    z_score: bool = True,
) -> Int[Array, "n 2"]:
    if lobe_sizes is None:
        mls = _maximum_filt_size(im.shape)
        lobe_sizes = list(range(3, mls // 3 + 2, 2))
    scale_sigmas = [1.2 * ls / 3 for ls in lobe_sizes]
    ss = response_scale_space_2d(
        im, lobe_sizes, normalize_responses, crop_responses
    )
    offset = (im.shape[-1] - ss.shape[-1]) // 2
    extt = find_threshold_extrema(ss, threshold, z_score)
    coords = jnp.stack(jnp.nonzero(extt), axis=-1)
    sigmas = jnp.asarray(scale_sigmas)[coords[:, 0].astype(jnp.int32)]
    if crop_responses:
        rcs = coords[:, 1:] + jnp.asarray([offset, offset])[None, :]
    else:
        rcs = coords[:, 1:]
    return sigmas, rcs


def describe_keypoints(
    im: IntegralImage,
    sigmas: Float[Array, " n"],
    coords: Int[Array, "n 2"],
    ori_haar_scale_mult: float = 4.0,
    ori_sample_scale_mult: float = 6.0,
    weight_scale_mult: float = 2.5,
    ori_num_angs: int = 100,
    ori_window_width: float = math.pi / 3,
    dsc_window_scale_mult: float = 20.0,
    dsc_n_subwindow_per_dim: int = 4,
    dsc_haar_mult: float = 2.0,
    dsc_n_pts_per_dim: int = 5,
) -> Sequence[SURFKeypoint2D]:
    pfun = Partial(
        describe_keypoint,
        im,
        ori_haar_scale_mult=ori_haar_scale_mult,
        ori_sample_scale_mult=ori_sample_scale_mult,
        weight_scale_mult=weight_scale_mult,
        ori_num_angs=ori_num_angs,
        ori_window_width=ori_window_width,
        dsc_window_scale_mult=dsc_window_scale_mult,
        dsc_n_subwindow_per_dim=dsc_n_subwindow_per_dim,
        dsc_haar_mult=dsc_haar_mult,
        dsc_n_pts_per_dim=dsc_n_pts_per_dim,
    )
    n_kpts = sigmas.shape[0]
    return [pfun(sigmas[i], coords[i, :]) for i in range(n_kpts)]


def describe_keypoint(
    im: IntegralImage,
    sigma: float,
    coord: Coord2D,
    ori_haar_scale_mult: float = 4.0,
    ori_sample_scale_mult: float = 6.0,
    weight_scale_mult: float = 2.5,
    ori_num_angs: int = 100,
    ori_window_width: float = math.pi / 3,
    dsc_window_scale_mult: float = 20.0,
    dsc_n_subwindow_per_dim: int = 4,
    dsc_haar_mult: float = 2.0,
    dsc_n_pts_per_dim: int = 5,
) -> Float[Array, " n"]:
    ori = assign_orientation(
        im,
        sigma,
        coord,
        ori_haar_scale_mult,
        ori_sample_scale_mult,
        weight_scale_mult,
        ori_num_angs,
        ori_window_width,
    )
    desc = assign_descriptor(
        im,
        sigma,
        coord,
        ori,
        dsc_window_scale_mult,
        dsc_n_subwindow_per_dim,
        dsc_haar_mult,
        dsc_n_pts_per_dim,
    )
    return SURFKeypoint2D(sigma, coord, ori, desc)


def upright_surf(
    im: Image,
    # localization
    lobe_sizes: Sequence[int],
    normalize_responses: bool,
    crop_responses: bool,
    extrema_threshold: float,
    extrema_zscore: bool,
    # descriptor args
    desc_window_scale_nstds: float = 20.0,
    desc_subdims_per_dim: int = 4,
    desc_haar_scale_nstds: float = 2.0,
    desc_npts_per_dim: int = 5,
) -> Sequence[SURFKeypoint2D]:
    x = integral_image(im)
    sigmas, rcs = detect_keypoints(
        x,
        lobe_sizes,
        normalize_responses,
        crop_responses,
        extrema_threshold,
        extrema_zscore,
    )

    def describe(sigma: float, coord: Coord2D) -> Float[Array, " n"]:
        return assign_descriptor(
            x,
            sigma,
            coord,
            0,
            desc_window_scale_nstds,
            desc_subdims_per_dim,
            desc_haar_scale_nstds,
            desc_npts_per_dim,
        )

    vec = jnp.stack(
        [describe(sigmas[i], rcs[i, :]) for i in range(sigmas.shape[0])], axis=0
    )
    return jnp.concatenate([rcs, sigmas[:, None], vec], axis=1)


def haar_response(im: IntegralImage, filt_size: int) -> Float[Array, "2 r c"]:
    coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(0, s) for s in im.shape], indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    x = (jax.vmap(Partial(haarx2, im, filt_size), 0, 0)(coords)).reshape(
        im.shape
    )
    y = (jax.vmap(Partial(haary2, im, filt_size), 0, 0)(coords)).reshape(
        im.shape
    )
    return jnp.stack([x, y], axis=0) / (filt_size**2)


def windowed_haar_response(
    im: IntegralImage,
    sigma: float,
    coord: Coord2D,
    scale_mult: float = 4.0,
    sample_mult: float = 6.0,
) -> Image:
    haar_filt_size = jnp.around(sigma * scale_mult, decimals=0).astype(
        jnp.int32
    )
    # ensure filter is even-sized
    haar_filt_size = jax.lax.select(
        haar_filt_size % 2 == 0, haar_filt_size, haar_filt_size - 1
    )
    # compute haar-wavelet response
    haar_resp = haar_response(im, haar_filt_size)
    # filter haar response down to the sampling scale
    sampling_radius = jnp.floor(sample_mult * sigma).astype(jnp.int32)
    top_left = coord - sampling_radius
    top_left = jnp.where(top_left < 0, 0, top_left)
    slice_size = sampling_radius * 2 + haar_filt_size // 2
    slice = Partial(
        jax.lax.dynamic_slice,
        start_indices=top_left,
        slice_sizes=[slice_size, slice_size],
    )
    return jax.vmap(slice, 0, 0)(haar_resp), slice


def weight_array(shape: int, ndim: int, sigma: float, scale: float) -> Image:
    # weight the points
    dist_from_pt = jnp.sqrt(
        jnp.sum(
            jnp.square(
                jnp.stack(
                    jnp.meshgrid(
                        *[
                            jnp.linspace(-shape / 2, shape / 2, shape)
                            for _ in range(ndim)
                        ],
                        indexing="ij",
                    ),
                    axis=0,
                )
            ),
            axis=0,
        )
    )
    wgt = jnp.exp(-dist_from_pt / (scale * sigma))
    wgt = wgt.at[dist_from_pt > shape / 2].set(0)
    return wgt


def assign_orientation(
    im: IntegralImage,
    sigma: float,
    coord: Coord2D,
    haar_scale_mult: float = 4.0,
    sample_scale_mult: float = 6.0,
    weight_scale_mult: float = 2.5,
    num_angs: int = 30,
    window_width: float = math.pi / 3,
) -> float:
    haar, _ = windowed_haar_response(
        im, sigma, coord, haar_scale_mult, sample_scale_mult
    )
    wgt = weight_array(haar.shape[-1], 2, sigma, weight_scale_mult)
    whaar = haar * wgt[None, ...]
    nhaar = haar / jnp.hypot(haar[0], haar[1])[None, ...]

    def contrib(ori: float):
        return jax.vmap(
            Partial(_contrib, ori=ori, thresh=window_width), (0, 0)
        )(whaar.reshape(2, -1), nhaar.reshape(2, -1)).sum(axis=-1)

    vecs = jax.vmap(contrib)(
        jnp.linspace(-jnp.pi, jnp.pi, num_angs, endpoint=False)
    )
    mags = jnp.hypot(vecs[:, 0], vecs[:, 1])
    angs = jnp.arctan2(vecs[:, 1], vecs[:, 0])
    return jnp.mean(angs[mags == jnp.amax(mags)])


def _contrib(
    wval: Float[Array, " 2"],
    nval: Float[Array, " 2"],
    ori: float,
    thresh: float,
) -> float:
    rel_ang = jnp.arccos(nval[0] * jnp.cos(ori) + nval[1] * jnp.sin(ori))
    return jnp.where(rel_ang < thresh, wval, 0)


def assign_orientation_svd(
    im: IntegralImage,
    sigma: float,
    coord: Coord2D,
    haar_scale_mult: float = 4.0,
    sample_scale_mult: float = 6.0,
    weight_scale_mult: float = 2.5,
) -> float:
    haar, _ = windowed_haar_response(
        im, sigma, coord, haar_scale_mult, sample_scale_mult
    )
    wgt = weight_array(haar.shape[-1], 2, sigma, weight_scale_mult)
    whaar = haar * wgt[None, ...]
    mu = 0.0
    _, _, v = jnp.linalg.svd(whaar - mu)
    return jnp.arctan2(v[0, 1], v[0, 0])


@partial(jax.jit, static_argnums=(2,))
def ori_vector_at_angle(
    whaar: Float[Array, "2 r c"],
    nhaar: Float[Array, "2 r c"],
    center_ori: float,
    window_width: float = math.pi / 3,
) -> Float[Array, "2"]:
    xy = jnp.array([jnp.cos(center_ori), jnp.sin(center_ori)])[:, None, None]
    close = jnp.arccos(jnp.sum(nhaar * xy, axis=0)) < window_width / 2
    print(close)
    return jnp.sum(whaar[:, close].reshape(2, -1), axis=1)


def assign_descriptor(
    im: IntegralImage,
    sigma: float,
    coord: Coord2D,
    orientation: float = 0,
    window_scale_mult: float = 20,
    n_subwindow_per_dim: int = 4,
    haar_mult: float = 2.0,
    n_pts_per_dim: int = 5,
) -> Float[Array, " n"]:
    win = extract_descriptor_window(
        im, sigma, coord, orientation, window_scale_mult, n_subwindow_per_dim
    )
    swin = split_window(win, n_subwindow_per_dim)
    compute_desc = Partial(
        subwindow_descriptor,
        sigma=sigma,
        haar_mult=haar_mult,
        n_pts_per_dim=n_pts_per_dim,
    )
    return (jax.vmap(compute_desc, 0, 0)(swin)).flatten()


def extract_descriptor_window(
    im: IntegralImage,
    sigma: float,
    coord: Coord2D,
    orientation: float = 0,
    window_scale_mult: float = 20,
    n_subwindow_per_dim: int = 4,
) -> IntegralImage:
    window_size = (
        jnp.around(window_scale_mult * sigma).astype(jnp.int32)
        // n_subwindow_per_dim
        * n_subwindow_per_dim
    )
    top_left = jnp.around(coord - window_size // 2, decimals=0).astype(
        jnp.int32
    )
    top_left = jnp.where(top_left > 0, top_left, 0)
    return jax.lax.dynamic_slice(im, top_left, [window_size, window_size])


def split_window(im: IntegralImage, n_subwindow_per_dim: int = 4) -> Volume:
    dim = im.shape[-1] // n_subwindow_per_dim
    chunkx = jnp.arange(0, dim * n_subwindow_per_dim, dim)
    chunky = jnp.arange(0, dim * n_subwindow_per_dim, dim)
    chunks = []
    for cx in chunkx:
        for cy in chunky:
            chunks.append(jax.lax.dynamic_slice(im, [cx, cy], [dim, dim]))
    return jnp.stack(chunks, axis=0)


def subwindow_descriptor(
    im: IntegralImage,
    sigma: float,
    haar_mult: float = 2.0,
    n_pts_per_dim: int = 5,
) -> Array:
    haar_resp = haar_response(
        im, jnp.around(sigma * haar_mult, decimals=0).astype(jnp.int32)
    )
    mid = im.shape[0] // 2
    stp = im.shape[0] // (n_pts_per_dim + 2)
    dx, dy, dxa, dya = 0, 0, 0, 0
    for rm in range(-n_pts_per_dim // 2, n_pts_per_dim // 2 + 1):
        for cm in range(-n_pts_per_dim // 2, n_pts_per_dim // 2 + 1):
            r = mid + stp * rm
            c = mid + stp * cm
            dx += haar_resp[0, r, c]
            dxa += jnp.abs(dx)
            dy += haar_resp[1, r, c]
            dya += jnp.abs(dy)
    return jnp.asarray([dx, dy, dxa, dya])
