from functools import partial
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int

from ._util import strided_batch
from .extrema import find_threshold_extrema
from .haar import haar_response_3d
from .hessian import hessian_determinant_3d
from .types import Coord3D, IntegralVolume, Volume

__all__ = ["upright_surf_3d"]


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def _detect_keypoints_local_3d(
    vol: IntegralVolume,
    prev_scale: Volume,
    lobe_sizes: Int[Array, " n"],
    spatial_scale: float,  # add sigma
    volumetric_scale: float,  # add rho
    crop_size: int,
    normalize_responses: bool = True,
    threshold: float = 6,
    z_score: bool = True,
    max_pts: int = 10,
) -> Tuple[Volume, Int[Array, "n 5"]]:
    # full-size response scale space
    def per_volume_fn(vol, spatial_scale, volumetric_scale):
        def per_lobe_fn(lobe):
            return hessian_determinant_3d(
                vol,
                volumetric_scale,
                spatial_scale,
                lobe,
                normalize=normalize_responses,
            )

        return jax.vmap(per_lobe_fn)(lobe_sizes)

    # nested vmap over (vol, spatial_scale, volumetric_scale)
    ss = jax.vmap(per_volume_fn)(vol, spatial_scale, volumetric_scale)
    ss = jnp.concatenate([jnp.broadcast_to(prev_scale, ss.shape), ss], axis=1)
    prev_scale = ss[:, -lobe_sizes.shape[0] + 1 :, ...].copy()
    volsize_at_filt = [
        vol.shape[1] - 2 * crop_size,
        vol.shape[2] - 2 * crop_size,
        vol.shape[3] - 2 * crop_size,
    ]
    slice_fn = Partial(
        jax.lax.dynamic_slice,
        start_indices=[crop_size, crop_size, crop_size],
        slice_sizes=volsize_at_filt,
    )
    ss = jax.vmap(lambda s: jax.vmap(slice_fn)(s))(ss)

    def find_per_batch_extrema(scaled_vols):
        extt = find_threshold_extrema(
            scaled_vols, threshold, z_score, scaled_vols.shape[0]
        )
        coords = jnp.stack(
            jnp.nonzero(extt, size=max_pts, fill_value=0), axis=-1
        )
        coords = coords.at[:, 1:].add(crop_size)
        return coords

    coords = jax.vmap(find_per_batch_extrema)(ss)
    return prev_scale, coords


def upright_surf_3d(
    vol: IntegralVolume,
    lobe_sizes: Sequence[int],
    octave_size: int,
    spatial_scale: float = 1.0,  # set sigma
    volumetric_scale: float = 1.0,  # set rho
    # localization
    normalize_responses: bool = True,
    extrema_threshold: float = 6.0,
    extrema_zscore: bool = True,
    # descriptor args
    desc_haar_scale_nstds: float = 2.0,
    desc_window_scale_nstds: float = 20.0,
    desc_subwins_per_dim: Sequence[int] = (4, 4, 3),  # M, M, N
    desc_npts_per_dim: int = 5,
    max_pts_in_octave: int = 10,
    orientation_invariant: bool = True,
) -> Float[Array, "m d"]:
    """Find and compute descriptors for 3D U-SURF keypoints of the input volume.

    Args:
        vol (Volume): input 3D volume.
        lobe_sizes (Sequence[int]): lobe size of the filter to use at each scale.
        octave_size (int): number of scales per octave.
        spatial_scale (float): scale factor for XY plane
        volumetric_scale (float): scale factor for Z axis
        normalize_responses (bool): normalize the filter responses to the size of the filter.
        extrema_threshold (float): threshold strength for calling points as candidate extrema.
        extrema_zscore (bool): whether the threshold is absolute or z-scored.
        desc_haar_scale_nstds (float): scale of haar filters for descriptors.
        desc_window_scale_nstds (float): number of std. dev's of scale for descriptor window.
        desc_subwins_per_dim (Sequence[int]): number of subwindows in (x,y,z) for descriptor.
        desc_npts_per_dim (int): number of points in each subwindow for descriptor.
        max_pts_in_octave (int): maximum points per octave.

    Returns:
        Array: 3D U-SURF keypoints and descriptors.
            rows: keypoints
            cols: [scale_spatial, scale_volumetric, z, y, x, ...descriptor...]
    """
    if octave_size % 2 == 0:
        raise ValueError("octave_size must be an odd number")
    hoct_rd = octave_size // 2
    if len(lobe_sizes) % (hoct_rd + 1) == 0:
        raise ValueError("# of lobe sizes must evenly divide octave_size//2+1")

    first_halfoct, rest_lobes = lobe_sizes[:hoct_rd], lobe_sizes[hoct_rd:]
    # pre-compute the first 1/2 of the first octave
    hess_fun = Partial(
        hessian_determinant_3d,
        vol,
        spatial_scale,
        volumetric_scale,
        normalize=True,
    )
    prev_scales = jax.vmap(hess_fun, 0, 0)(
        jnp.asarray(first_halfoct, dtype=jnp.int32)
    )

    # generate the function for each scale
    surf_fun = Partial(
        _usurf_local_3d,
        vol,
        spatial_scale,
        volumetric_scale,
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
        vecs = vecs.at[:, 0].multiply(mid_lobe * (1.3 / 2))  # spatial scale
        vecs = vecs.at[:, 1].multiply(mid_lobe * (1.3 / 2))  # volumetric scale
        v.append(vecs)
    return jnp.concatenate(v, axis=0)


# processes a batch of scales to detect keypoints and compute descriptors
def _usurf_local_3d(
    vol: IntegralVolume,
    prev_scale: Volume,
    lobe_sizes: Int[Array, " n"],
    crop_size: int,
    mid_lobe: int,
    spatial_scale: float = 1.0,
    volumetric_scale: float = 1.0,
    haar_filt_mult: float = 2.0,
    window_scale_mult: float = 20.0,
    normalize_responses: bool = True,
    threshold: float = 6.0,
    z_score: bool = True,
    n_subwindow_per_dim: Sequence[int] = (4, 4, 3),
    n_pts_per_dim: int = 5,
    max_pts: int = 10,
) -> Tuple[Volume, Float[Array, "m d"]]:
    sigma = mid_lobe * (1.2 / 3)
    # finds keypoints
    prev_scale, coords = _detect_keypoints_local_3d(
        vol,
        prev_scale,
        lobe_sizes,
        spatial_scale,
        volumetric_scale,
        crop_size,
        normalize_responses,
        threshold,
        z_score,
        max_pts,
    )

    haar_filt_size = jnp.around(sigma * haar_filt_mult, decimals=0).astype(
        jnp.int32
    )
    window_size = _descriptor_window_size_3d(
        sigma, window_scale_mult, n_subwindow_per_dim
    )
    chunk_size = (
        window_size[0] // n_subwindow_per_dim[0],
        window_size[1] // n_subwindow_per_dim[1],
        window_size[2] // n_subwindow_per_dim[2],
    )

    # extracts a descriptor window and splits it into subregions
    def describe(coord: Coord3D):
        swin = _split_descriptor_window_3d(
            _extract_descriptor_window_3d(vol, window_size, coord), chunk_size
        )
        descs = jax.vmap(
            Partial(
                _subwindow_descriptor_3d,
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


def _descriptor_window_size_3d(
    sigma: float,
    window_scale_mult: float = 20.0,
    n_subwindow_per_dim: Sequence[int] = (4, 4, 3),  # MxMxN
) -> Tuple[int, int, int]:
    base_size = jnp.around(window_scale_mult * sigma).astype(jnp.int32)
    return (  # check this
        base_size // n_subwindow_per_dim[0] * n_subwindow_per_dim[0],
        base_size // n_subwindow_per_dim[1] * n_subwindow_per_dim[1],
        base_size // n_subwindow_per_dim[2] * n_subwindow_per_dim[2],
    )


# extracts cuboid region around a keypoint
def _extract_descriptor_window_3d(
    vol: IntegralVolume,
    window_size: Tuple[int, int, int],
    coord: Coord3D,
    orientation: Optional[Float[Array, "3"]] = None,
) -> Float[Array, "{wz} {wy} {wx}"]:
    top_left = jnp.around(
        coord - jnp.array(window_size) // 2, decimals=0
    ).astype(jnp.int32)
    top_left = jnp.where(top_left > 0, top_left, 0)
    bot_right = top_left + jnp.array(window_size)
    return vol[
        top_left[0] : bot_right[0],
        top_left[1] : bot_right[1],
        top_left[2] : bot_right[2],
    ]


# divides the descriptor window into a 3D grid of subregions
def _split_descriptor_window_3d(
    vol: Volume, chunk_size: Tuple[int, int, int]
) -> Float[Array, "n_subwindows {cz} {cy} {cx}"]:
    steps_z = jnp.arange(0, vol.shape[0], chunk_size[0])
    steps_y = jnp.arange(0, vol.shape[1], chunk_size[1])
    steps_x = jnp.arange(0, vol.shape[2], chunk_size[2])
    coords = jnp.stack(
        jnp.meshgrid(steps_z, steps_y, steps_x), axis=-1
    ).reshape(-1, 3)
    map_fun = Partial(jax.lax.dynamic_slice, vol, slice_sizes=chunk_size)
    return jax.vmap(map_fun, 0, 0)(coords)


# computes haar wavelet responses
def _subwindow_descriptor_3d(
    vol: IntegralVolume, haar_filt_size: int, n_pts_per_dim: int = 5
) -> Float[Array, " 6"]:
    haar_resp = haar_response_3d(vol, haar_filt_size)
    mid_z, mid_y, mid_x = (
        vol.shape[0] // 2,
        vol.shape[1] // 2,
        vol.shape[2] // 2,
    )
    stp_z = vol.shape[0] // (n_pts_per_dim + 2)
    stp_y = vol.shape[1] // (n_pts_per_dim + 2)
    stp_x = vol.shape[2] // (n_pts_per_dim + 2)

    pts_z = (
        jnp.arange(-n_pts_per_dim // 2, n_pts_per_dim // 2 + 1) * stp_z + mid_z
    )
    pts_y = (
        jnp.arange(-n_pts_per_dim // 2, n_pts_per_dim // 2 + 1) * stp_y + mid_y
    )
    pts_x = (
        jnp.arange(-n_pts_per_dim // 2, n_pts_per_dim // 2 + 1) * stp_x + mid_x
    )

    coords = jnp.stack(jnp.meshgrid(pts_z, pts_y, pts_x), axis=-1).reshape(
        -1, 3
    )

    def map_fun(coord: Coord3D) -> Float[Array, " 6"]:
        dx = haar_resp[0, coord[0], coord[1], coord[2]]
        dy = haar_resp[1, coord[0], coord[1], coord[2]]
        dz = haar_resp[2, coord[0], coord[1], coord[2]]
        return jnp.asarray([dx, dy, dz, jnp.abs(dx), jnp.abs(dy), jnp.abs(dz)])

    return jnp.sum(jax.vmap(map_fun, 0, 0)(coords), axis=0)
