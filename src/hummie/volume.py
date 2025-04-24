from typing import Optional, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Int

from .types import IntegralVolume


def _maximum_filt_size_for_dim(dim: int):
    curr_filt_size = 9
    while curr_filt_size < dim:
        curr_filt_size += 6
    return curr_filt_size


def lobe_size_to_sigma(lobe_size: int):
    return 1.2 * lobe_size / 3


def detect_keypoints(
    vol: IntegralVolume,
    lobe_sizes_z: Optional[Sequence[int]],
    lobe_sizes_rc: Optional[Sequence[int]],
    normalize_responses: bool = True,
    crop_responses: bool = True,
    threshold: float = 6.0,
    z_score: bool = True,
) -> Int[Array, "n 2"]:
    if lobe_sizes_rc is None:
        mfs = _maximum_filt_size_for_dim(min(vol.shape[1], vol.shape[2]))
        lobe_sizes_rc = list(range(3, mfs // 3 + 2, 2))
    if lobe_sizes_z is None:
        mfs = _maximum_filt_size_for_dim(vol.shape[0])
        lobe_sizes_z = list(range(3, mfs // 3 + 2, 2))
    # scale_sigmas_rc = list(map(lobe_size_to_sigma, lobe_sizes_rc))
    # scale_sigmas_z = list(map(lobe_size_to_sigma, lobe_sizes_z))
    # make grid of lobes to compute over
    lobe_sizes_z = jnp.array(lobe_sizes_z, dtype=jnp.int32)
    lobe_sizes_rc = jnp.array(lobe_sizes_rc, dtype=jnp.int32)
    fpars = jnp.stack(
        jnp.meshgrid(lobe_sizes_z, lobe_sizes_rc, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    lsz, lsrc = fpars[:, 0][:, None], fpars[:, 1][:, None]
    # TODO: finish this, return proper pars
    return lsz, lsrc
