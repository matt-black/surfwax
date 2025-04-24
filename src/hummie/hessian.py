"""Functions for computing fast, approximate Hessians."""

from typing import Sequence

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int, Real

from .boxfilter import dxx2, dxx3, dxy2, dxy3, dxz3, dyy2, dyy3, dyz3, dzz3
from .types import (
    Coord2D,
    Coord3D,
    Image,
    ImageScaleSpace,
    IntegralImage,
    IntegralVolume,
    Volume,
    VolumeScaleSpace,
)


def _hessian_det2(im: IntegralImage, lobe_size: int, coord: Coord2D) -> float:
    dxx = dxx2(im, lobe_size, coord)
    dyy = dyy2(im, lobe_size, coord)
    dxy = dxy2(im, lobe_size, coord)
    return jnp.multiply(dxx, dyy) - 0.81 * jnp.square(dxy)


def hessian_determinant_2d(
    im: IntegralImage, lobe_size: int, normalize: bool
) -> Image:
    """hessian_determinant_2d Compute the pixel-wise hessian determinant for a 2D integral image.

    Args:
        im (IntegralImage): integral image.
        lobe_size (int): lobe size of the filter.
        normalize (bool): normalize the values to the size of the filter.

    Returns:
        Image: (approximate) determinant of the hessian at each pixel.
    """
    coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(0, s) for s in im.shape], indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    fun = Partial(_hessian_det2, im, lobe_size)
    hd = jax.vmap(fun, 0, 0)(coords).reshape(im.shape)
    return hd / (3 * lobe_size) ** 2 if normalize else hd


def _hessian_det3(
    vol: IntegralVolume, lobe_size_z: int, lobe_size_rc: int, coord: Coord3D
) -> float:
    dxx = dxx3(vol, lobe_size_z, lobe_size_rc, coord)
    dyy = dyy3(vol, lobe_size_z, lobe_size_rc, coord)
    dzz = dzz3(vol, lobe_size_z, lobe_size_rc, coord)
    dxy = dxy3(vol, lobe_size_z, lobe_size_rc, coord)
    dxz = dxz3(vol, lobe_size_z, lobe_size_rc, coord)
    dyz = dyz3(vol, lobe_size_z, lobe_size_rc, coord)
    return (
        dxx * (dyy * dzz - jnp.square(dxy))
        + dxy * (dxz * dyz - dxy * dzz)
        + dxz * (dxy * dyz - dyy * dxz)
    )


def hessian_determinant_3d(
    vol: IntegralVolume, lobe_size_z: int, lobe_size_rc: int, normalize: bool
) -> Volume:
    """hessian_determinant_3d Compute the voxel-wise hessian determinant for a 3D integral volume.

    Args:
        vol (IntegralVolume): integral volume.
        lobe_size_z (int): lobe size of the filter in the z-direction.
        lobe_size_rc (int): lobe size of the filter in the rc-directions.
        normalize (bool): normalize the values to the size of the filter.

    Returns:
        Volume: determinant of the hessian at each voxel.
    """
    coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(0, s) for s in vol.shape], indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    fun = Partial(_hessian_det3, vol, lobe_size_z, lobe_size_rc)
    hd = jax.vmap(fun, 0, 0)(coords).reshape(vol.shape)
    if normalize:
        wgt = 1 / ((3 * lobe_size_z) * (3 * lobe_size_rc) ** 2)
        return hd * wgt
    else:
        return hd


def response_scale_space_2d(
    im: IntegralImage,
    lobe_sizes: Sequence[int] | Int[Array, " n"],
    normalize: bool = True,
    crop: bool = False,
) -> ImageScaleSpace:
    """response_scale_space_2d Compute a 3D response scale for a 2D image.

    Args:
        im (IntegralImage): integral image.
        lobe_sizes (Sequence[int] | Int[Array, "n"]): lobe size of filters for each scale.
        normalize (bool, optional): normalize the responses to the filter size. Defaults to True.
        crop (bool, optional): crop to only regions where all filters fully see the image. Defaults to False.

    Returns:
        ImageScaleSpace: scale space.
    """
    hess = Partial(hessian_determinant_2d, im, normalize=normalize)
    lobe_sizes = jnp.array(lobe_sizes, dtype=jnp.int32)[:, None]
    if crop:
        max_filt_size = jnp.max(2 * lobe_sizes)
        imsze_at_maxfilt = [
            im.shape[0] - 2 * max_filt_size,
            im.shape[1] - 2 * max_filt_size,
        ]
        slice = Partial(
            jax.lax.dynamic_slice,
            start_indices=[max_filt_size, max_filt_size],
            slice_sizes=imsze_at_maxfilt,
        )

        def calc(lobe_size: int) -> Array:
            return slice(hess(lobe_size))

    else:
        calc = hess
    return jax.vmap(calc, 0, 0)(lobe_sizes)


def response_scale_3d(
    vol: IntegralVolume,
    lobe_sizes_z: Sequence[int] | Int[Array, " n"],
    lobe_sizes_rc: Sequence[int] | Int[Array, " n"],
    normalize: bool = True,
    crop: bool = True,
) -> VolumeScaleSpace:
    """response_scale_3d Compute a 5D response scale for a 3D volume.

    Args:
        vol (IntegralVolume): integral volume.
        lobe_sizes_z (Sequence[int] | Int[Array, "n"]): lobe sizes in the z-direction for filters.
        lobe_sizes_rc (Sequence[int] | Int[Array, "n"]): lobe sizes in the rc-directions for filters.
        normalize (bool, optional): normalize the responses to the filter size. Defaults to True.
        crop (bool, optional): crop the scale to only regions where all filters fully see the image. Defaults to True.

    Returns:
        VolumeScaleSpace: scale space.
    """
    hess = Partial(hessian_determinant_3d, vol, normalize=normalize)
    lobe_sizes_rc = jnp.array(lobe_sizes_rc, dtype=jnp.int32)[:, None]
    lobe_sizes_z = jnp.array(lobe_sizes_z, dtype=jnp.int32)[:, None]
    lobe_sizes = jnp.stack(
        jnp.meshgrid(lobe_sizes_z, lobe_sizes_rc), axis=-1
    ).reshape(-1, 2)
    if crop:
        max_filt_rc = jnp.max(2 * lobe_sizes_rc)
        max_filt_z = jnp.max(2 * lobe_sizes_z)
        imsze_at_maxfilt = [
            vol.shape[0] - 2 * max_filt_z,
            vol.shape[1] - 2 * max_filt_rc,
            vol.shape[2] - 2 * max_filt_rc,
        ]
        slice = Partial(
            jax.lax.dynamic_slice,
            start_indices=[max_filt_z, max_filt_rc, max_filt_rc],
            slice_sizes=imsze_at_maxfilt,
        )

        def calc(lobe_size_z: int, lobe_size_rc: int) -> Array:
            return slice(hess(lobe_size_z, lobe_size_rc))

    else:
        calc = hess
    return jax.vmap(calc, 0, 0)(
        lobe_sizes[:, 0][:, None], lobe_sizes[:, 1][:, None]
    )


def hessian_at_center_pixel_2d(
    cube: Real[Array, "3 3 3"],
) -> Float[Array, "3 3 3"]:
    """hessian_at_center_pixel_2d Compute the value of the hessian at the center pixel of the input cube.

    Args:
        cube (Real[Array, "3 3 3"]): cube of pixel values.

    Returns:
        Array: (3,3)-shaped array
            [(dxx, dxy, dxz),
             (dxy, dyy, dyz),
             (dxz, dyz, dzz)]
    """
    cpv = cube[1, 1, 1]
    dxx = cube[1, 1, 2] - 2 * cpv + cube[1, 1, 0]
    dyy = cube[1, 2, 1] - 2 * cpv + cube[1, 0, 1]
    dzz = cube[2, 1, 1] - 2 * cpv + cube[0, 1, 1]
    dxy = 0.25 * (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0])
    dxz = 0.25 * (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0])
    dyz = 0.25 * (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1])
    return jnp.asarray([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
