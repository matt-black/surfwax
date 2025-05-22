"""Boxfiltering functions & integral images.

Box filters are convolutional filters made up of rectangular boxes. In SURF, they are used to approximate the Laplacian of Gaussian filter. In conjunction with integral images, they enable fast computation of derivatives.

2D implementation is based on that described in [1].
3D implementation and references are taken from [2].

References
---
[1] Viola, P., Jones, M.: Rapid object detection using a boosted cascade of simple features. In: CVPR (1). (2001) 511-518.
[2] Ke, Yan, Rahul Sukthankar, and Martial Hebert. "Efficient visual event detection using volumetric features." Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1. Vol. 1. IEEE, 2005.
"""

from collections.abc import Callable
from numbers import Number
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from .types import (
    Coord2D,
    Coord3D,
    Image,
    ImageOrVolume,
    IntegralImage,
    IntegralImageOrVolume,
    IntegralVolume,
    Volume,
)


def integral_array(x: ImageOrVolume) -> IntegralImageOrVolume:
    """Compute integral representation of an image or volume.

    Args:
        x (ImageOrVolume): image or volume.

    Raises:
        ValueError: if input is not 2- or 3D.

    Returns:
        ImageOrVolume: integral representation of the input array.
    """
    ndim = len(x.shape)
    if ndim == 2:
        return integral_image(x)
    elif ndim == 3:
        return integral_volume(x)
    else:
        raise ValueError(
            "invalid array size, input must be 2- or 3-dimensional"
        )


def integral_image(x: Image) -> IntegralImage:
    """Compute the integral image representation of the input image.

    Args:
        x (Image): input image

    Returns:
        Image: integral image.
    """
    return jnp.cumsum(jnp.cumsum(x, 0), 1)


def integral_volume(x: Volume) -> IntegralVolume:
    """Compute the integral volume representation of the input volume.

    Args:
        x (Volume): input volume.

    Returns:
        Volume: integral volume.
    """
    return jnp.cumsum(jnp.cumsum(jnp.cumsum(x, 0), 1), 2)


def separable_boxlog(
    x: Array,
    kernel_sizes: Tuple[int, int],
    axes: Tuple[int, int],
    normalize: bool = False,
) -> Array:
    """Convolve input array with a box filter that approximates a LoG filter.

    Convolution is done by breaking the LoG-approximating box filter into separable components and convolving these components separately along each axis.

    Args:
        x (Array): array
        kernel_sizes (Tuple[int,int]):
        axes (Tuple[int,int]): axes to have (pseudo-)derivative calculated for.
        normalize (bool, optional): normalize the output response by size of kernel. Defaults to False.

    Raises:
        ValueError: if kernel_size is less than 9 or (size - 9) is not evenly divisible by 6.

    Returns:
        Array
    """
    # check that the size of the filter is a valid one
    # figure out size of structures in the filter
    if axes[0] == axes[1]:
        f1, f2 = _sep_boxlog_filtpair(kernel_sizes[0])
    else:
        f1, f2 = _sep_boxlog_filtpair(kernel_sizes)
    for ax, filt in zip(axes, [f1, f2]):
        x = jnp.apply_along_axis(_separable_conv, ax, x, filt)
    if normalize:
        wgt = 1.0 / (f1.shape[0] * f2.shape[0])
        return wgt * x
    else:
        return x


def _sep_boxlog_filtpair(kernel_size: int | Tuple[int, int]):
    if isinstance(kernel_size, int):
        lobe_size = kernel_size // 3
        n = (kernel_size - 9) // 6
        lobe_width = 5 + 4 * n
        pad_width = (kernel_size - lobe_width) // 2
        one = jnp.ones((lobe_size,))
        f1 = jnp.concatenate([one, -2 * one, one], axis=0)
        pad = jnp.zeros((pad_width,))
        one = jnp.ones((kernel_size - pad_width * 2,))
        f2 = jnp.concatenate([pad, one, pad], axis=0)
    else:
        lobe_size = [ks // 3 for ks in kernel_size]
        ns = [(ks - 9) // 6 for ks in kernel_size]
        lobe_width = [5 + 4 * n for n in ns]
        pad_width = [(ks - lw) // 2 for ks, lw in zip(kernel_size, lobe_width)]
        opw = [
            (ks - (2 * ls + 1)) // 2 for ks, ls in zip(kernel_size, lobe_size)
        ]
        f1 = jnp.concatenate(
            [
                jnp.zeros((opw[0],)),
                jnp.ones((lobe_size[0],)),
                jnp.zeros((1,)),
                jnp.ones((lobe_size[0],)) * -1,
                jnp.zeros((opw[0],)),
            ],
            axis=0,
        )
        f2 = jnp.concatenate(
            [
                jnp.zeros((opw[1],)),
                jnp.ones((lobe_size[1],)),
                jnp.zeros((1,)),
                jnp.ones((lobe_size[1],)) * -1,
                jnp.zeros((opw[1],)),
            ],
            axis=0,
        )
    return f1, f2


def _separable_conv(
    x: Float[Array, " s"], filter: Float[Array, " t"]
) -> Float[Array, " s"]:
    # IMPORTANT: need to flip the filter because what jax calls a convolution is actually a correlation.
    # convolution is just correlation with the filter mirrored, which is why the filter is flipped here.
    return jax.lax.conv_general_dilated(
        x[None, None, :], filter[None, None, ::-1], (1,), padding="same"
    )[0, 0, :]


def box_filters_2d(size: int) -> Float[Array, "3 {size} {size}"]:
    """Generate box filters for approximating LoG.

    Args:
        size (int): size of the filter, must be 9 + 6n where n is an integer.

    Raises:
        ValueError: if invalid size specified

    Returns:
        Float[Array, "3 {size} {size}"]: filter bank [lxx, lyy, lxy]
    """
    if size < 9:
        raise ValueError(f"size must be >=9, you specified {size}")
    if (size - 9) % 6 != 0:
        raise ValueError("size-9 must be evenly divisible by 6")
    # figure out dims of the directional filter (xx, yy)
    lobe_size = size // 3
    n = (size - 9) // 6
    lobe_width = 5 + 4 * n
    pad_width = (size - lobe_width) // 2
    pad = jnp.zeros((size, pad_width))
    one = jnp.ones((lobe_size, lobe_width))
    two = jnp.ones((lobe_size, lobe_width)) * 2
    # construct direction lxx, lyy filters
    lyy = jnp.concatenate(
        [pad, jnp.concatenate([one, -two, one], axis=0), pad], axis=1
    )
    lxx = jnp.transpose(lyy)
    # make diagonal filter
    one = jnp.ones((lobe_size, lobe_size))
    opw = (size - (2 * lobe_size + 1)) // 2
    lxy = jnp.zeros((size, size))
    slc1 = slice(opw, opw + lobe_size)
    slc2 = slice(opw + lobe_size + 1, opw + 2 * lobe_size + 1)
    lxy = lxy.at[slc1, slc1].set(one)
    lxy = lxy.at[slc1, slc2].set(-one)
    lxy = lxy.at[slc2, slc1].set(-one)
    lxy = lxy.at[slc2, slc2].set(one)
    return jnp.stack([lxx, lyy, lxy], axis=0)


# implementation in terms of indexing
def _try_get_im(im: Image, r: int, c: int) -> Number:
    too_small = jnp.logical_or(r < 0, c < 0)
    too_large = jnp.logical_or(r >= im.shape[0], c >= im.shape[1])
    return jnp.where(jnp.logical_or(too_small, too_large), 0, im[r, c])


def _try_get_vol(v: Volume, z: int, r: int, c: int) -> Number:
    okay_z = jnp.logical_and(z >= 0, z < v.shape[0])
    okay_r = jnp.logical_and(r >= 0, r < v.shape[1])
    okay_c = jnp.logical_and(c >= 0, c < v.shape[2])
    okay = jnp.logical_and(jnp.logical_and(okay_r, okay_c), okay_z)
    return jnp.where(okay, v[z, r, c], 0)


def _boxsum2d_from_abcd(
    im: IntegralImage, a: Coord2D, b: Coord2D, c: Coord2D, d: Coord2D
) -> Number:
    get = Partial(_try_get_im, im)
    return get(a[0], a[1]) + get(d[0], d[1]) - get(b[0], b[1]) - get(c[0], c[1])


def _boxsum2d_from_tl(
    im: IntegralImage, lobe_size: int, top_left: Coord2D
) -> Number:
    top, lft = top_left[0], top_left[1]
    get = Partial(_try_get_im, im)
    return (
        get(top, lft)
        + get(top + lobe_size, lft + lobe_size)
        - get(top, lft + lobe_size)
        - get(top + lobe_size, lft)
    )


def _boxsum3d_from_abcdefgh(
    vol: IntegralVolume,
    a: Coord3D,
    b: Coord3D,
    c: Coord3D,
    d: Coord3D,
    e: Coord3D,
    f: Coord3D,
    g: Coord3D,
    h: Coord3D,
) -> Number:
    get = Partial(_try_get_vol, vol)
    return (
        get(e[0], e[1], e[2])
        - get(a[0], a[1], a[2])
        - get(f[0], f[1], f[2])
        - get(g[0], g[1], g[2])
        + get(b[0], b[1], b[2])
        + get(c[0], c[1], c[2])
        + get(h[0], h[1], h[2])
        - get(d[0], d[1], d[2])
    )


def _boxsum3d_from_tl(
    vol: IntegralVolume, lobe_z: int, lobe_rc: int, top_left: Coord3D
) -> Number:
    d = top_left.copy()
    b = d.at[2].add(lobe_rc)
    a = b.at[0].add(lobe_z)
    c = d.at[0].add(lobe_z)
    h = d.at[1].add(lobe_rc)
    g = c.at[1].add(lobe_rc)
    f = b.at[1].add(lobe_rc)
    e = a.at[1].add(lobe_rc)
    return _boxsum3d_from_abcdefgh(vol, a, b, c, d, e, f, g, h)


def apply_filter_2d(
    filt: Callable[[IntegralImage, int, Coord2D], Number],
    im: IntegralImage,
    lobe_size: int,
) -> Image:
    """Apply the box filter to the 2D integral image.

    Args:
        filt (Callable[[IntegralImage, int, Coord2D], Number]): the filter to apply.
        im (IntegralImage): integral image.
        lobe_size (int): lobe size of the filter (2nd parameter to filter function).

    Returns:
        Image: image with filter applied.
    """
    coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(0, s) for s in im.shape], indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    fun = Partial(filt, im, lobe_size)
    return jax.vmap(fun, 0, 0)(coords).reshape(im.shape)


def apply_filter_3d(
    filt: Callable[[IntegralVolume, int, int, Coord2D], Number],
    vol: IntegralVolume,
    lobe_size_z: int,
    lobe_size_rc: int,
) -> Volume:
    """Apply the box filter to a 3D integral volume.

    Args:
        filt (Callable[[IntegralVolume, int, int, Coord2D], Number]): the filter to apply.
        vol (IntegralVolume): integral volume.
        lobe_size_z (int): lobe size of the filter in the z-direction (2nd param. to filter function).
        lobe_size_rc (int): lobe size of the filter in the rc-directions (3rd param. to filter function).

    Returns:
        Volume: volume with filter applied.
    """
    coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(0, s) for s in vol.shape], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    fun = Partial(filt, vol, lobe_size_z, lobe_size_rc)
    return jax.vmap(fun, 0, 0)(coords).reshape(vol.shape)


def dyy2(im: IntegralImage, lobe_size: int, coord: Coord2D) -> Number:
    """2D LoG filter in yy-direction, value at specified coordinate.

    Args:
        im (IntegralImage): integral image.
        lobe_size (int): lobe size of the filter.
        coord (Coord2D): coordinate to compute value at.

    Returns:
        Number: value.
    """
    boxsum = Partial(_boxsum2d_from_abcd, im)
    cr, cc = coord[0], coord[1]
    qd = lobe_size // 2
    qo = lobe_size - 1
    # left "side" of filter, a->d is moving down
    b = (cr - qd, cc - qo)
    a = (b[0] - lobe_size, b[1])
    c = (cr + qd, b[1])
    d = (c[0] + lobe_size, b[1])
    # right "side" of filter, e->h is moving down
    f = (cr - qd, cc + qo)
    e = (f[0] - lobe_size, f[1])
    g = (cr + qd, f[1])
    h = (g[0] + lobe_size, f[1])
    return boxsum(a, e, b, f) - 2 * boxsum(b, f, c, g) + boxsum(c, g, d, h)


def dxx2(im: IntegralImage, lobe_size: int, coord: Coord2D) -> Number:
    """2D LoG filter in xx-direction, value at specified coordinate.

    Args:
        im (IntegralImage): integral image.
        lobe_size (int): lobe size of the filter.
        coord (Coord2D): coordinate to compute value at.

    Returns:
        Number: value.
    """
    boxsum = Partial(_boxsum2d_from_abcd, im)
    cr, cc = coord[0], coord[1]
    qd = lobe_size // 2
    qo = lobe_size - 1
    # top "side" of filter, a->d is moving right
    b = (cr - qo, cc - qd)
    a = (b[0], b[1] - lobe_size)
    c = (b[0], cc + qd)
    d = (b[0], c[1] + lobe_size)
    # bottom "side" of filter, e->h is moving right
    f = (cr + qo, cc - qd)
    e = (f[0], f[1] - lobe_size)
    g = (f[0], cc + qd)
    h = (f[0], g[1] + lobe_size)
    return boxsum(a, b, e, f) - 2 * boxsum(b, c, f, g) + boxsum(c, d, g, h)


def dxy2(im: IntegralImage, lobe_size: int, coord: Coord2D) -> Number:
    """2D LoG filter in xy-direction, value at specified coordinate.

    Args:
        im (IntegralImage): integral image.
        lobe_size (int): lobe size of the filter.
        coord (Coord2D): coordinate to compute value at.

    Returns:
        Number: value.
    """
    cr, cc = coord[0], coord[1]
    boxsum = Partial(_boxsum2d_from_tl, im, lobe_size)
    q1tl = (cr - 1 - lobe_size, cc - 1 - lobe_size)
    q2tl = (cr - 1 - lobe_size, cc + 1)
    q3tl = (cr + 1, cc - 1 - lobe_size)
    q4tl = (cr + 1, cc + 1)
    return boxsum(q1tl) - boxsum(q2tl) - boxsum(q3tl) + boxsum(q4tl)


def dzz3(
    vol: IntegralVolume, lobe_size_z: int, lobe_size_rc: int, coord: Coord3D
) -> Number:
    """3D LoG filter in the zz-direction, value at specified coordinate.

    Args:
        vol (IntegralVolume): integral volume.
        lobe_size_z (int): lobe size of the filter in the z-direction.
        lobe_size_rc (int): lobe size of the filter in the r- and c-directions.
        coord (Coord3D): coordinate to compute value at.

    Returns:
        Number: value.
    """
    boxsum = Partial(_boxsum3d_from_abcdefgh, vol)
    cz, cr, cc = coord[0], coord[1], coord[2]
    qdz = lobe_size_z // 2
    qor, qoc = lobe_size_rc - 1, lobe_size_rc - 1
    zoff = lobe_size_z // 2 - lobe_size_z // 4
    # layout of filter:
    # front face
    # a h
    # b g
    # c f
    # d e
    b = (cz - zoff, cr + lobe_size_rc // 2, cc - lobe_size_rc // 2)
    c = (b[0] + qdz, b[1], b[2])
    a = (b[0] - qdz, b[1], b[2])
    d = (c[0] + qdz, c[1], c[2])
    e = (d[0], d[1], d[2] + qoc)
    f = (e[0] - qdz, e[1], e[2])
    g = (f[0] - qdz, f[1], f[2])
    h = (g[0] - qdz, g[1], g[2])
    # rear face (viewed from the front)
    # i p
    # j o
    # k n
    # l m
    i = (a[0], a[1] - qor, a[2])
    j = (b[0], b[1] - qor, b[2])
    k = (c[0], c[1] - qor, c[2])
    ell = (d[0], d[1] - qor, d[2])
    m = (e[0], e[1] - qor, e[2])
    n = (f[0], f[1] - qor, f[2])
    o = (g[0], g[1] - qor, g[2])
    p = (h[0], h[1] - qor, g[2])
    return (
        boxsum(o, p, j, i, g, h, b, a)
        - 2 * boxsum(n, o, k, j, f, g, c, b)
        + boxsum(m, n, ell, k, e, f, d, c)
    )


def dyy3(
    vol: IntegralVolume, lobe_size_z: int, lobe_size_rc: int, coord: Coord3D
) -> Number:
    """3D LoG filter in the yy-direction, value at specified coordinate.

    Args:
        vol (IntegralVolume): integral volume.
        lobe_size_z (int): lobe size of the filter in the z-direction.
        lobe_size_rc (int): lobe size of the filter in the r- and c-directions.
        coord (Coord3D): coordinate to compute value at.

    Returns:
        Number: value.
    """
    boxsum = Partial(_boxsum3d_from_abcdefgh, vol)
    cz, cr, cc = coord[0], coord[1], coord[2]
    qdr = lobe_size_rc // 2
    qoz, qoc = lobe_size_z - 1, lobe_size_rc - 1
    yoff = lobe_size_rc // 2 - lobe_size_rc // 4
    # layout of the filter
    # front face:
    b = (cz - lobe_size_z // 2, cr - yoff, cc - lobe_size_rc // 2)
    c = (b[0], b[1] + qdr, b[2])
    a = (b[0], b[1] - qdr, b[2])
    d = (c[0], c[1] + qdr, b[2])
    e = (d[0], d[1], d[2] + qoc)
    f = (e[0], e[1] - qdr, e[2])
    g = (f[0], f[1] - qdr, f[2])
    h = (g[0], g[1] - qdr, g[2])
    i = (a[0] + qoz, a[1], a[2])
    j = (b[0] + qoz, b[1], b[2])
    k = (c[0] + qoz, c[1], c[2])
    ell = (d[0] + qoz, d[1], d[2])
    m = (e[0] + qoz, e[1], e[2])
    n = (f[0] + qoz, f[1], f[2])
    o = (g[0] + qoz, g[1], g[2])
    p = (h[0] + qoz, h[1], h[2])
    return (
        boxsum(p, h, i, a, o, g, j, b)
        - 2 * boxsum(o, g, j, b, n, f, k, c)
        + boxsum(n, f, k, c, m, e, ell, d)
    )


def dxx3(
    vol: IntegralVolume, lobe_size_z: int, lobe_size_rc: int, coord: Coord3D
) -> Number:
    """3D LoG filter in the xx-direction, value at specified coordinate.

    Args:
        vol (IntegralVolume): integral volume.
        lobe_size_z (int): lobe size of the filter in the z-direction.
        lobe_size_rc (int): lobe size of the filter in the r- and c-directions.
        coord (Coord3D): coordinate to compute value at.

    Returns:
        Number: value.
    """
    boxsum = Partial(_boxsum3d_from_abcdefgh, vol)
    cz, cr, cc = coord[0], coord[1], coord[2]
    qdc = lobe_size_rc // 2
    qoz, qor = lobe_size_z - 1, lobe_size_rc - 1
    xoff = lobe_size_rc // 2 - lobe_size_rc // 4
    b = (cz - lobe_size_z // 2, cr - lobe_size_rc // 2, cc - xoff)
    a = (b[0], b[1], b[2] - qdc)
    c = (b[0], b[1], b[2] + qdc)
    d = (c[0], c[1], c[2] + qdc)
    e = (a[0], a[1] + qor, a[2])
    f = (b[0], b[1] + qor, b[2])
    g = (c[0], c[1] + qor, c[2])
    h = (d[0], d[1] + qor, d[2])
    i = (a[0] + qoz, a[1], a[2])
    j = (b[0] + qoz, b[1], b[2])
    k = (c[0] + qoz, c[1], c[2])
    ell = (d[0] + qoz, d[1], d[2])
    m = (e[0] + qoz, e[1], e[2])
    n = (f[0] + qoz, f[1], f[2])
    o = (g[0] + qoz, g[1], g[2])
    p = (h[0] + qoz, h[1], h[2])
    return (
        boxsum(j, b, i, a, n, f, m, e)
        - 2 * boxsum(k, c, j, b, o, g, n, f)
        + boxsum(ell, d, k, c, p, h, o, g)
    )


def dxz3(
    vol: IntegralVolume, lobe_size_z: int, lobe_size_rc: int, coord: Coord3D
) -> Number:
    """3D LoG filter in the xz-direction, value at specified coordinate.

    Args:
        vol (IntegralVolume): integral volume.
        lobe_size_z (int): lobe size of the filter in the z-direction.
        lobe_size_rc (int): lobe size of the filter in the r- and c-directions.
        coord (Coord3D): coordinate to compute value at.

    Returns:
        Number: value.
    """
    boxsum = Partial(_boxsum3d_from_tl, vol, lobe_size_z, lobe_size_rc)
    qor = lobe_size_rc // 2
    cz, cr, cc = coord[0], coord[1], coord[2]
    q0tl = (cz - 1 - lobe_size_z, cr - qor, cc - 1 - lobe_size_rc)
    q1tl = (cz - 1 - lobe_size_z, cr - qor, cc + 1)
    q2tl = (q0tl[0] + lobe_size_z + 1, q0tl[1], q0tl[2])
    q3tl = (q1tl[0] + lobe_size_z + 1, q1tl[1], q1tl[2])
    return boxsum(q0tl) - boxsum(q1tl) - boxsum(q2tl) + boxsum(q3tl)


def dyz3(
    vol: IntegralVolume, lobe_size_z: int, lobe_size_rc: int, coord: Coord3D
) -> Number:
    """3D LoG filter in the yz-direction, value at specified coordinate.

    Args:
        vol (IntegralVolume): integral volume.
        lobe_size_z (int): lobe size of the filter in the z-direction.
        lobe_size_rc (int): lobe size of the filter in the r- and c-directions.
        coord (Coord3D): coordinate to compute value at.

    Returns:
        Number: value.
    """
    boxsum = Partial(_boxsum3d_from_tl, vol, lobe_size_z, lobe_size_rc)
    qoc = lobe_size_rc // 2
    cz, cr, cc = coord[0], coord[1], coord[2]
    q0tl = (cz - 1 - lobe_size_z, cr - 1 - lobe_size_rc, cc - qoc)
    q1tl = (q0tl[0], q0tl[1] + lobe_size_rc + 1, q0tl[2])
    q2tl = (q0tl[0] + lobe_size_z + 1, q0tl[1], q0tl[2])
    q3tl = (q1tl[0] + lobe_size_z + 1, q1tl[1], q1tl[2])
    return boxsum(q0tl) - boxsum(q1tl) - boxsum(q2tl) + boxsum(q3tl)


def dxy3(
    vol: IntegralVolume, lobe_size_z: int, lobe_size_rc: int, coord: Coord3D
) -> Number:
    """3D LoG filter in the xy-direction, value at specified coordinate.

    Args:
        vol (IntegralVolume): integral volume.
        lobe_size_z (int): lobe size of the filter in the z-direction.
        lobe_size_rc (int): lobe size of the filter in the r- and c-directions.
        coord (Coord3D): coordinate to compute value at.

    Returns:
        Number: value.
    """
    boxsum = Partial(_boxsum3d_from_tl, vol, lobe_size_z, lobe_size_rc)
    qoz = lobe_size_z // 2
    cz, cr, cc = coord[0], coord[1], coord[2]
    q0tl = (cz - qoz, cr - 1 - lobe_size_rc, cc - lobe_size_rc - 1)
    q1tl = (q0tl[0], q0tl[1], q0tl[2] + lobe_size_rc + 1)
    q2tl = (q0tl[0], q0tl[1] + lobe_size_rc + 1, q0tl[2])
    q3tl = (q2tl[0], q2tl[1], q2tl[2] + lobe_size_rc + 1)
    return boxsum(q0tl) - boxsum(q1tl) - boxsum(q2tl) + boxsum(q3tl)


def haarx2(im: IntegralImage, filt_size: int, coord: Coord2D) -> Number:
    """Compute value of x-directional Haar filter at specified coordinate.

    Args:
        im (IntegralImage): integral image.
        filt_size (int): size of Haar filter (in pixels).
        coord (Coord2D): coordinate to compute value at.

    Returns:
        Number: value of x-directional Haar filter.
    """
    cr, cc = coord[0], coord[1]
    boxsum = Partial(_boxsum2d_from_abcd, im)
    lobe_size = filt_size // 2
    lsm1 = lobe_size - 1
    # layout of (a,b,c,d,e,f)
    # a   b   c
    #
    # d   e   f
    a = (cr - lobe_size, cc - lobe_size)
    b = (cr - lobe_size, cc)
    c = (cr - lobe_size, cc + lsm1)
    d = (cr + lsm1, cc - lobe_size)
    e = (cr + lsm1, cc)
    f = (cr + lsm1, cc + lsm1)
    return boxsum(a, b, d, e) - boxsum(b, c, e, f)


def haary2(im: IntegralImage, filt_size: int, coord: Coord2D) -> Number:
    """Compute value of y-directional Haar filter at specified coordinate.

    Args:
        im (IntegralImage): integral image.
        filt_size (int): size of Haar filter (in pixels).
        coord (Coord2D): coordinate to compute value at.

    Returns:
        Number: value of y-directional Haar filter.
    """
    cr, cc = coord[0], coord[1]
    boxsum = Partial(_boxsum2d_from_abcd, im)
    lobe_size = filt_size // 2
    lsm1 = lobe_size - 1
    # layout of (a,b,c,d,e,f)
    # a   b
    # c   d
    # e   f
    a = (cr - lobe_size, cc - lobe_size)
    b = (cr - lobe_size, cc + lsm1)
    c = (cr, cc - lobe_size)
    d = (cr, cc + lsm1)
    e = (cr + lobe_size, cc - lobe_size)
    f = (cr + lobe_size, cc + lsm1)
    return boxsum(a, b, c, d) - boxsum(c, d, e, f)


def boxsum3d(
    vol: IntegralVolume,
    a: Coord3D,
    b: Coord3D,
    c: Coord3D,
    d: Coord3D,
    e: Coord3D,
    f: Coord3D,
    g: Coord3D,
    h: Coord3D,
) -> Number:
    """Compute sum of 3D box with by 8 corners in an integral volume"""
    return (
        vol[h[0], h[1], h[2]]
        - vol[e[0], e[1], e[2]]
        - vol[d[0], d[1], d[2]]
        + vol[a[0], a[1], a[2]]
        - vol[g[0], g[1], g[2]]
        + vol[f[0], f[1], f[2]]
        + vol[c[0], c[1], c[2]]
        - vol[b[0], b[1], b[2]]
    )


def haarx3d(vol: IntegralVolume, filt_size: int, coord: Coord3D) -> Number:
    """Compute value of 3D x-directional Haar response (right - left) cuboid

    Args:
        vol (IntegralVolume): integral volume.
        filt_size (int): size of Haar filter (in pixels).
        coord (Coord3D): coordinate to compute value at.

    Returns:
        Number: value of x-directional 3D Haar filter.
    """
    cz, cy, cx = coord[2], coord[1], coord[0]
    boxsum = Partial(boxsum3d, vol)
    lobe_size = filt_size // 2
    lsm1 = lobe_size - 1
    # left cuboid (x - lobe_size to x)
    a = (cz - lobe_size, cy - lobe_size, cx - lobe_size)
    b = (cz - lobe_size, cy - lobe_size, cx)
    c = (cz - lobe_size, cy + lsm1, cx - lobe_size)
    d = (cz - lobe_size, cy + lsm1, cx)
    e = (cz + lsm1, cy - lobe_size, cx - lobe_size)
    f = (cz + lsm1, cy - lobe_size, cx)
    g = (cz + lsm1, cy + lsm1, cx - lobe_size)
    h = (cz + lsm1, cy + lsm1, cx)
    left = boxsum(a, b, c, d, e, f, g, h)
    # right cuboid (x to x + lobe_size)
    a = (cz - lobe_size, cy - lobe_size, cx)
    b = (cz - lobe_size, cy - lobe_size, cx + lsm1)
    c = (cz - lobe_size, cy + lsm1, cx)
    d = (cz - lobe_size, cy + lsm1, cx + lsm1)
    e = (cz + lsm1, cy - lobe_size, cx)
    f = (cz + lsm1, cy - lobe_size, cx + lsm1)
    g = (cz + lsm1, cy + lsm1, cx)
    h = (cz + lsm1, cy + lsm1, cx + lsm1)
    right = boxsum(a, b, c, d, e, f, g, h)
    return right - left


def haary3d(vol: IntegralVolume, filt_size: int, coord: Coord3D) -> Number:
    """Compute value of 3D y-directional Haar response (bottom - top) cuboid

    Args:
        vol (IntegralVolume): integral volume.
        filt_size (int): size of Haar filter (in pixels).
        coord (Coord3D): coordinate to compute value at.

    Returns:
        Number: value of y-directional 3D Haar filter.
    """
    cz, cy, cx = coord[2], coord[1], coord[0]
    boxsum = Partial(boxsum3d, vol)
    lobe_size = filt_size // 2
    lsm1 = lobe_size - 1
    # top(y - lobe_size to y)
    a = (cz - lobe_size, cy - lobe_size, cx - lobe_size)
    b = (cz - lobe_size, cy - lobe_size, cx + lsm1)
    c = (cz - lobe_size, cy, cx - lobe_size)
    d = (cz - lobe_size, cy, cx + lsm1)
    e = (cz + lsm1, cy - lobe_size, cx - lobe_size)
    f = (cz + lsm1, cy - lobe_size, cx + lsm1)
    g = (cz + lsm1, cy, cx - lobe_size)
    h = (cz + lsm1, cy, cx + lsm1)
    top = boxsum(a, b, c, d, e, f, g, h)
    # bottom(y to y + lobe_size)
    a = (cz - lobe_size, cy, cx - lobe_size)
    b = (cz - lobe_size, cy, cx + lsm1)
    c = (cz - lobe_size, cy + lobe_size, cx - lobe_size)
    d = (cz - lobe_size, cy + lobe_size, cx + lsm1)
    e = (cz + lsm1, cy, cx - lobe_size)
    f = (cz + lsm1, cy, cx + lsm1)
    g = (cz + lsm1, cy + lobe_size, cx - lobe_size)
    h = (cz + lsm1, cy + lobe_size, cx + lsm1)
    bottom = boxsum(a, b, c, d, e, f, g, h)
    return bottom - top


def haarz3d(vol: IntegralVolume, filt_size: int, coord: Coord3D) -> Number:
    """Compute value of 3D z-directional Haar response (front - back) cuboid

    Args:
        vol (IntegralVolume): integral volume.
        filt_size (int): size of Haar filter (in pixels).
        coord (Coord3D): coordinate to compute value at.

    Returns:
        Number: value of z-directional 3D Haar filter.
    """
    cz, cy, cx = coord[2], coord[1], coord[0]
    boxsum = Partial(boxsum3d, vol)
    lobe_size = filt_size // 2
    lsm1 = lobe_size - 1
    # front(z - lobe_size to z)
    a = (cz - lobe_size, cy - lobe_size, cx - lobe_size)
    b = (cz - lobe_size, cy - lobe_size, cx + lsm1)
    c = (cz - lobe_size, cy + lsm1, cx - lobe_size)
    d = (cz - lobe_size, cy + lsm1, cx + lsm1)
    e = (cz, cy - lobe_size, cx - lobe_size)
    f = (cz, cy - lobe_size, cx + lsm1)
    g = (cz, cy + lsm1, cx - lobe_size)
    h = (cz, cy + lsm1, cx + lsm1)
    front = boxsum(a, b, c, d, e, f, g, h)
    # back(z to z + lobe_size)
    a = (cz, cy - lobe_size, cx - lobe_size)
    b = (cz, cy - lobe_size, cx + lsm1)
    c = (cz, cy + lsm1, cx - lobe_size)
    d = (cz, cy + lsm1, cx + lsm1)
    e = (cz + lobe_size, cy - lobe_size, cx - lobe_size)
    f = (cz + lobe_size, cy - lobe_size, cx + lsm1)
    g = (cz + lobe_size, cy + lsm1, cx - lobe_size)
    h = (cz + lobe_size, cy + lsm1, cx + lsm1)
    back = boxsum(a, b, c, d, e, f, g, h)
    return back - front
