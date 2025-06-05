import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from .boxfilter import haarx2, haarx3, haary2, haary3, haarz3
from .types import IntegralImage, IntegralVolume


def haar_response_2d(
    im: IntegralImage, filt_size: int
) -> Float[Array, "2 r c"]:
    """haar_response Compute the responses to a Haar wavelet of specified size for the input image.

    Args:
        im (IntegralImage): input image.
        filt_size (int): size of the Haar filter (in the filtering direction).

    Returns:
        Float[Array, "2 r c"]: x, y responses at each pixel.
    """
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


def haar_response_3d(
    vol: IntegralVolume, filt_size: int
) -> Float[Array, "2 d h w"]:
    """haar_response Compute the responses to a 3D Haar wavelet of specified size for the input volume.

    Args:
        vol (IntegralVolume): input volume.
        filt_size (int): size of the Haar filter (in the filtering direction).

    Returns:
        Float[Array, "3 d h w"]: dx, dy, dx responses at each voxel
    """

    coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(0, s) for s in vol.shape], indexing="ij"),
        axis=-1,
    ).reshape(
        -1, 3
    )  # [num_voxels, 3]
    dx = (jax.vmap(Partial(haarx3, vol, filt_size), 0, 0)(coords)).reshape(
        vol.shape
    )
    dy = (jax.vmap(Partial(haary3, vol, filt_size), 0, 0)(coords)).reshape(
        vol.shape
    )
    dz = (jax.vmap(Partial(haarz3, vol, filt_size), 0, 0)(coords)).reshape(
        vol.shape
    )
    return jnp.stack([dx, dy, dz], axis=0) / (filt_size**3)
