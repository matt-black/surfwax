import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from .boxfilter import haarx2, haary2
from .types import IntegralImage


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
