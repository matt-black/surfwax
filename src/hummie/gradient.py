"""Functions for calculating gradients."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Real


def gradient_at_center_pixel_2d(
    cube: Real[Array, "3 3 3"],
) -> Float[Array, " 3"]:
    """gradient_at_center_pixel_2d Compute the value of the gradient at the center pixel of the input cube.

    Args:
        cube (Real[Array, "3 3 3"]): cube of pixel values.

    Returns:
        Array: (3,1)-shaped array (dz, dy, dx)
    """
    dx = 0.5 * (cube[1, 1, 2] - cube[1, 1, 0])
    dy = 0.5 * (cube[1, 2, 1] - cube[1, 0, 1])
    dz = 0.5 * (cube[2, 1, 1] - cube[0, 1, 1])
    return jnp.asarray([dx, dy, dz])[:, None]
