"""Miscellaneous small utility functions, internal to library."""

from typing import Sequence


def strided_batch(vals: Sequence, n: int, stride: int = 1):
    """Yield batches of values from `vals` of specified length (`n`), stepping through the sequence of `vals` with some `stride`.

    Args:
        vals (Sequence): values to yield from.
        n (int): number of values in a single batch.
        stride (int, optional): step size. Defaults to 1.

    Raises:
        ValueError: if n < 1

    Yields:
        Tuple
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    for i in range(0, len(vals) - n, stride):
        yield tuple(vals[i : i + n])
