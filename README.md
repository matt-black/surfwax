# surfwax

A JAX implementation of 2D and 3D SURF.

## SURF

SURF is "Speeded Up Robust Features", originally described in [1], is an algorithm for finding and characterizing scale- and rotation-invariant keypoints in images.
A good overview of the algorithm can be found on [Wikipedia](https://en.wikipedia.org/wiki/Speeded_up_robust_features).

For volumes, a SURF-like algorithm described in [2] is used.

## Documentation

Build locally with `mkdocs`. From the root directory of this repository:

```
> mkdocs serve
```

This should start a local server, accessible at [localhost:8000](http://localhost:8000).

## Dependencies

- [JAX](https://github.com/jax-ml/jax)
- [jaxtyping](https://github.com/patrick-kidger/jaxtyping)

## Development Setup

Relies on [uv](https://github.com/astral-sh/uv). After installing and cloning the repository, run:

```
> uv venv
> uv sync
> uv pip install -e .
```
Then to set up `pre-commit`:
```
> source .venv/bin/activate
> pre-commit install
```

If running on a Mac and you want to use JAX metal (untested):

```
> uv venv
> source .venv/bin/activate
(surfwax) > uv pip install numpy wheel
(surfwax) > uv pip install jax-metal
(surfwax) > uv sync
(surfwax) > uv pip install -e .
```
Then setup `pre-commit`, as described above.

## References

[1] Bay, H., Tuytelaars, T., Van Gool, L. (2006). SURF: Speeded Up Robust Features. In: Leonardis, A., Bischof, H., Pinz, A. (eds) Computer Vision – ECCV 2006. ECCV 2006. Lecture Notes in Computer Science, vol 3951. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11744023_32

[2] Willems, Geert, Tinne Tuytelaars, and Luc Van Gool. "An efficient dense and scale-invariant spatio-temporal interest point detector." Computer Vision–ECCV 2008: 10th European Conference on Computer Vision, Marseille, France, October 12-18, 2008, Proceedings, Part II 10. Springer Berlin Heidelberg, 2008.
