# hummie

A JAX implementation of 2D and 3D SURF.

## SURF

SURF is "Speeded Up Robust Features", originally described in [1], is an algorithm for finding and characterizing scale- and rotation-invariant keypoints in images.
A good overview of the algorithm can be found on [Wikipedia](https://en.wikipedia.org/wiki/Speeded_up_robust_features).

For volumes, a SURF-like algorithm described in [2] is used.

## Documentation

Coming soon...

## Dependencies

- [JAX](https://github.com/jax-ml/jax)
- [jaxtyping](https://github.com/patrick-kidger/jaxtyping)

## References

[1] Bay, H., Tuytelaars, T., Van Gool, L. (2006). SURF: Speeded Up Robust Features. In: Leonardis, A., Bischof, H., Pinz, A. (eds) Computer Vision – ECCV 2006. ECCV 2006. Lecture Notes in Computer Science, vol 3951. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11744023_32

[2] Willems, Geert, Tinne Tuytelaars, and Luc Van Gool. "An efficient dense and scale-invariant spatio-temporal interest point detector." Computer Vision–ECCV 2008: 10th European Conference on Computer Vision, Marseille, France, October 12-18, 2008, Proceedings, Part II 10. Springer Berlin Heidelberg, 2008.
