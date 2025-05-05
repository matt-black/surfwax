# `boxfilter`

Boxfiltering functions & integral images.

Box filters are convolutional filters made up of rectangular boxes. In SURF, they are used to approximate the Laplacian of Gaussian filter. In conjunction with integral images, they enable fast computation of derivatives.

2D implementation is based on that described in [1].

3D implementation and references are taken from [2].

### References
[1] Viola, P., Jones, M.: Rapid object detection using a boosted cascade of simple features. In: CVPR (1). (2001) 511-518.

[2] Ke, Yan, Rahul Sukthankar, and Martial Hebert. "Efficient visual event detection using volumetric features." Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1. Vol. 1. IEEE, 2005.

## Integral Image/Volumes

::: surfwax.boxfilter.integral_array
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.integral_image
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.integral_volume
    handler: python
    options:
        show_source: false
        show_root_heading: true

## Filtering

::: surfwax.boxfilter.separable_boxlog
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.box_filters_2d
    handler: python
    options:
        show_source: false
        show_root_heading: true


::: surfwax.boxfilter.apply_filter_2d
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.apply_filter_3d
    handler: python
    options:
        show_source: false
        show_root_heading: true

## 2D LoG Filters

::: surfwax.boxfilter.dxx2
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.dyy2
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.dxy2
    handler: python
    options:
        show_source: false
        show_root_heading: true

## 3D LoG Filters

::: surfwax.boxfilter.dxx3
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.dyy3
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.dzz3
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.dxy3
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.dxz3
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.dyz3
    handler: python
    options:
        show_source: false
        show_root_heading: true

## Haar Filters

::: surfwax.boxfilter.haarx2
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: surfwax.boxfilter.haary2
    handler: python
    options:
        show_source: false
        show_root_heading: true
