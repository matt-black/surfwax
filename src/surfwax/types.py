"""Useful types for SURF."""

from typing import Union

from jaxtyping import Array, Bool, Float, Int

# convenience types for images & volumes
Image = Union[Int[Array, "r c"], Float[Array, "r c"]]
Volume = Union[Int[Array, "r c"], Float[Array, "z r c"]]
ImageOrVolume = Union[Image, Volume]
# for readability, provide IntegralImage
IntegralImage = Image
IntegralVolume = Volume
IntegralImageOrVolume = Union[IntegralImage, IntegralVolume]

# convenience types for coordinates
Coord2D = Int[Array, " 2"]
Coord3D = Int[Array, " 3"]
Coord = Union[Coord2D, Coord3D]

# scale spaces
ImageScaleSpace = Float[Array, "n r c"]
VolumeScaleSpace = Float[Array, "n1 n2 z r c"]
ScaleSpace = Union[ImageScaleSpace, VolumeScaleSpace]
# thresholded scale spaces
ThresholdedScaleSpace = Union[Bool[Array, "n r c"], Bool[Array, "n1 n2 z r c"]]

# octave datatypes
ImageOctave = Float[Array, "n r c"]
VolumeOctave = Float[Array, "n1 n2 z r c"]
Octave = Union[ImageOctave, VolumeOctave]
