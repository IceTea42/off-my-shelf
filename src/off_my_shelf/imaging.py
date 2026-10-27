"""Image helpers: load, normalise, encode, crop and mask, with Pillow only."""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from off_my_shelf.schema import BoundingBox

ImageFormat = Literal["JPEG", "PNG", "WEBP"]
MIME: dict[ImageFormat, str] = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
BoolMask = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PixelBox:
    """Box in pixels; ``right`` and ``bottom`` are exclusive."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError(f"empty box {self}")

    @property
    def area(self) -> int:
        return (self.right - self.left) * (self.bottom - self.top)

    def iou(self, other: "PixelBox") -> float:
        width = min(self.right, other.right) - max(self.left, other.left)
        height = min(self.bottom, other.bottom) - max(self.top, other.top)
        if width <= 0 or height <= 0:
            return 0.0
        inter = width * height
        return inter / (self.area + other.area - inter)

    def padded(self, fraction: float, width: int, height: int) -> "PixelBox":
        """Grow by ``fraction`` of the box size on each side, clamped to the image."""
        dx = round((self.right - self.left) * fraction)
        dy = round((self.bottom - self.top) * fraction)
        return PixelBox(
            max(0, self.left - dx),
            max(0, self.top - dy),
            min(width, self.right + dx),
            min(height, self.bottom + dy),
        )

    def normalised(self, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            x0=self.left / width,
            y0=self.top / height,
            x1=self.right / width,
            y1=self.bottom / height,
        )


def load_image(path: Path, max_side: int) -> Image.Image:
    """RGB image with EXIF rotation applied, downscaled so its longest side is ``max_side``."""
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def encode(image: Image.Image, fmt: ImageFormat = "JPEG", quality: int = 88) -> bytes:
    buffer = io.BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buffer, fmt, quality=quality, optimize=True)
    else:
        image.save(buffer, fmt)
    return buffer.getvalue()


def data_url(data: bytes, fmt: ImageFormat) -> str:
    return f"data:{MIME[fmt]};base64,{base64.b64encode(data).decode('ascii')}"


def crop(image: Image.Image, box: PixelBox) -> Image.Image:
    return image.crop((box.left, box.top, box.right, box.bottom))


def mask_image(mask: BoolMask) -> Image.Image:
    """Binary mask as an 8-bit image: 255 inside the object, 0 outside."""
    return Image.fromarray(mask.astype(np.uint8) * 255)


def cutout(image: Image.Image, mask: BoolMask, box: PixelBox) -> Image.Image:
    """The object on a transparent background, cropped to ``box``."""
    rgba = image.convert("RGBA")
    rgba.putalpha(mask_image(mask))
    return crop(rgba, box)
