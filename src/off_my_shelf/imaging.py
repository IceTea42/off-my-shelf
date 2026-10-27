"""Image helpers: load, normalise, encode, crop and mask, with Pillow only."""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
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


def flatten(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """RGB image with any transparency composited onto a solid background."""
    rgba = image.convert("RGBA")
    base = Image.new("RGBA", rgba.size, (*background, 255))
    return Image.alpha_composite(base, rgba).convert("RGB")


SHOT_BACKGROUND = (235, 235, 235)
HIDDEN_COLOUR = (255, 0, 255)


def shot_input(image: Image.Image, mask: BoolMask, box: PixelBox) -> tuple[Image.Image, float]:
    """The item on a plain background, with its hidden parts painted magenta.

    Hidden = inside the item's outline (convex hull of its mask) but not visible, i.e. covered
    by another object. Returns the image and the hidden share of the outline, 0 to 1.
    """
    local = mask[box.top : box.bottom, box.left : box.right].astype(np.uint8)
    hull = np.zeros_like(local)
    points = cv2.findNonZero(local)
    if points is not None:
        cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
    hidden = (hull == 1) & (local == 0)
    pixels = np.array(crop(image.convert("RGB"), box))
    out = np.empty_like(pixels)
    out[:] = SHOT_BACKGROUND
    out[local == 1] = pixels[local == 1]
    out[hidden] = HIDDEN_COLOUR
    share = float(hidden.sum() / max(int(hull.sum()), 1))
    return Image.fromarray(out), share
