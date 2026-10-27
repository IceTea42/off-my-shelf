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


HIDDEN_COLOUR = (255, 0, 255)
STUDIO_WHITE = (255, 255, 255)


@dataclass(frozen=True, slots=True)
class StudioCanvas:
    """An item placed on a square white canvas, built only from the photo's real pixels."""

    real: Image.Image
    """The item on white with a soft shadow; hidden parts are left white."""
    to_fill: Image.Image
    """The same canvas with hidden parts painted magenta, for an image model to complete."""
    visible: BoolMask
    hidden: BoolMask
    hidden_share: float
    """Hidden share of the item's outline (convex hull of its mask), 0 to 1."""


def studio_canvas(
    image: Image.Image, mask: BoolMask, box: PixelBox, size: int = 1024, margin: float = 0.1
) -> StudioCanvas:
    """Centre the masked item on a white square, scaled to fill it, with a soft floor shadow.

    Hidden = inside the item's convex hull but not visible, i.e. covered by another object.
    """
    local = mask[box.top : box.bottom, box.left : box.right].astype(np.uint8)
    hull = np.zeros_like(local)
    points = cv2.findNonZero(local)
    if points is not None:
        cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
    hidden_local = (hull == 1) & (local == 0)
    share = float(hidden_local.sum() / max(int(hull.sum()), 1))

    scale = size * (1 - 2 * margin) / max(box.right - box.left, box.bottom - box.top)
    width = max(1, round((box.right - box.left) * scale))
    height = max(1, round((box.bottom - box.top) * scale))
    left, top = (size - width) // 2, (size - height) // 2

    def place(layer: np.ndarray) -> BoolMask:
        resized = cv2.resize(
            layer.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
        out = np.zeros((size, size), dtype=bool)
        out[top : top + height, left : left + width] = resized.astype(bool)
        return out

    visible, hidden = place(local), place(hidden_local)
    pixels = crop(image.convert("RGB"), box).resize((width, height), Image.Resampling.LANCZOS)
    item = np.zeros((size, size, 3), dtype=np.uint8)
    item[top : top + height, left : left + width] = np.array(pixels)

    canvas = np.array(_shadow(size, left, top + height, width), dtype=np.uint8)
    canvas[visible] = item[visible]
    to_fill = canvas.copy()
    to_fill[hidden] = HIDDEN_COLOUR
    return StudioCanvas(Image.fromarray(canvas), Image.fromarray(to_fill), visible, hidden, share)


def _shadow(size: int, left: int, bottom: int, width: int) -> Image.Image:
    """A white square with a soft elliptical shadow under where the item stands."""
    shade = np.zeros((size, size), dtype=np.uint8)
    centre = (left + width // 2, min(size - 1, bottom))
    axes = (max(1, int(width * 0.42)), max(1, int(size * 0.025)))
    cv2.ellipse(shade, centre, axes, 0, 0, 360, 70, -1)
    soft = cv2.GaussianBlur(shade, (0, 0), size * 0.015)
    white = np.full((size, size, 3), STUDIO_WHITE, dtype=np.int16)
    return Image.fromarray((white - np.asarray(soft)[..., None]).clip(0, 255).astype(np.uint8))


def pixel_lock(generated: Image.Image, canvas: StudioCanvas) -> tuple[Image.Image, float]:
    """Keep the model's pixels only where the item was hidden; restore every visible pixel.

    Returns the locked image and the drift: mean absolute difference (0 to 1) between the
    generated and the real pixels on the visible part. A high drift means the model moved or
    redrew the item, so its completion cannot be trusted.
    """
    size = canvas.real.size
    out = np.array(generated.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    real = np.array(canvas.real)
    drift = 1.0
    if canvas.visible.any():
        diff = np.abs(out.astype(np.int16) - real.astype(np.int16))
        drift = float(diff[canvas.visible].mean() / 255)
    out[canvas.visible] = real[canvas.visible]
    return Image.fromarray(out), drift
