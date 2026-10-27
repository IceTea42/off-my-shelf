"""Image helpers."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from off_my_shelf.imaging import (
    HIDDEN_COLOUR,
    PixelBox,
    crop,
    data_url,
    encode,
    load_image,
    mask_image,
    pixel_lock,
    studio_canvas,
)


def test_pixel_box_geometry() -> None:
    a = PixelBox(0, 0, 10, 10)
    b = PixelBox(5, 5, 15, 15)
    assert a.area == 100
    assert a.iou(b) == pytest.approx(25 / 175)
    assert a.iou(PixelBox(20, 20, 30, 30)) == 0.0
    assert a.padded(0.5, width=12, height=100) == PixelBox(0, 0, 12, 15)
    assert a.normalised(20, 40).model_dump() == {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.25}
    with pytest.raises(ValueError, match="empty box"):
        PixelBox(5, 5, 5, 10)


def test_load_image_downscales_and_converts(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    Image.new("RGBA", (4000, 2000), (10, 20, 30, 255)).save(path)
    image = load_image(path, max_side=1000)
    assert image.size == (1000, 500)
    assert image.mode == "RGB"


def test_encode_and_data_url() -> None:
    data = encode(Image.new("RGB", (8, 8), "red"), "PNG")
    assert data.startswith(b"\x89PNG")
    assert data_url(data, "PNG").startswith("data:image/png;base64,iVBOR")


def _plate() -> tuple[Image.Image, np.ndarray, PixelBox]:
    image = Image.new("RGB", (40, 40), (0, 0, 200))
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    mask[16:24, 16:24] = False  # a leaf covers the middle of the plate
    return image, mask, PixelBox(10, 10, 30, 30)


def test_studio_canvas_uses_real_pixels_and_marks_hidden_parts() -> None:
    image, mask, box = _plate()
    canvas = studio_canvas(image, mask, box, size=100, margin=0.1)
    assert canvas.real.size == (100, 100) and canvas.real.mode == "RGB"
    assert canvas.hidden_share == pytest.approx(64 / 400)
    assert canvas.real.getpixel((15, 15)) == (0, 0, 200)  # real item pixels
    assert canvas.real.getpixel((50, 50)) == (255, 255, 255)  # hidden part left white
    assert canvas.to_fill.getpixel((50, 50)) == HIDDEN_COLOUR
    assert canvas.real.getpixel((2, 2)) == (255, 255, 255)  # background
    assert canvas.visible.sum() + canvas.hidden.sum() == pytest.approx(80 * 80, rel=0.05)
    assert mask_image(mask).getextrema() == (0, 255)
    assert crop(image, box).size == (20, 20)


def test_pixel_lock_restores_visible_pixels_and_measures_drift() -> None:
    image, mask, box = _plate()
    canvas = studio_canvas(image, mask, box, size=100)
    faithful = canvas.to_fill.copy()
    faithful.paste((10, 10, 190), (40, 40, 60, 60))  # model fills the hole
    locked, drift = pixel_lock(faithful, canvas)
    assert drift == pytest.approx(0.0, abs=0.01)
    assert locked.getpixel((50, 50)) == (10, 10, 190)  # filled hidden part kept
    assert locked.getpixel((15, 15)) == (0, 0, 200)  # visible part is the real photo

    redrawn = Image.new("RGB", (512, 512), (200, 120, 40))  # a different object, other size
    locked, drift = pixel_lock(redrawn, canvas)
    assert drift > 0.5
    assert locked.size == (100, 100) and locked.getpixel((15, 15)) == (0, 0, 200)
