"""Image helpers."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from off_my_shelf.imaging import (
    HIDDEN_COLOUR,
    SHOT_BACKGROUND,
    PixelBox,
    crop,
    data_url,
    encode,
    flatten,
    load_image,
    mask_image,
    shot_input,
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


def test_shot_input_marks_hidden_parts() -> None:
    image = Image.new("RGB", (10, 10), "blue")
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    mask[4:6, 4:6] = False  # a leaf covers the middle of the plate
    box = PixelBox(1, 1, 9, 9)
    source, share = shot_input(image, mask, box)
    assert source.size == (8, 8) and source.mode == "RGB"
    assert source.getpixel((4, 4)) == HIDDEN_COLOUR
    assert source.getpixel((2, 2)) == (0, 0, 255)
    assert source.getpixel((0, 0)) == SHOT_BACKGROUND
    assert share == pytest.approx(4 / 36)
    assert mask_image(mask).getextrema() == (0, 255)
    assert crop(image, box).size == (8, 8)


def test_flatten_removes_transparency() -> None:
    clear = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    assert flatten(clear, (1, 2, 3)).getpixel((0, 0)) == (1, 2, 3)
