"""Image helpers."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from off_my_shelf.imaging import PixelBox, crop, cutout, data_url, encode, load_image, mask_image


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


def test_mask_cutout_and_crop() -> None:
    image = Image.new("RGB", (10, 10), "blue")
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    box = PixelBox(3, 2, 7, 5)
    assert mask_image(mask).getextrema() == (0, 255)
    piece = cutout(image, mask, box)
    assert piece.size == (4, 3) and piece.mode == "RGBA"
    assert piece.getchannel("A").getextrema() == (255, 255)
    assert crop(image, box).size == (4, 3)
