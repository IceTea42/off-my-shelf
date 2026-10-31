"""A 5-second vertical "shelf to studio" clip per item, built from the photo's own pixels.

The clip tells what the app did: the real photo of the shelf, then everything but the item
dims while its SAM 3.1 outline lights up, the camera pushes in on the item, and it lands as
its clean product shot on a studio card with its title and a hook line written by Muse Spark.
No video model and no price: the item in the clip is exactly the item for sale. Encoded as
H.264 MP4 with PyAV, 1080x1920, ready for Reels, Shorts, TikTok or a WhatsApp status.
"""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from off_my_shelf.schema import Item, ModelCall
from off_my_shelf.spark import SparkAnalyzer

AD_SIZE = (1080, 1920)
FPS = 30
SECONDS = 5.0
MARGIN = 72
WINDOW_TOP = 360  # the square picture window sits between the hook and the title
# Phase ends, as fractions of the clip: photo, spotlight, push-in, studio card
PHOTO_END, SPOTLIGHT_END, PUSH_END = 0.18, 0.4, 0.68
DIM = 0.3  # how bright the rest of the photo stays under the spotlight
FONT_DIR = Path(__file__).parent / "fonts"
INK, MUTED, PAPER = (29, 28, 26), (107, 103, 95), (246, 244, 239)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(FONT_DIR / name, size)


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and font.getlength(f"{lines[-1]} {word}") <= width:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    if len(lines) > max_lines:
        lines = [*lines[: max_lines - 1], " ".join(lines[max_lines - 1 :])]
        while font.getlength(lines[-1] + "…") > width and " " in lines[-1]:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] += "…"
    return lines


def _ease(x: float) -> float:
    """Smooth start and stop (cubic ease-in-out) on 0..1."""
    x = min(max(x, 0.0), 1.0)
    return 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2  # noqa: PLR2004


def _phase(t: float, start: float, end: float) -> float:
    return _ease((t - start) / (end - start))


@dataclass(frozen=True, slots=True)
class Scene:
    """What the clip is made of, all from the run: photo, item mask, product shot, words."""

    photo: Image.Image
    mask: np.ndarray  # bool, photo size
    shot: Image.Image
    title: str
    hook: str | None


def _text_layer(scene: Scene) -> Image.Image:
    """Hook (top), title and brand line (bottom) on a transparent layer."""
    layer = Image.new("RGBA", AD_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width = AD_SIZE[0] - 2 * MARGIN
    if scene.hook:
        y = 150
        for line in _wrap(scene.hook, _font(64, bold=True), width, 2):
            draw.text((MARGIN, y), line, font=_font(64, bold=True), fill=INK)
            y += 80
    y = WINDOW_TOP + width + 70
    for line in _wrap(scene.title, _font(56), width, 3):
        draw.text((MARGIN, y), line, font=_font(56), fill=INK)
        y += 72
    draw.text((MARGIN, AD_SIZE[1] - 130), "Off My Shelf", font=_font(40, bold=True), fill=MUTED)
    return layer


def _square(box: tuple[float, float, float, float], w: int, h: int) -> np.ndarray:
    """The smallest square around a box, kept inside the photo where it fits."""
    left, top, right, bottom = box
    side = min(max(right - left, bottom - top), w, h)
    cx = min(max((left + right) / 2, side / 2), w - side / 2)
    cy = min(max((top + bottom) / 2, side / 2), h - side / 2)
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])


def _frames(scene: Scene) -> Iterator[Image.Image]:
    window = AD_SIZE[0] - 2 * MARGIN
    photo = np.asarray(scene.photo.convert("RGB"), dtype=np.float32)
    h, w = photo.shape[:2]
    soft = cv2.GaussianBlur(scene.mask.astype(np.float32), (0, 0), max(w, h) / 400)[..., None]
    kernel = np.ones((max(3, w // 150),) * 2, np.uint8)
    edge = cv2.dilate(scene.mask.astype(np.uint8), kernel).astype(bool) & ~scene.mask
    paper = np.array(PAPER, dtype=np.float32)
    ys, xs = np.nonzero(scene.mask)
    pad = 0.12 * max(np.ptp(xs), np.ptp(ys))
    start = _square((0, 0, w, h), w, h)
    end = _square((xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad), w, h)
    shot = scene.shot.convert("RGB")
    shot.thumbnail((window, window))
    studio = Image.new("RGB", (window, window), PAPER)
    studio.paste(shot, ((window - shot.width) // 2, (window - shot.height) // 2))
    text = _text_layer(scene)
    total = round(FPS * SECONDS)
    for index in range(total):
        t = index / (total - 1)
        light = _phase(t, PHOTO_END, SPOTLIGHT_END)
        push = _phase(t, SPOTLIGHT_END, PUSH_END)
        land = _phase(t, PUSH_END, PUSH_END + 0.12)
        # the rest of the photo dims, then fades to the studio paper as the camera moves in
        rest = photo * (1 - (1 - DIM) * light) * (1 - push) + paper * push * light
        frame = soft * photo + (1 - soft) * rest
        frame[edge] = frame[edge] * (1 - light) + 255 * light  # the SAM outline lights up
        left, top, right, bottom = start + (end - start) * push
        view = Image.fromarray(frame.clip(0, 255).astype(np.uint8)).resize(
            (window, window), Image.Resampling.BILINEAR, box=(left, top, right, bottom)
        )
        picture = Image.blend(view, studio, land) if land else view
        canvas = Image.new("RGB", AD_SIZE, PAPER)
        canvas.paste(picture, (MARGIN, WINDOW_TOP))
        words = text.copy()
        words.putalpha(text.getchannel("A").point(lambda a, k=land: round(a * k)))
        canvas.paste(words, (0, 0), words)
        yield canvas


def render_ad(scene: Scene, out: Path) -> Path:
    """Write the MP4 clip for one item and return its path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = AD_SIZE
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, FPS)
        stream.options = {"preset": "veryfast", "crf": "23"}
        for frame in _frames(scene):
            picture = av.VideoFrame.from_ndarray(np.asarray(frame), format="rgb24")
            for packet in stream.encode(picture):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return out


def load_scene(pack_dir: Path, item: Item, hook: str | None) -> Scene:
    """The item's photo (the still it was found in), its mask and its product shot."""
    detection = item.detection
    with Image.open(pack_dir / detection.source.media) as raw:
        photo = raw.convert("RGB")
    if detection.mask_path is not None:
        with Image.open(pack_dir / detection.mask_path) as raw:
            mask = np.asarray(raw.convert("L").resize(photo.size)) > 127  # noqa: PLR2004
    else:  # no mask saved: light up the item's box
        mask = np.zeros((photo.height, photo.width), bool)
        b = detection.bbox
        mask[
            round(b.y0 * photo.height) : round(b.y1 * photo.height),
            round(b.x0 * photo.width) : round(b.x1 * photo.width),
        ] = True
    with Image.open(pack_dir / (item.product_shots or [detection.crop_path])[0]) as raw:
        shot = raw.convert("RGB")
    return Scene(photo, mask, shot, item.analysis.listing.title, hook)


async def make_item_ad(
    spark: SparkAnalyzer | None, pack_dir: Path, item: Item
) -> tuple[str, ModelCall | None]:
    """Render ``ads/<item_id>.mp4`` for one item; the hook is skipped when ``spark`` is None."""
    hook, call = None, None
    if spark is not None:
        result = await spark.ad_hook(item.analysis.listing)
        hook, call = result.value.hook, result.call
    rel = f"ads/{item.item_id}.mp4"
    scene = load_scene(pack_dir, item, hook)
    await asyncio.to_thread(render_ad, scene, pack_dir / rel)
    return rel, call
