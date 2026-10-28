"""A 10-second vertical video ad per item, built from its real product shot.

Every frame shows the photo's own pixels (a slow zoom on the product shot), with the title, the
price and a one-line hook written by Muse Spark on top. No video model is involved, so the item in
the ad is exactly the item for sale. Encoded as H.264 MP4 with PyAV, 1080x1920, ready for
Reels, Shorts, TikTok or a WhatsApp status.
"""

import asyncio
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from off_my_shelf.schema import Item, ModelCall
from off_my_shelf.spark import SparkAnalyzer

AD_SIZE = (1080, 1920)
FPS = 30
SECONDS = 10
ZOOM = 0.12  # the shot grows by 12 % over the ad
MARGIN = 72
FONT_DIR = Path(__file__).parent / "fonts"
INK, MUTED, ACCENT, PAPER = (29, 28, 26), (107, 103, 95), (31, 122, 77), (246, 244, 239)


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


def _overlay(title: str, price: str, hook: str | None) -> Image.Image:
    """The still parts of the ad: background, hook, title, price and brand line."""
    width, height = AD_SIZE
    card = Image.new("RGB", AD_SIZE, PAPER)
    draw = ImageDraw.Draw(card)
    text_width = width - 2 * MARGIN
    if hook:
        y = 150
        for line in _wrap(hook, _font(64, bold=True), text_width, 2):
            draw.text((MARGIN, y), line, font=_font(64, bold=True), fill=INK)
            y += 80
    y = 1380
    for line in _wrap(title, _font(56), text_width, 2):
        draw.text((MARGIN, y), line, font=_font(56), fill=INK)
        y += 72
    price_font = _font(96, bold=True)
    left, top, right, bottom = draw.textbbox((0, 0), price, font=price_font)
    pill = (MARGIN, y + 40, MARGIN + right - left + 80, y + 40 + bottom - top + 60)
    draw.rounded_rectangle(pill, radius=40, fill=ACCENT)
    draw.text((pill[0] + 40 - left, pill[1] + 30 - top), price, font=price_font, fill="white")
    draw.text((MARGIN, height - 130), "Off My Shelf", font=_font(40, bold=True), fill=MUTED)
    return card


def _frames(shot: Image.Image, card: Image.Image) -> Iterator[Image.Image]:
    """The shot zooming slowly inside a square window between the hook and the title."""
    window = AD_SIZE[0] - 2 * MARGIN
    top = 360
    base = shot.convert("RGB")
    base.thumbnail((window, window))
    total = FPS * SECONDS
    for index in range(total):
        scale = 1.0 + ZOOM * index / (total - 1)
        size = (round(base.width * scale), round(base.height * scale))
        zoomed = base.resize(size, Image.Resampling.BILINEAR)
        left = (zoomed.width - min(zoomed.width, window)) // 2
        upper = (zoomed.height - min(zoomed.height, window)) // 2
        view = zoomed.crop(
            (left, upper, left + min(zoomed.width, window), upper + min(zoomed.height, window))
        )
        frame = card.copy()
        frame.paste(view, ((AD_SIZE[0] - view.width) // 2, top + (window - view.height) // 2))
        yield frame


def render_ad(shot: Image.Image, title: str, price: str, hook: str | None, out: Path) -> Path:
    """Write the MP4 ad for one item and return its path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    card = _overlay(title, price, hook)
    with av.open(str(out), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = AD_SIZE
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, FPS)
        stream.options = {"preset": "veryfast", "crf": "23"}
        for frame in _frames(shot, card):
            picture = av.VideoFrame.from_ndarray(np.asarray(frame), format="rgb24")
            for packet in stream.encode(picture):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return out


def price_label(item: Item) -> str:
    price = item.analysis.price
    unit = "€" if price.currency == "EUR" else price.currency
    return f"{price.suggested:.0f} {unit}"


async def make_item_ad(
    spark: SparkAnalyzer | None, pack_dir: Path, item: Item
) -> tuple[str, ModelCall | None]:
    """Render ``ads/<item_id>.mp4`` for one item; the hook is skipped when ``spark`` is None."""
    hook, call = None, None
    if spark is not None:
        result = await spark.ad_hook(item.analysis.listing)
        hook, call = result.value.hook, result.call
    with Image.open(pack_dir / (item.product_shots or [item.detection.crop_path])[0]) as shot:
        image = shot.convert("RGB")
    rel = f"ads/{item.item_id}.mp4"
    title = item.analysis.listing.title
    await asyncio.to_thread(render_ad, image, title, price_label(item), hook, pack_dir / rel)
    return rel, call
