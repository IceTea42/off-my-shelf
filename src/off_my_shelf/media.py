"""Input media: collect photos and videos, and turn each into normalised still images.

Photos (JPEG, PNG, WebP, HEIC) are loaded upright and downscaled. Videos are sampled about
once per second and the sharpest, well-spaced frames are kept, so a slow pan over a shelf
becomes a few clear stills.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

from off_my_shelf.imaging import load_image

register_heif_opener()

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".webm", ".avi"})


class UnsupportedMediaError(ValueError):
    """A file is neither a supported photo nor a supported video."""


@dataclass(frozen=True, slots=True)
class Still:
    """One image to analyse, with where it came from."""

    source: Path
    image: Image.Image
    frame_index: int | None = None
    timestamp_s: float | None = None

    @property
    def stem(self) -> str:
        suffix = "" if self.frame_index is None else f"-f{self.frame_index:05d}"
        return f"{self.source.stem}{suffix}"


def collect_inputs(paths: Iterable[Path]) -> list[Path]:
    """Expand directories (non-recursive) and keep supported files, sorted by name."""
    files: list[Path] = []
    for path in paths:
        candidates = sorted(path.iterdir()) if path.is_dir() else [path]
        files.extend(p for p in candidates if p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES)
    return files


def stills(path: Path, max_side: int, max_frames: int = 6) -> list[Still]:
    """The stills to analyse for one input file."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return [Still(path, load_image(path, max_side))]
    if suffix in VIDEO_SUFFIXES:
        return list(_video_stills(path, max_side, max_frames))
    raise UnsupportedMediaError(f"unsupported file type: {path.name}")


def _sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _video_stills(path: Path, max_side: int, max_frames: int) -> Iterator[Still]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise UnsupportedMediaError(f"cannot open video: {path.name}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps))
        scored: list[tuple[float, int, np.ndarray]] = []
        for index in range(0, max(total, 1), step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            scored.append((_sharpness(frame), index, frame))
    finally:
        capture.release()

    min_gap = max(step, total // (max_frames * 2) if total else step)
    chosen: list[tuple[int, np.ndarray]] = []
    for _, index, frame in sorted(scored, key=lambda item: item[0], reverse=True):
        if all(abs(index - other) >= min_gap for other, _ in chosen):
            chosen.append((index, frame))
        if len(chosen) == max_frames:
            break
    for index, frame in sorted(chosen, key=lambda item: item[0]):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        yield Still(path, image, frame_index=index, timestamp_s=round(index / fps, 2))
