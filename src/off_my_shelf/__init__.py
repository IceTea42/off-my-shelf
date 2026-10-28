"""Off My Shelf: turn photos or short videos of unused items into resale listings."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("off-my-shelf")
except PackageNotFoundError:  # run from the source tree without install (Hugging Face Space)
    __version__ = "0.0.0+source"
