"""Bundle a real run as the app's demo pack ("Try the demo"), small enough to commit.

``off-my-shelf make-demo packs/<run>`` copies the pack's JSON and every file it references into
``src/off_my_shelf/demo_pack``, with photos downscaled so each file stays well under 1 MB.
"""

import shutil
from pathlib import Path

from PIL import Image

from off_my_shelf.imaging import encode
from off_my_shelf.mocks import MOCK_PACK_DIR, PACK_FILENAME, load_pack, save_pack
from off_my_shelf.schema import ListingPack

DEMO_PACK_DIR = Path(__file__).parent / "demo_pack"
DEMO_MAX_SIDE = 900
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def demo_pack_dir() -> Path:
    """The bundled real-run demo when there is one, else the synthetic mock pack."""
    return DEMO_PACK_DIR if (DEMO_PACK_DIR / PACK_FILENAME).is_file() else MOCK_PACK_DIR


def referenced_files(pack: ListingPack) -> set[str]:
    files = set(pack.media)
    for item in pack.items:
        files |= {item.detection.crop_path, *item.product_shots}
        files |= {path for path in (item.detection.mask_path, item.ad_video) if path}
    return files


def make_demo(pack_dir: Path, out: Path = DEMO_PACK_DIR) -> list[Path]:
    """Replace ``out`` with a compact copy of the pack; returns the files written."""
    pack = load_pack(pack_dir)
    shutil.rmtree(out, ignore_errors=True)
    save_pack(out, pack)
    written = [out / PACK_FILENAME]
    for rel in sorted(referenced_files(pack)):
        source, target = pack_dir / rel, out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() in PHOTO_SUFFIXES and not rel.startswith("masks/"):
            with Image.open(source) as image:
                image.thumbnail((DEMO_MAX_SIDE, DEMO_MAX_SIDE))
                png = source.suffix.lower() == ".png"
                target.write_bytes(encode(image, "PNG") if png else encode(image, quality=82))
        else:
            shutil.copyfile(source, target)
        written.append(target)
    return written
