"""A 3D model and a 3-second 360-degree turn of one item, from SAM 3D Objects on a Modal GPU.

The item's crop and its SAM 3.1 mask go to the endpoint deployed by ``scripts/modal_sam3d.py``
(Meta's open-weight SAM 3D Objects); it returns a GLB model and an MP4 of the model turning
once. Both are saved in the pack next to the item's other files.
"""

import base64
import time
from pathlib import Path

import httpx
from PIL import Image

from off_my_shelf.config import Settings
from off_my_shelf.imaging import PixelBox, encode
from off_my_shelf.pipeline import CROP_PADDING
from off_my_shelf.schema import Item, ModelCall, PipelineStep

MODEL = "sam-3d-objects"
SECONDS = 3.0
GPU_USD_PER_SECOND = 1.95 / 3600  # Modal L40S list price
TIMEOUT_S = 15 * 60.0  # a cold start loads the model first


class TurntableError(RuntimeError):
    """The 3D endpoint is not set up or did not answer with a model."""


def item_mask(pack_dir: Path, item: Item, size: tuple[int, int]) -> Image.Image:
    """The item's mask cut to the same padded box as its crop, at the crop's size."""
    if item.detection.mask_path is None:
        raise TurntableError(f"{item.item_id} has no mask to build its 3D model from")
    with Image.open(pack_dir / item.detection.mask_path) as full:
        mask = full.convert("L")
    bbox = item.detection.bbox
    box = PixelBox(
        round(bbox.x0 * mask.width),
        round(bbox.y0 * mask.height),
        round(bbox.x1 * mask.width),
        round(bbox.y1 * mask.height),
    ).padded(CROP_PADDING, mask.width, mask.height)
    cut = mask.crop((box.left, box.top, box.right, box.bottom))
    return cut.resize(size, Image.Resampling.NEAREST)


async def make_turntable(settings: Settings, pack_dir: Path, item: Item) -> ModelCall:
    """Write ``turns/<item_id>.mp4`` and ``models/<item_id>.glb`` and set them on the item."""
    if not settings.modal_sam3d_url or settings.modal_sam3d_key is None:
        raise TurntableError("3D is not set up (MODAL_SAM3D_URL and MODAL_SAM3D_KEY)")
    with Image.open(pack_dir / item.detection.crop_path) as raw:
        photo = raw.convert("RGB")
    payload = {
        "image": base64.b64encode(encode(photo, "PNG")).decode("ascii"),
        "mask": base64.b64encode(encode(item_mask(pack_dir, item, photo.size), "PNG")).decode(),
        "seconds": SECONDS,
    }
    headers = {"Authorization": f"Bearer {settings.modal_sam3d_key.get_secret_value()}"}
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        response = await client.post(settings.modal_sam3d_url, json=payload, headers=headers)
    if response.status_code != httpx.codes.OK:
        raise TurntableError(f"3D endpoint answered {response.status_code}: {response.text[:200]}")
    seconds = time.perf_counter() - start
    result = response.json()
    item.turntable_video = f"turns/{item.item_id}.mp4"
    item.model_3d = f"models/{item.item_id}.glb"
    for rel, data in ((item.turntable_video, result["video"]), (item.model_3d, result["model"])):
        (pack_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (pack_dir / rel).write_bytes(base64.b64decode(data))
    return ModelCall(
        step=PipelineStep.TURNTABLE,
        model=MODEL,
        cache_key=f"turntable:{item.item_id}",
        cost_usd=round(seconds * GPU_USD_PER_SECOND, 4),
        latency_ms=round(seconds * 1000),
        provider="modal",
    )
