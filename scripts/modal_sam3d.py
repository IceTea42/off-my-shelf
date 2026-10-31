"""SAM 3D Objects (Meta open weights) on a Modal GPU: one item photo + its mask -> 3D model.

Returns the item as a GLB model and a short 360-degree turn (H.264 MP4) rendered from the model's
Gaussian splat. The app calls it from the item card ("360° turn"); nothing runs until then.

Once:    accept the licence at https://huggingface.co/facebook/sam-3d-objects, then put a read
         token (HF_TOKEN) and a key of your choice (MODAL_SAM3D_KEY) in ``.env``
Deploy:  uv run modal deploy scripts/modal_sam3d.py   (first build: ~30 min, compiles PyTorch3D)
Then:    MODAL_SAM3D_URL=<the printed ...-sam3d-turntable.modal.run URL> in ``.env``

The image follows the repo's own setup (doc/setup.md): CUDA 12.1, PyTorch 2.5.1, Python 3.11.
The checkpoints download once into the Modal Volume ``sam3d-weights``.
"""

import base64
import io
import os
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import modal
import numpy as np
from fastapi import Header, HTTPException
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/facebookresearch/sam-3d-objects"
COMMIT = "f91db411c50efee93d8db7aeb323885650f6f722"
WEIGHTS = Path("/weights")
CONFIG = WEIGHTS / "hf" / "pipeline.yaml"
GPU = "L40S"  # 48 GB; the model needs 32 GB of VRAM
FPS = 24
FLASH_ATTN = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libegl1")
    .env(
        {
            "CONDA_PREFIX": "/usr/local/cuda",  # the repo's notebook code reads CUDA_HOME from it
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
            "FORCE_CUDA": "1",
            "LIDRA_SKIP_INIT": "true",
            "PIP_EXTRA_INDEX_URL": "https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121",
        }
    )
    .run_commands(f"git clone {REPO} /sam3d && git -C /sam3d checkout {COMMIT}")
    .run_commands("cd /sam3d && pip install -e '.[dev]'")
    .run_commands(f"pip install {FLASH_ATTN}", "cd /sam3d && pip install -e '.[p3d]'", gpu=GPU)
    .run_commands(
        "cd /sam3d && PIP_FIND_LINKS=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/"
        "torch-2.5.1_cu121.html pip install -e '.[inference]'",
        "cd /sam3d && python patching/hydra",
        gpu=GPU,
    )
    .pip_install("fastapi[standard]", "huggingface-hub<1.0")
)
weights = modal.Volume.from_name("sam3d-weights", create_if_missing=True)
app = modal.App("off-my-shelf-sam3d")


class TurntableRequest(BaseModel):
    """Item photo and its mask (base64 PNG or JPEG, same size), and the turn's length."""

    image: str
    mask: str
    seconds: float = 3.0


def _download_weights() -> None:
    """Fetch the gated checkpoints into the Volume once (needs HF_TOKEN with licence accepted)."""
    if CONFIG.is_file():
        return
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - container only

    staging = Path(snapshot_download("facebook/sam-3d-objects", local_dir=WEIGHTS / "hf-download"))
    shutil.move(str(staging / "checkpoints"), str(CONFIG.parent))
    shutil.rmtree(staging, ignore_errors=True)
    weights.commit()


def _decode(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


def _mp4(frames: list[np.ndarray]) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "turn.mp4"  # +faststart rewrites the file, so not in memory
        with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
            stream = container.add_stream("libx264", rate=FPS)
            stream.height, stream.width = frames[0].shape[:2]
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, FPS)
            stream.options = {"preset": "veryfast", "crf": "20"}
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        return path.read_bytes()


@app.cls(
    image=image,
    gpu=GPU,
    volumes={str(WEIGHTS): weights},
    secrets=[modal.Secret.from_dotenv(ROOT)],
    scaledown_window=5 * 60,
    timeout=10 * 60,
    max_containers=1,
)
class Sam3D:
    @modal.enter()
    def load(self) -> None:
        _download_weights()
        sys.path.insert(0, "/sam3d/notebook")
        from inference import Inference  # type: ignore[import-not-found]  # noqa: PLC0415

        self.model = Inference(str(CONFIG), compile=False)

    @modal.fastapi_endpoint(method="POST")
    def turntable(
        self, request: TurntableRequest, authorization: str = Header("")
    ) -> dict[str, str]:
        if authorization != f"Bearer {os.environ['MODAL_SAM3D_KEY']}":
            raise HTTPException(401, "wrong key")
        from inference import (  # noqa: PLC0415
            make_scene,
            ready_gaussian_for_video_rendering,
            render_video,
        )

        photo = np.asarray(_decode(request.image).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(_decode(request.mask).convert("L")) > 127  # noqa: PLR2004
        output = self.model(photo, mask, seed=42)
        scene = ready_gaussian_for_video_rendering(make_scene(output))
        frames = render_video(
            scene,
            r=1,
            fov=60,
            pitch_deg=15,
            yaw_start_deg=-45,
            resolution=512,
            num_frames=round(FPS * request.seconds),
            bg_color=(1, 1, 1),
        )["color"]
        model = io.BytesIO()
        output["glb"].export(model, file_type="glb")
        return {
            "video": base64.b64encode(_mp4(frames)).decode("ascii"),
            "model": base64.b64encode(model.getvalue()).decode("ascii"),
        }
