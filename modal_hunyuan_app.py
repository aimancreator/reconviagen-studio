"""Protected Modal API for the official Hunyuan3D-2mv multiview pipeline.

The browser UI remains local. This deployment contains only model preloading,
GPU inference, and a protected health endpoint.
"""

from __future__ import annotations

import modal

APP_NAME = "hunyuan3d-2mv"
GPU = "H100"
CACHE_ROOT = "/cache"
HF_CACHE_DIR = "/cache/huggingface"
REMBG_CACHE_DIR = "/cache/rembg"
SOURCE_DIR = "/opt/Hunyuan3D-2"
SOURCE_COMMIT = "f8db63096c8282cb27354314d896feba5ba6ff8a"

SHAPE_REPOSITORY = "tencent/Hunyuan3D-2mv"
SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-mv"
PAINT_REPOSITORY = "tencent/Hunyuan3D-2"
PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"
DELIGHT_SUBFOLDER = "hunyuan3d-delight-v2-0"
VIEW_ORDER = ("front", "left", "back", "right")

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("hunyuan3d-2mv-model-cache", create_if_missing=True)

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": "8",
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "U2NET_HOME": REMBG_CACHE_DIR,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OPENCV_IO_ENABLE_OPENEXR": "1",
        }
    )
    .apt_install(
        "build-essential",
        "cmake",
        "ffmpeg",
        "git",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libx11-6",
        "libxext6",
        "ninja-build",
    )
    .run_commands(
        (
            "python -m pip install --upgrade pip setuptools==75.8.0 wheel "
            "packaging ninja pybind11"
        ),
        (
            "python -m pip install torch==2.5.1 torchvision==0.20.1 "
            "--index-url https://download.pytorch.org/whl/cu124"
        ),
        (
            "python -m pip install numpy==1.26.4 pillow==11.1.0 "
            "diffusers==0.33.1 transformers==4.49.0 accelerate==1.3.0 "
            "huggingface_hub[hf_xet]==0.33.4 safetensors==0.5.3 "
            "einops==0.8.1 omegaconf==2.3.0 pyyaml==6.0.2 tqdm==4.67.1 "
            "opencv-python-headless==4.11.0.86 scipy==1.15.2 scikit-image==0.25.2 "
            "trimesh==4.6.8 pymeshlab==2023.12.post3 pygltflib==1.16.3 "
            "xatlas==0.0.10 rembg==2.0.67 onnxruntime==1.21.0 "
            "fastapi==0.115.12 python-multipart==0.0.20"
        ),
    )
    .run_commands(
        (
            "git clone --branch main --single-branch "
            "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /opt/Hunyuan3D-2"
        ),
        f"git -C {SOURCE_DIR} checkout {SOURCE_COMMIT}",
        f"python -m pip install -e {SOURCE_DIR} --no-deps",
    )
    .run_commands(
        (
            f"python -m pip install {SOURCE_DIR}/hy3dgen/texgen/custom_rasterizer "
            "--no-build-isolation"
        ),
    )
    .run_commands(
        (
            f"python -m pip install {SOURCE_DIR}/hy3dgen/texgen/differentiable_renderer "
            "--no-build-isolation"
        ),
        (
            "python -c \"import custom_rasterizer_kernel, mesh_processor, torch; "
            "assert torch.__version__.startswith('2.5.1')\""
        ),
    )
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface_hub[hf_xet]==0.33.4",
        "rembg==2.0.67",
        "onnxruntime==1.21.0",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "U2NET_HOME": REMBG_CACHE_DIR,
        }
    )
)


@app.function(
    image=download_image,
    volumes={CACHE_ROOT: model_cache},
    timeout=60 * 60 * 4,
)
def download_models() -> dict[str, str]:
    """Download only the official subfolders used by this deployment."""
    from huggingface_hub import snapshot_download
    from rembg import new_session

    shape_path = snapshot_download(
        SHAPE_REPOSITORY,
        allow_patterns=[
            f"{SHAPE_SUBFOLDER}/config.yaml",
            f"{SHAPE_SUBFOLDER}/model.fp16.safetensors",
        ],
    )
    paint_path = snapshot_download(
        PAINT_REPOSITORY,
        allow_patterns=[
            f"{DELIGHT_SUBFOLDER}/**",
            f"{PAINT_SUBFOLDER}/**",
        ],
    )
    new_session("u2net")
    model_cache.commit()
    print(f"[cached] {SHAPE_REPOSITORY}/{SHAPE_SUBFOLDER}")
    print(f"[cached] {PAINT_REPOSITORY}/{PAINT_SUBFOLDER}")
    print("[cached] rembg/u2net")
    return {"shape": shape_path, "paint": paint_path, "rembg": REMBG_CACHE_DIR}


@app.cls(
    image=gpu_image,
    gpu=GPU,
    cpu=8,
    memory=65_536,
    volumes={CACHE_ROOT: model_cache},
    timeout=60 * 60,
    scaledown_window=5 * 60,
    max_containers=1,
)
class Hunyuan3DMultiView:
    @modal.enter()
    def load_models(self) -> None:
        import os
        import sys

        os.chdir(SOURCE_DIR)
        if SOURCE_DIR not in sys.path:
            sys.path.insert(0, SOURCE_DIR)

        import torch
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.background_remover = BackgroundRemover()
        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            SHAPE_REPOSITORY,
            subfolder=SHAPE_SUBFOLDER,
            variant="fp16",
            use_safetensors=True,
            device="cuda",
            dtype=torch.float16,
        )
        self.paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            PAINT_REPOSITORY,
            subfolder=PAINT_SUBFOLDER,
        )

    @staticmethod
    def _integer(form: object, name: str, default: int) -> int:
        value = getattr(form, "get")(name, str(default))
        try:
            return int(str(value))
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc

    def _prepare_image(self, image):
        import numpy as np
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)
        has_alpha = image.mode == "RGBA" and np.asarray(image.getchannel("A")).min() < 250
        if not has_alpha:
            image = self.background_remover(image.convert("RGB"))
        return image.convert("RGBA")

    @modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=True)
    async def generate(self, request):
        import gc
        import io
        import time
        import uuid

        import torch
        from fastapi.responses import JSONResponse, Response
        from PIL import Image, UnidentifiedImageError

        started = time.perf_counter()
        try:
            form = await request.form()
            if form.get("front") is None:
                return JSONResponse(
                    {"detail": "A front view is required for Hunyuan3D-2mv."},
                    status_code=422,
                )

            steps = self._integer(form, "steps", 30)
            octree_resolution = self._integer(form, "octree_resolution", 380)
            seed = self._integer(form, "seed", 0)
            texture_mode = str(form.get("texture_mode", "textured"))

            if steps not in {20, 30, 50}:
                raise ValueError("steps must be 20, 30, or 50")
            if octree_resolution not in {256, 380}:
                raise ValueError("octree_resolution must be 256 or 380")
            if not 0 <= seed <= 2_147_483_647:
                raise ValueError("seed is outside the supported range")
            if texture_mode not in {"textured", "shape"}:
                raise ValueError("texture_mode must be textured or shape")

            images = {}
            total_bytes = 0
            for view in VIEW_ORDER:
                upload = form.get(view)
                if upload is None:
                    continue
                data = await upload.read()
                total_bytes += len(data)
                if not data or len(data) > 15 * 1024 * 1024:
                    raise ValueError("Each view must be non-empty and no larger than 15 MB")
                try:
                    image = Image.open(io.BytesIO(data))
                    image.load()
                except (UnidentifiedImageError, OSError) as exc:
                    raise ValueError(f"The {view} upload is not a readable image") from exc
                images[view] = self._prepare_image(image)

            if total_bytes > 60 * 1024 * 1024:
                raise ValueError("The complete view set must be no larger than 60 MB")

            with torch.inference_mode():
                mesh = self.shape_pipeline(
                    image=images,
                    num_inference_steps=steps,
                    octree_resolution=octree_resolution,
                    num_chunks=20_000,
                    generator=torch.manual_seed(seed),
                    output_type="trimesh",
                )[0]

                if texture_mode == "textured":
                    texture_images = [images[view] for view in VIEW_ORDER if view in images]
                    mesh = self.paint_pipeline(mesh, image=texture_images)

                payload = mesh.export(file_type="glb")
                if isinstance(payload, str):
                    payload = payload.encode()

            elapsed = time.perf_counter() - started
            filename = f"hunyuan3d-2mv-{uuid.uuid4().hex[:8]}.glb"
            return Response(
                content=payload,
                media_type="model/gltf-binary",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "X-Output-Filename": filename,
                    "X-Generation-Seconds": f"{elapsed:.2f}",
                    "X-Input-Views": str(len(images)),
                    "Cache-Control": "no-store",
                },
            )
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            return JSONResponse(
                {"detail": f"Hunyuan3D-2mv inference failed: {type(exc).__name__}: {exc}"},
                status_code=500,
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").uv_pip_install(
        "fastapi[standard]>=0.116,<1"
    )
)
@modal.fastapi_endpoint(docs=True, requires_proxy_auth=True)
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": "Hunyuan3D-2mv",
        "gpu": GPU,
        "views": list(VIEW_ORDER),
        "license": "Tencent Hunyuan Community License",
    }
