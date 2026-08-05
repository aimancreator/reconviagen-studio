"""Modal GPU API for ReconViaGen v0.5.

Only this inference endpoint is deployed. The user-facing studio remains local
in ``local_app.py``.
"""

from __future__ import annotations

from pathlib import Path

import modal
from fastapi import Request

APP_NAME = "reconviagen-v05"
GPU = "H100"
NO_RETRIES = 0
IDLE_GPU_SECONDS = 2
UPSTREAM_LOCAL_DIR = Path(__file__).resolve().parent / "vendor" / "ReconViaGen"
UPSTREAM_REMOTE_DIR = "/opt/reconviagen"
HF_CACHE_DIR = "/cache/huggingface"
REMESH_BASE_RESOLUTION_ERROR = "Failed to find a base resolution that is a multiple of 2"


def _to_glb_with_remesh_fallback(postprocess: object, **kwargs: object) -> object:
    """Export with upstream remeshing, falling back only for CuMesh's grid assertion."""
    to_glb = getattr(postprocess, "to_glb")
    try:
        return to_glb(
            **kwargs,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
        )
    except AssertionError as exc:
        if REMESH_BASE_RESOLUTION_ERROR not in str(exc):
            raise
        print(
            "[export] CuMesh rejected narrow-band remeshing; "
            "retrying GLB export with standard mesh cleaning."
        )
        return to_glb(**kwargs, remesh=False)

if modal.is_local() and not UPSTREAM_LOCAL_DIR.exists():
    raise RuntimeError(
        "Official ReconViaGen v0.5 source is missing. Clone it into vendor/ReconViaGen."
    )

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("reconviagen-v05-model-cache", create_if_missing=True)

# ReconViaGen v0.5 pins the PyTorch 2.4/cu121 runtime and ships cp310 wheels.
# CuMesh requires a CUDA >=12.4 build toolkit; PyTorch supports this minor-version
# extension-build mismatch while retaining its official cu121 runtime packages.
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
            "SPCONV_ALGO": "native",
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "XFORMERS_DISABLED": "1",
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            "TORCH_HOME": f"{HF_CACHE_DIR}/torch",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .apt_install(
        "build-essential",
        "cmake",
        "ffmpeg",
        "git",
        "git-lfs",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libjpeg-dev",
        "libx11-6",
        "libxext6",
        "ninja-build",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja",
        (
            "python -m pip install torch==2.4.0 torchvision==0.19.0 "
            "--index-url https://download.pytorch.org/whl/cu121"
        ),
        (
            "python -m pip install pillow imageio imageio-ffmpeg tqdm easydict "
            "opencv-python-headless scipy rembg onnxruntime trimesh open3d xatlas "
            "pyvista pymeshfix igraph lpips kornia==0.8.2 zstandard rtree "
            "fast-simplification plyfile fastapi python-multipart hf-transfer "
            "huggingface_hub==1.7.2 transformers==5.3.0 timm==1.0.24"
        ),
        (
            "python -m pip install xformers==0.0.27.post2 "
            "--index-url https://download.pytorch.org/whl/cu121"
        ),
        (
            "python -m pip install "
            "https://github.com/Dao-AILab/flash-attention/releases/download/"
            "v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-"
            "cp310-cp310-linux_x86_64.whl"
        ),
        "python -m pip install spconv-cu120",
        (
            "python -m pip install kaolin -f "
            "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html"
        ),
        (
            "python -m pip install "
            "git+https://github.com/EasternJournalist/utils3d.git@"
            "9a4eb15e4021b67b12c460c7057d642626897ec8"
        ),
    )
    .add_local_dir(
        UPSTREAM_LOCAL_DIR,
        remote_path=UPSTREAM_REMOTE_DIR,
        copy=True,
        ignore=[".git", ".git/**", "**/__pycache__/**"],
    )
    .env({"CC": "gcc", "CXX": "g++"})
    .run_commands(
        ("python -m pip install /opt/reconviagen/extensions/nvdiffrast --no-build-isolation"),
    )
    .run_commands(
        (
            "python -m pip install "
            "git+https://github.com/JeffreyXiang/CuMesh.git@"
            "4843631790426a0c4f3ea1b36ec678783c63d84a "
            "--no-build-isolation"
        ),
    )
    .run_commands(
        # The conda-based upstream setup pairs torch 2.4 with Triton 3.2.
        # Install both explicitly so pip does not replace torch with a newer major.
        "python -m pip install triton==3.2.0 --no-deps",
        (
            "python -m pip install "
            "git+https://github.com/JeffreyXiang/FlexGEMM.git@"
            "401bb2d8f1c79dcbc7a696e87a48de00dacbf982 "
            "--no-build-isolation --no-deps"
        ),
        (
            "python -c \"import importlib.metadata as m, torch, triton; "
            "assert torch.__version__.startswith('2.4.0'); "
            "assert triton.__version__ == '3.2.0'; "
            "assert m.version('flex-gemm') == '1.0.0'\""
        ),
    )
    .run_commands(
        (
            "python -m pip install /opt/reconviagen/wheels/TRELLIS.2/o-voxel "
            "--no-build-isolation --no-deps"
        ),
    )
    .run_commands(
        # safetensors 0.8 uses a Torch storage API unavailable in torch 2.4.
        "python -m pip install safetensors==0.5.3 --no-deps",
    )
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub[hf_xet]>=1.7.2,<2")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            # The gated DINOv3 Xet CAS path can reject an otherwise valid
            # fine-grained token. The authenticated HTTP bridge is slower but
            # reliable, and this image runs only for one-time Volume preloads.
            "HF_HUB_DISABLE_XET": "1",
        }
    )
)

MODEL_REPOSITORIES = (
    "Stable-X/trellis-vggt-v0-2",
    "Stable-X/vggt-object-v0-1",
    "microsoft/TRELLIS.2-4B",
    "microsoft/TRELLIS-image-large",
    "ZhengPeng7/BiRefNet",
    # Keep the gated repository last: an Xet authorization failure can poison
    # later downloads in the same process with a "Previous task error".
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
)


@app.function(
    image=download_image,
    volumes={HF_CACHE_DIR: model_cache},
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    retries=NO_RETRIES,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
)
def download_models() -> dict[str, dict[str, str]]:
    """Populate the persistent Hugging Face cache before the first GPU request."""
    import os

    from huggingface_hub import snapshot_download

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    downloaded: dict[str, str] = {}
    blocked: dict[str, str] = {}
    for repository in MODEL_REPOSITORIES:
        try:
            options = {}
            if repository.startswith("facebook/dinov3-"):
                options = {"token": hf_token, "max_workers": 1}
            downloaded[repository] = snapshot_download(repository, **options)
            # Persist completed repositories even if a later gated model is denied.
            model_cache.commit()
            print(f"[cached] {repository}")
        except Exception as exc:
            blocked[repository] = f"{type(exc).__name__}: {exc}"
            reason = " ".join(str(exc).splitlines())[:600]
            print(f"[blocked] {repository}: {type(exc).__name__}: {reason}")
    return {"downloaded": downloaded, "blocked": blocked}


@app.function(
    image=download_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=5 * 60,
    retries=NO_RETRIES,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
)
def verify_huggingface_access() -> dict[str, object]:
    """Check the Modal secret and gated DINOv3 grant without exposing the token."""
    import json
    import os

    from huggingface_hub import HfApi

    api = HfApi()
    token_present = bool(
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    result: dict[str, object] = {"token_present": token_present}

    try:
        identity = api.whoami()
        result["huggingface_account"] = identity.get("name", "unknown")
    except Exception as exc:
        result["identity_error"] = f"{type(exc).__name__}: {exc}"

    try:
        info = api.model_info("facebook/dinov3-vitl16-pretrain-lvd1689m")
        result["dinov3_access"] = True
        result["dinov3_revision"] = info.sha
    except Exception as exc:
        result["dinov3_access"] = False
        result["dinov3_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, sort_keys=True))
    return result


@app.cls(
    image=gpu_image,
    gpu=GPU,
    cpu=8,
    memory=65_536,
    volumes={HF_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60,
    # Cost/failure policy: never keep a warm H100, never add a buffer
    # container, never retry ordinary Function errors, and release the GPU as
    # soon as Modal's minimum idle window allows.
    #
    # Modal itself reschedules a crashed container for deployed Functions even
    # when retries=0. The local character runner therefore treats any observed
    # 5xx as fatal, immediately runs `modal app stop reconviagen-v05`, saves the
    # last 500 log lines, and halts the entire queue without resubmitting.
    retries=NO_RETRIES,
    min_containers=0,
    buffer_containers=0,
    scaledown_window=IDLE_GPU_SECONDS,
    max_containers=1,
)
class ReconViaGen:
    @modal.enter()
    def load_model(self) -> None:
        import gc
        import os
        import sys

        os.chdir(UPSTREAM_REMOTE_DIR)
        trellis2_root = f"{UPSTREAM_REMOTE_DIR}/wheels/TRELLIS.2"
        if UPSTREAM_REMOTE_DIR not in sys.path:
            sys.path.insert(0, UPSTREAM_REMOTE_DIR)
        if trellis2_root not in sys.path:
            sys.path.insert(0, trellis2_root)

        import torch
        from trellis.pipelines import TrellisVGGTTo3DPipeline
        from trellis.pipelines.trellis_hybrid_pipeline import TrellisHybridPipeline
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        vggt = TrellisVGGTTo3DPipeline.from_pretrained("Stable-X/trellis-vggt-v0-2")
        vggt.cuda()
        vggt.VGGT_model.cuda()
        vggt.birefnet_model.cuda()
        vggt.models.pop("slat_decoder_gs", None)

        # The upstream v0.5 demo enables staged CPU/GPU offload even on large GPUs.
        vggt.VGGT_model.cpu()
        for model in vggt.models.values():
            model.cpu()

        trellis2 = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        trellis2.cuda()
        trellis2.low_vram = True

        self.pipeline = TrellisHybridPipeline(vggt, trellis2, low_vram=True)
        gc.collect()
        torch.cuda.empty_cache()
        # Persist Torch Hub's DINOv2 code and weights downloaded during the
        # first cold start so later containers reuse them from the Volume.
        model_cache.commit()

    @staticmethod
    def _integer(form: object, name: str, default: int) -> int:
        value = getattr(form, "get")(name, str(default))
        try:
            return int(str(value))
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc

    @modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=True)
    async def generate(self, request: Request):
        import gc
        import io
        import time
        import uuid

        import torch
        from fastapi.responses import JSONResponse, Response
        from PIL import Image, ImageOps, UnidentifiedImageError
        import o_voxel

        started = time.perf_counter()
        try:
            form = await request.form()
            uploads = form.getlist("images")
            if not 1 <= len(uploads) <= 8:
                return JSONResponse({"detail": "Upload between 1 and 8 views."}, status_code=422)

            strategy = str(form.get("strategy", "adaptive_guidance_weight"))
            pipeline_type = str(form.get("pipeline_type", "1024_cascade"))
            ss_source = str(form.get("ss_source", "mesh"))
            seed = self._integer(form, "seed", 0)
            decimation_target = self._integer(form, "decimation_target", 500_000)
            texture_size = self._integer(form, "texture_size", 2048)

            strategies = {
                "average_right",
                "weighted_average",
                "sequential",
                "average",
                "adaptive_guidance_weight",
                "fixed_guidance_rescale",
            }
            if strategy not in strategies:
                raise ValueError("Unknown multi-view fusion strategy")
            if pipeline_type not in {"512", "1024", "1024_cascade", "1536_cascade"}:
                raise ValueError("Unknown pipeline_type")
            if ss_source not in {"direct", "mesh", "mvtrellis2"}:
                raise ValueError("Unknown ss_source")
            if not 0 <= seed <= 2_147_483_647:
                raise ValueError("seed is outside the supported range")
            if not 100_000 <= decimation_target <= 1_000_000:
                raise ValueError("decimation_target must be 100000–1000000")
            if texture_size not in {1024, 2048, 4096}:
                raise ValueError("texture_size must be 1024, 2048, or 4096")

            images = []
            for upload in uploads:
                data = await upload.read()
                if not data or len(data) > 15 * 1024 * 1024:
                    raise ValueError("Each image must be non-empty and no larger than 15 MB")
                try:
                    image = Image.open(io.BytesIO(data))
                    image.load()
                except (UnidentifiedImageError, OSError) as exc:
                    raise ValueError("One of the uploads is not a readable image") from exc
                image = ImageOps.exif_transpose(image)
                if image.mode != "RGBA":
                    image = image.convert("RGBA")
                images.append(image)

            ss_params = {
                "steps": 12,
                "cfg_strength": 7.5,
                "cfg_interval": [0.6, 1.0],
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            }
            slat_params = {
                "steps": 12,
                "cfg_strength": 7.5,
                "cfg_interval": [0.6, 1.0],
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            }
            shape_params = {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            }
            texture_params = {
                "steps": 12,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            }

            run_kwargs = {
                "seed": seed,
                "ss_sampler_params": ss_params,
                "slat_sampler_params": slat_params,
                "shape_slat_sampler_params": shape_params,
                "tex_slat_sampler_params": texture_params,
                "pipeline_type": pipeline_type,
                "preprocess_image": True,
                # The returned latent state contains the exact decoded resolution.
                # Mesh.voxel_shape is an internal sparse-tensor shape and must not
                # be used as CuMesh's remeshing resolution.
                "return_latent": True,
                "ss_source": ss_source,
            }
            with torch.inference_mode():
                if len(images) == 1:
                    meshes, latent_state = self.pipeline.run(images, **run_kwargs)
                else:
                    meshes, latent_state = self.pipeline.run_multi_image(
                        images, strategy=strategy, **run_kwargs
                    )
                output_resolution = int(latent_state[2])
                del latent_state
                torch.cuda.empty_cache()
                mesh = meshes[0]
                mesh.simplify(16_777_216)
                glb = _to_glb_with_remesh_fallback(
                    o_voxel.postprocess,
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=self.pipeline.pbr_attr_layout,
                    grid_size=output_resolution,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=decimation_target,
                    texture_size=texture_size,
                    use_tqdm=True,
                )
                output = io.BytesIO()
                glb.export(output, file_type="glb", extension_webp=True)
                payload = output.getvalue()

            elapsed = time.perf_counter() - started
            filename = f"reconviagen-{uuid.uuid4().hex[:8]}.glb"
            return Response(
                content=payload,
                media_type="model/gltf-binary",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "X-Output-Filename": filename,
                    "X-Generation-Seconds": f"{elapsed:.2f}",
                    "X-Input-Views": str(len(images)),
                    "X-Pipeline-Type": pipeline_type,
                    "X-Output-Resolution": str(output_resolution),
                    "X-SS-Source": ss_source,
                    "X-Strategy": strategy,
                    "Cache-Control": "no-store",
                },
            )
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        except Exception as exc:
            # Modal logs retain the full traceback; callers get a concise message.
            import traceback

            traceback.print_exc()
            return JSONResponse(
                {"detail": f"ReconViaGen inference failed: {type(exc).__name__}: {exc}"},
                status_code=500,
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").uv_pip_install(
        "fastapi[standard]>=0.116,<1"
    ),
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
)
@modal.fastapi_endpoint(docs=True, requires_proxy_auth=True)
def health() -> dict[str, str]:
    return {"status": "ok", "model": "ReconViaGen-v0.5", "gpu": GPU}
