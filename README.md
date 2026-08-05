# ReconViaGen Studio

A local multi-view web interface backed by a separately deployed Modal H100 API.
The browser UI is **not** hosted on Modal; only ReconViaGen v0.5 inference runs there.

## Where each part lives

- **Local workspace:** the Python web UI, its `.venv`, and a pinned copy of the
  ReconViaGen v0.5 source. The source copy is small compared with the checkpoints and
  lets the Modal build use an exact, auditable revision.
- **Modal Image:** the pinned ReconViaGen source and compiled CUDA dependencies are
  baked into the immutable H100 runtime during `modal deploy`. The GPU does not read
  source code from this computer at inference time.
- **Modal Volume:** the multi-gigabyte Hugging Face checkpoints live in the persistent
  `reconviagen-v05-model-cache` Volume. They are downloaded directly inside Modal and
  survive scale-to-zero and redeploys.
- **Local browser:** the generated GLB comes back through the protected endpoint for
  preview and download. It is not published as a Modal website.

Putting application source in a Volume would make releases mutable and harder to
reproduce. Modal Images are the appropriate home for code; Volumes are the appropriate
home for large, persistent model data.

## What the upstream model accepts

The official v0.5 branch accepts one or more object images. Inputs are converted to
RGBA automatically, and images without useful alpha are passed through BiRefNet
background removal. The upstream demo also accepts video, but it simply samples video
frames and sends those frames into the same image pipeline. This studio focuses on the
native multi-image path and allows 1–8 ordered PNG, JPEG, or WebP views.

For best results, use 2–8 photos of one isolated object with stable lighting, distance,
and scale. Cover front, three-quarter, side, and back angles. The default fusion method
is upstream's recommended `adaptive_guidance_weight`.

## Local setup

The project virtual environment is `.venv` and uses Python 3.11.

```bash
git submodule update --init --recursive
UV_CACHE_DIR=.uv-cache uv sync --dev
cp .env.example .env
```

Keep the populated `.env` file local. It contains the deployed endpoint's proxy
credentials and is intentionally excluded from Git. Modal CLI account credentials
created by `modal setup` live outside this repository and do not belong in `.env`.

## Modal setup and deployment

1. Accept the DINOv3 model terms on Hugging Face and create a read token from the same
   account:
   <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>
2. Authenticate the Modal CLI:

   ```bash
   UV_CACHE_DIR=.uv-cache uv run modal setup
   ```

3. Create the Hugging Face secret used by the model downloader and GPU container
   (skip this when `huggingface-secret` already exists):

   ```bash
   UV_CACHE_DIR=.uv-cache uv run modal secret create huggingface-secret HF_TOKEN=hf_your_token --force
   ```

4. Build and deploy only the inference API:

   ```bash
   UV_CACHE_DIR=.uv-cache uv run modal deploy modal_app.py
   ```

5. Create a proxy token and write the protected endpoint configuration to the ignored,
   mode-`0600` `.env` file. The helper never prints the token secret:

   ```bash
   UV_CACHE_DIR=.uv-cache uv run python scripts/configure_modal_proxy.py \
     https://your-workspace--reconviagen-v05-reconviagen-generate.modal.run
   ```

6. Pre-download the model weights into the persistent Modal Volume:

   ```bash
   UV_CACHE_DIR=.uv-cache uv run modal run modal_app.py::download_models
   ```

   If DINOv3 appears under `blocked`, the Hugging Face account behind
   `huggingface-secret` has not yet been granted access. The downloader commits every
   completed public repository, so rerunning it after approval only fetches what is
   missing.

## Run the local studio

```bash
UV_CACHE_DIR=.uv-cache uv run uvicorn local_app:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. Upload and order the views, choose settings, and click
**Generate 3D model**. The local Python server securely forwards the request to Modal,
then the browser previews and downloads the returned GLB.

## Cost-safe failure policy

The deployed GPU worker is configured with no Function retries, no warm or
buffer containers, a one-container maximum, and a two-second idle scale-down
window. The character folder runner additionally treats any Modal/GPU 5xx as a
hard stop: it stops the deployed `reconviagen-v05` App, captures the last 500 log
lines locally, cancels the queue, and never redeploys or resubmits
automatically.

Modal's platform-level crash recovery for deployed Function containers is
separate from the `retries=0` setting. Always inspect the saved failure log and
fix the cause before manually running `modal deploy` again.

## GPU choice

The endpoint requests one Modal `H100`. TRELLIS.2 requires at least 24 GB VRAM, is
verified on A100/H100, and its published 512³/1024³/1536³ timings are measured on H100.
ReconViaGen v0.5 adds its VGGT reconstruction stage, so H100 is the most faithful
single-GPU target while retaining upstream's low-VRAM CPU/GPU offload path.

## Sources and licenses

- ReconViaGen v0.5 source: <https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen/tree/v0.5>
- TRELLIS.2-4B model card: <https://huggingface.co/microsoft/TRELLIS.2-4B>
- Modal GPU documentation: <https://modal.com/docs/guide/gpu>
- Modal Web Function authentication: <https://modal.com/docs/guide/webhook-proxy-auth>

ReconViaGen and TRELLIS.2 are MIT licensed. DINOv3 uses the DINOv3 license and requires
accepting its Hugging Face access terms.
