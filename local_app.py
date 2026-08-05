"""Local ReconViaGen Studio.

The browser talks only to this local FastAPI process. This process validates the
uploads, adds Modal proxy-auth headers, and forwards the generation request to
the deployed GPU endpoint. The UI itself is never deployed to Modal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

load_dotenv(ROOT / ".env")

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_BYTES = 80 * 1024 * 1024
MAX_IMAGE_PIXELS = 64_000_000
ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_STRATEGIES = {
    "adaptive_guidance_weight",
    "weighted_average",
    "average_right",
    "sequential",
    "average",
    "fixed_guidance_rescale",
}
ALLOWED_PIPELINES = {"512", "1024", "1024_cascade", "1536_cascade"}
ALLOWED_SS_SOURCES = {"direct", "mesh", "mvtrellis2"}
ALLOWED_TEXTURE_SIZES = {1024, 2048, 4096}

app = FastAPI(
    title="ReconViaGen Studio",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _endpoint_config() -> tuple[str, dict[str, str]]:
    endpoint_url = os.getenv("MODAL_ENDPOINT_URL", "").strip()
    token_id = os.getenv("MODAL_PROXY_TOKEN_ID", "").strip()
    token_secret = os.getenv("MODAL_PROXY_TOKEN_SECRET", "").strip()

    if not endpoint_url:
        raise HTTPException(
            status_code=503,
            detail="Modal endpoint is not configured. Set MODAL_ENDPOINT_URL in .env.",
            headers={"X-Failure-Origin": "local-config"},
        )
    if not token_id or not token_secret:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modal proxy authentication is not configured. Set "
                "MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET in .env."
            ),
            headers={"X-Failure-Origin": "local-config"},
        )
    return endpoint_url, {"Modal-Key": token_id, "Modal-Secret": token_secret}


async def _read_and_validate(upload: UploadFile) -> tuple[str, bytes, str]:
    media_type = (upload.content_type or "").lower()
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"{upload.filename or 'Upload'} is not a PNG, JPEG, or WebP image.",
        )

    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=422, detail=f"{upload.filename or 'Upload'} is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{upload.filename or 'Upload'} is larger than 15 MB.",
        )

    try:
        with Image.open(__import__("io").BytesIO(data)) as image:
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{upload.filename or 'Upload'} is not a readable image.",
        ) from exc

    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"{upload.filename or 'Upload'} has too many pixels (maximum 64 MP).",
        )

    safe_name = Path(upload.filename or "view.png").name
    return safe_name, data, media_type


def _validate_options(
    strategy: str,
    pipeline_type: str,
    ss_source: str,
    seed: int,
    decimation_target: int,
    texture_size: int,
) -> None:
    if strategy not in ALLOWED_STRATEGIES:
        raise HTTPException(status_code=422, detail="Unknown multi-view fusion strategy.")
    if pipeline_type not in ALLOWED_PIPELINES:
        raise HTTPException(status_code=422, detail="Unknown output resolution.")
    if ss_source not in ALLOWED_SS_SOURCES:
        raise HTTPException(status_code=422, detail="Unknown sparse-structure source.")
    if not 0 <= seed <= 2_147_483_647:
        raise HTTPException(status_code=422, detail="Seed is outside the supported range.")
    if not 100_000 <= decimation_target <= 1_000_000:
        raise HTTPException(status_code=422, detail="Face target must be 100k–1M.")
    if texture_size not in ALLOWED_TEXTURE_SIZES:
        raise HTTPException(status_code=422, detail="Texture size must be 1024, 2048, or 4096.")


def _upstream_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        text = response.text.strip()
        return text[:800] or f"Modal returned HTTP {response.status_code}."
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body.get("message")
        if detail:
            return str(detail)[:800]
    return str(body)[:800]


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    endpoint = bool(os.getenv("MODAL_ENDPOINT_URL", "").strip())
    auth = bool(
        os.getenv("MODAL_PROXY_TOKEN_ID", "").strip()
        and os.getenv("MODAL_PROXY_TOKEN_SECRET", "").strip()
    )
    return {
        "ok": True,
        "endpoint_configured": endpoint,
        "auth_configured": auth,
        "max_images": MAX_IMAGES,
        "gpu": "H100",
        "model": "ReconViaGen v0.5",
        "capabilities": {
            "pipeline_types": ["512", "1024", "1024_cascade", "1536_cascade"],
            "strategies": [
                "adaptive_guidance_weight",
                "weighted_average",
                "average_right",
                "average",
                "sequential",
                "fixed_guidance_rescale",
            ],
            "structure_sources": ["mesh", "direct", "mvtrellis2"],
        },
    }


@app.post("/api/generate")
async def generate(
    images: Annotated[list[UploadFile], File(description="Ordered object views")],
    strategy: Annotated[str, Form()] = "adaptive_guidance_weight",
    pipeline_type: Annotated[str, Form()] = "1024_cascade",
    ss_source: Annotated[str, Form()] = "mesh",
    seed: Annotated[int, Form()] = 0,
    decimation_target: Annotated[int, Form()] = 500_000,
    texture_size: Annotated[int, Form()] = 2048,
) -> Response:
    if not 1 <= len(images) <= MAX_IMAGES:
        raise HTTPException(status_code=422, detail="Upload between 1 and 8 views.")
    _validate_options(strategy, pipeline_type, ss_source, seed, decimation_target, texture_size)

    validated = [await _read_and_validate(upload) for upload in images]
    if sum(len(item[1]) for item in validated) > MAX_TOTAL_BYTES:
        raise HTTPException(status_code=413, detail="The image set is larger than 80 MB.")

    endpoint_url, headers = _endpoint_config()
    files = [("images", (name, data, media_type)) for name, data, media_type in validated]
    fields = {
        "strategy": strategy,
        "pipeline_type": pipeline_type,
        "ss_source": ss_source,
        "seed": str(seed),
        "decimation_target": str(decimation_target),
        "texture_size": str(texture_size),
    }
    timeout_seconds = float(os.getenv("RECONVIAGEN_REQUEST_TIMEOUT_SECONDS", "3600"))
    timeout = httpx.Timeout(timeout_seconds, connect=60.0, write=180.0, pool=60.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            upstream = await client.post(endpoint_url, data=fields, files=files, headers=headers)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="The Modal generation timed out. Try the 512³ preset or fewer views.",
            headers={"X-Failure-Origin": "modal-transport"},
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach the Modal endpoint: {exc}",
            headers={"X-Failure-Origin": "modal-transport"},
        ) from exc

    if upstream.status_code != 200:
        raise HTTPException(
            status_code=upstream.status_code,
            detail=_upstream_error(upstream),
            headers={"X-Failure-Origin": "modal-upstream"},
        )

    filename = upstream.headers.get("X-Output-Filename", "reconviagen-model.glb")
    safe_filename = Path(filename).name
    return Response(
        content=upstream.content,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "X-Generation-Seconds": upstream.headers.get("X-Generation-Seconds", ""),
            "X-Input-Views": upstream.headers.get("X-Input-Views", str(len(validated))),
            "X-Pipeline-Type": upstream.headers.get("X-Pipeline-Type", pipeline_type),
            "X-Output-Resolution": upstream.headers.get("X-Output-Resolution", ""),
            "X-SS-Source": upstream.headers.get("X-SS-Source", ss_source),
            "X-Strategy": upstream.headers.get("X-Strategy", strategy),
            "X-Model": "ReconViaGen-v0.5",
            "Cache-Control": "no-store",
        },
    )
