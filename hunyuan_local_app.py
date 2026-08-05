"""Local-only Hunyuan3D-2mv studio and protected Modal proxy."""

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

# Reuse the existing protected Modal proxy token, while keeping the temporary
# Hunyuan endpoint separate from the ReconViaGen endpoint.
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.hunyuan", override=True)

VIEW_ORDER = ("front", "left", "back", "right")
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_BYTES = 60 * 1024 * 1024
MAX_IMAGE_PIXELS = 64_000_000
ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_STEPS = {20, 30, 50}
ALLOWED_RESOLUTIONS = {256, 380}
ALLOWED_TEXTURE_MODES = {"textured", "shape"}

app = FastAPI(
    title="Hunyuan3D-2mv Local Studio",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _endpoint_config() -> tuple[str, dict[str, str]]:
    endpoint_url = os.getenv("HUNYUAN_MODAL_ENDPOINT_URL", "").strip()
    token_id = os.getenv("MODAL_PROXY_TOKEN_ID", "").strip()
    token_secret = os.getenv("MODAL_PROXY_TOKEN_SECRET", "").strip()

    if not endpoint_url:
        raise HTTPException(
            status_code=503,
            detail="Hunyuan Modal endpoint is not configured in .env.hunyuan.",
        )
    if not token_id or not token_secret:
        raise HTTPException(
            status_code=503,
            detail="Modal proxy authentication is not configured in .env.",
        )
    return endpoint_url, {"Modal-Key": token_id, "Modal-Secret": token_secret}


async def _read_and_validate(view: str, upload: UploadFile) -> tuple[str, bytes, str]:
    media_type = (upload.content_type or "").lower()
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"The {view} view must be a PNG, JPEG, or WebP image.",
        )

    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=422, detail=f"The {view} view is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"The {view} view is larger than 15 MB.",
        )

    try:
        with Image.open(__import__("io").BytesIO(data)) as image:
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"The {view} upload is not a readable image.",
        ) from exc

    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"The {view} view exceeds the 64 MP limit.",
        )

    safe_name = Path(upload.filename or f"{view}.png").name
    return safe_name, data, media_type


def _upstream_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text.strip()[:800] or f"Modal returned HTTP {response.status_code}."
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body.get("message")
        if detail:
            return str(detail)[:800]
    return str(body)[:800]


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "hunyuan.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    endpoint = bool(os.getenv("HUNYUAN_MODAL_ENDPOINT_URL", "").strip())
    auth = bool(
        os.getenv("MODAL_PROXY_TOKEN_ID", "").strip()
        and os.getenv("MODAL_PROXY_TOKEN_SECRET", "").strip()
    )
    return {
        "ok": True,
        "endpoint_configured": endpoint,
        "auth_configured": auth,
        "views": list(VIEW_ORDER),
        "gpu": "H100",
        "model": "Hunyuan3D-2mv",
    }


@app.post("/api/generate")
async def generate(
    front: Annotated[UploadFile | None, File()] = None,
    left: Annotated[UploadFile | None, File()] = None,
    back: Annotated[UploadFile | None, File()] = None,
    right: Annotated[UploadFile | None, File()] = None,
    steps: Annotated[int, Form()] = 30,
    octree_resolution: Annotated[int, Form()] = 380,
    seed: Annotated[int, Form()] = 0,
    texture_mode: Annotated[str, Form()] = "textured",
) -> Response:
    if front is None:
        raise HTTPException(status_code=422, detail="Add a front view before generating.")
    if steps not in ALLOWED_STEPS:
        raise HTTPException(status_code=422, detail="Steps must be 20, 30, or 50.")
    if octree_resolution not in ALLOWED_RESOLUTIONS:
        raise HTTPException(status_code=422, detail="Detail must be 256 or 380.")
    if not 0 <= seed <= 2_147_483_647:
        raise HTTPException(status_code=422, detail="Seed is outside the supported range.")
    if texture_mode not in ALLOWED_TEXTURE_MODES:
        raise HTTPException(status_code=422, detail="Unknown texture mode.")

    uploads = {"front": front, "left": left, "back": back, "right": right}
    validated = {
        view: await _read_and_validate(view, upload)
        for view, upload in uploads.items()
        if upload is not None
    }
    if sum(len(item[1]) for item in validated.values()) > MAX_TOTAL_BYTES:
        raise HTTPException(status_code=413, detail="The complete view set exceeds 60 MB.")

    endpoint_url, headers = _endpoint_config()
    files = {
        view: (name, data, media_type)
        for view, (name, data, media_type) in validated.items()
    }
    fields = {
        "steps": str(steps),
        "octree_resolution": str(octree_resolution),
        "seed": str(seed),
        "texture_mode": texture_mode,
    }
    timeout_seconds = float(os.getenv("HUNYUAN_REQUEST_TIMEOUT_SECONDS", "3600"))
    timeout = httpx.Timeout(timeout_seconds, connect=60.0, write=180.0, pool=60.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            upstream = await client.post(endpoint_url, data=fields, files=files, headers=headers)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Hunyuan generation timed out. Try Balanced detail or Shape only.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach the protected Modal endpoint: {exc}",
        ) from exc

    if upstream.status_code != 200:
        raise HTTPException(status_code=upstream.status_code, detail=_upstream_error(upstream))

    filename = Path(
        upstream.headers.get("X-Output-Filename", "hunyuan3d-2mv-model.glb")
    ).name
    return Response(
        content=upstream.content,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Generation-Seconds": upstream.headers.get("X-Generation-Seconds", ""),
            "X-Input-Views": upstream.headers.get("X-Input-Views", str(len(validated))),
            "X-Model": "Hunyuan3D-2mv",
            "Cache-Control": "no-store",
        },
    )
