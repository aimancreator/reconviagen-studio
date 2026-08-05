#!/usr/bin/env python3
"""Generate one GLB from ordered character reference images.

This is the non-interactive counterpart to the local ReconViaGen web UI. It
calls the same FastAPI route in-process, so image validation, Modal proxy
authentication, generation settings, and error handling stay in one place.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import local_app  # noqa: E402

MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class GenerationRejected(RuntimeError):
    """The request was valid locally but Modal rejected it as a client error."""


class ModalServiceFailure(RuntimeError):
    """Modal or the GPU worker failed and the deployed App must be stopped."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a ReconViaGen GLB from 3–8 ordered character images."
    )
    parser.add_argument("images", nargs="+", type=Path, help="Ordered PNG, JPEG, or WebP views")
    parser.add_argument("--output", required=True, type=Path, help="Destination .glb path")
    parser.add_argument("--strategy", default="adaptive_guidance_weight")
    parser.add_argument("--pipeline-type", default="1024_cascade")
    parser.add_argument("--ss-source", default="mesh")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decimation-target", type=int, default=500_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    return parser.parse_args(argv)


def validate_image_paths(paths: list[Path]) -> list[Path]:
    if not 3 <= len(paths) <= local_app.MAX_IMAGES:
        raise ValueError(f"Expected 3–{local_app.MAX_IMAGES} images, received {len(paths)}.")

    validated: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path in seen:
            raise ValueError(f"Duplicate image path: {path}")
        if not path.is_file():
            raise ValueError(f"Image does not exist: {path}")
        if path.suffix.lower() not in MEDIA_TYPES:
            raise ValueError(f"Unsupported image type: {path.name}")
        seen.add(path)
        validated.append(path)
    return validated


def response_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text.strip()[:1000] or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("error") or payload)[:1000]
    return str(payload)[:1000]


def ensure_generation_success(response: httpx.Response) -> None:
    if response.status_code == 200:
        return

    failure_origin = response.headers.get("X-Failure-Origin", "")
    message = (
        f"ReconViaGen generation failed with HTTP {response.status_code}: "
        f"{response_error(response)}"
    )
    if failure_origin == "local-config":
        raise ValueError(message)
    if response.status_code >= 500:
        raise ModalServiceFailure(message)
    raise GenerationRejected(message)


async def generate(args: argparse.Namespace) -> dict[str, object]:
    images = validate_image_paths(args.images)
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".glb":
        raise ValueError("The reconstruction output must use the .glb extension.")

    files = [
        (
            "images",
            (path.name, path.read_bytes(), MEDIA_TYPES[path.suffix.lower()]),
        )
        for path in images
    ]
    fields = {
        "strategy": args.strategy,
        "pipeline_type": args.pipeline_type,
        "ss_source": args.ss_source,
        "seed": str(args.seed),
        "decimation_target": str(args.decimation_target),
        "texture_size": str(args.texture_size),
    }

    transport = httpx.ASGITransport(app=local_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://local") as client:
        response = await client.post("/api/generate", data=fields, files=files)

    ensure_generation_success(response)
    if not response.content:
        raise ModalServiceFailure("ReconViaGen returned an empty GLB.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    temporary_output.write_bytes(response.content)
    temporary_output.replace(output)

    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "input_views": len(images),
        "pipeline_type": response.headers.get("X-Pipeline-Type", args.pipeline_type),
        "strategy": response.headers.get("X-Strategy", args.strategy),
        "generation_seconds": response.headers.get("X-Generation-Seconds", ""),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        result = asyncio.run(generate(parse_args(argv)))
    except ModalServiceFailure as exc:
        print(f"MODAL_SERVICE_FAILURE: {exc}", file=sys.stderr)
        return 10
    except GenerationRejected as exc:
        print(f"GENERATION_REJECTED: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
