"""Run one low-cost, two-view GLB smoke test through the local studio."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = [
    ROOT / "vendor/ReconViaGen/assets/example_multi_image/robot_1.png",
    ROOT / "vendor/ReconViaGen/assets/example_multi_image/robot_2.png",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/generate")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reconviagen-smoke.glb")
    parser.add_argument("images", nargs="*", type=Path, default=DEFAULT_IMAGES)
    args = parser.parse_args()

    images = [path.resolve() for path in args.images]
    if not 1 <= len(images) <= 8:
        raise SystemExit("Provide between one and eight images.")
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing input files: {', '.join(missing)}")

    fields = {
        "strategy": "adaptive_guidance_weight",
        "pipeline_type": "512",
        "ss_source": "mesh",
        "seed": "0",
        "decimation_target": "100000",
        "texture_size": "1024",
    }

    with ExitStack() as stack:
        files = [
            (
                "images",
                (path.name, stack.enter_context(path.open("rb")), "image/png"),
            )
            for path in images
        ]
        response = httpx.post(
            args.url,
            data=fields,
            files=files,
            timeout=httpx.Timeout(3700, connect=60, write=180, pool=60),
        )

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise SystemExit(f"Generation failed ({response.status_code}): {detail}")
    if response.headers.get("content-type", "").split(";", 1)[0] != "model/gltf-binary":
        raise SystemExit("Endpoint returned an unexpected content type.")
    if not response.content.startswith(b"glTF"):
        raise SystemExit("Endpoint response is not a binary GLB file.")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    seconds = response.headers.get("x-generation-seconds", "unknown")
    print(f"Saved {output} ({len(response.content):,} bytes, generation={seconds}s)")


if __name__ == "__main__":
    main()
