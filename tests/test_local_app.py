from __future__ import annotations

import io

import httpx
import pytest
from PIL import Image

import local_app


def image_bytes(format_name: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (48, 48), "#c8ff3d").save(output, format=format_name)
    return output.getvalue()


@pytest.fixture
def transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=local_app.app)


@pytest.mark.asyncio
async def test_index_and_health(transport: httpx.ASGITransport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        index = await client.get("/")
        health = await client.get("/api/health")

    assert index.status_code == 200
    assert "Turn every angle" in index.text
    assert health.status_code == 200
    assert health.json()["max_images"] == 8
    assert health.json()["model"] == "ReconViaGen v0.5"
    assert health.json()["capabilities"]["pipeline_types"] == [
        "512",
        "1024",
        "1024_cascade",
        "1536_cascade",
    ]
    assert 'data-value="1024"' in index.text
    assert 'value="fixed_guidance_rescale"' in index.text
    assert "directResolution" not in index.text


@pytest.mark.asyncio
async def test_generate_rejects_missing_images(transport: httpx.ASGITransport) -> None:
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/generate", data={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_generate_requires_endpoint_config(
    transport: httpx.ASGITransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MODAL_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            files=[("images", ("front.png", image_bytes(), "image/png"))],
        )

    assert response.status_code == 503
    assert "MODAL_ENDPOINT_URL" in response.json()["detail"]
    assert response.headers["x-failure-origin"] == "local-config"


@pytest.mark.asyncio
async def test_generate_rejects_non_image(
    transport: httpx.ASGITransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODAL_ENDPOINT_URL", "https://example.test/generate")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            files=[("images", ("notes.txt", b"not an image", "text/plain"))],
        )

    assert response.status_code == 422
    assert "PNG, JPEG, or WebP" in response.json()["detail"]


@pytest.mark.asyncio
async def test_generate_forwards_ordered_multiple_images(
    transport: httpx.ASGITransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODAL_ENDPOINT_URL", "https://example.test/generate")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    captured: dict[str, object] = {}

    class StubClient:
        def __init__(self, **_: object) -> None:
            pass

        async def __aenter__(self) -> "StubClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> httpx.Response:
            captured["url"] = url
            captured.update(kwargs)
            return httpx.Response(
                200,
                content=b"glTF-test",
                headers={
                    "X-Output-Filename": "ordered.glb",
                    "X-Generation-Seconds": "1.25",
                },
                request=httpx.Request("POST", url),
            )

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(local_app.httpx, "AsyncClient", StubClient)

    async with real_async_client(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/generate",
            files=[
                ("images", ("front.png", image_bytes(), "image/png")),
                ("images", ("back.png", image_bytes(), "image/png")),
            ],
            data={"strategy": "adaptive_guidance_weight", "pipeline_type": "512"},
        )

    assert response.status_code == 200
    assert response.content == b"glTF-test"
    assert response.headers["content-disposition"] == 'attachment; filename="ordered.glb"'
    assert response.headers["x-input-views"] == "2"
    assert response.headers["x-pipeline-type"] == "512"
    assert response.headers["x-ss-source"] == "mesh"
    assert response.headers["x-strategy"] == "adaptive_guidance_weight"
    assert captured["url"] == "https://example.test/generate"
    assert [item[1][0] for item in captured["files"]] == ["front.png", "back.png"]  # type: ignore[index]
    assert captured["headers"] == {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"}
