from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from scripts.generate_character import (
    GenerationRejected,
    ModalServiceFailure,
    ensure_generation_success,
    validate_image_paths,
)


def test_validate_image_paths_requires_three_views(tmp_path: Path) -> None:
    first = tmp_path / "front.png"
    second = tmp_path / "back.png"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="Expected 3"):
        validate_image_paths([first, second])


def test_validate_image_paths_preserves_order(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("front.png", "left.jpg", "back.webp")]
    for path in paths:
        path.write_bytes(b"placeholder")

    assert validate_image_paths(paths) == [path.resolve() for path in paths]


def test_validate_image_paths_rejects_duplicate(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("front.png", "left.png", "back.png")]
    for path in paths:
        path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="Duplicate"):
        validate_image_paths([paths[0], paths[1], paths[1]])


def test_local_configuration_error_does_not_trigger_modal_fail_stop() -> None:
    response = httpx.Response(
        503,
        json={"detail": "Modal endpoint is not configured."},
        headers={"X-Failure-Origin": "local-config"},
    )

    with pytest.raises(ValueError, match="not configured"):
        ensure_generation_success(response)


def test_modal_server_error_triggers_fail_stop_exit_classification() -> None:
    response = httpx.Response(
        502,
        json={"detail": "Container crashed."},
        headers={"X-Failure-Origin": "modal-upstream"},
    )

    with pytest.raises(ModalServiceFailure, match="Container crashed"):
        ensure_generation_success(response)


def test_upstream_client_error_is_rejected_without_fail_stop() -> None:
    response = httpx.Response(
        422,
        json={"detail": "Bad generation option."},
        headers={"X-Failure-Origin": "modal-upstream"},
    )

    with pytest.raises(GenerationRejected, match="Bad generation option"):
        ensure_generation_success(response)
