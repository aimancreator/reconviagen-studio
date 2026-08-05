from __future__ import annotations

import pytest

from modal_app import REMESH_BASE_RESOLUTION_ERROR, _to_glb_with_remesh_fallback


class StubPostprocess:
    def __init__(self, first_error: AssertionError) -> None:
        self.first_error = first_error
        self.calls: list[dict[str, object]] = []

    def to_glb(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise self.first_error
        return "glb-result"


def test_export_retries_without_remeshing_for_cumesh_resolution_assertion() -> None:
    postprocess = StubPostprocess(AssertionError(REMESH_BASE_RESOLUTION_ERROR))

    result = _to_glb_with_remesh_fallback(postprocess, grid_size=1024)

    assert result == "glb-result"
    assert postprocess.calls == [
        {
            "grid_size": 1024,
            "remesh": True,
            "remesh_band": 1,
            "remesh_project": 0,
        },
        {"grid_size": 1024, "remesh": False},
    ]


def test_export_does_not_mask_unrelated_assertions() -> None:
    postprocess = StubPostprocess(AssertionError("invalid texture attributes"))

    with pytest.raises(AssertionError, match="invalid texture attributes"):
        _to_glb_with_remesh_fallback(postprocess, grid_size=1024)

    assert len(postprocess.calls) == 1
