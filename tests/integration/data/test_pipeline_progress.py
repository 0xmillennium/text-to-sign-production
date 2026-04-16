"""Progress contract tests across data pipeline stages."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

import text_to_sign_production.data.filtering as filtering_mod
import text_to_sign_production.data.manifests as manifests_mod
import text_to_sign_production.data.normalize as normalize_mod
import text_to_sign_production.data.raw as raw_mod
from text_to_sign_production.data.constants import DEFAULT_FILTER_CONFIG_RELATIVE_PATH

pytestmark = pytest.mark.integration


def _record_progress_calls(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_iter_with_progress(
        iterable: Iterable[Any],
        total: int | None = None,
        desc: str = "",
        unit: str = "items",
    ) -> Iterator[Any]:
        calls.append({"total": total, "desc": desc, "unit": unit})
        yield from iterable

    monkeypatch.setattr(module, "iter_with_progress", fake_iter_with_progress)
    return calls


def test_long_running_pipeline_steps_use_shared_item_progress(
    tiny_dataset_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_progress_calls = _record_progress_calls(monkeypatch, raw_mod)
    raw_mod.build_raw_manifests(splits=("train",))
    assert raw_progress_calls == [{"total": 2, "desc": "Raw manifest train", "unit": "rows"}]

    normalize_progress_calls = _record_progress_calls(monkeypatch, normalize_mod)
    normalize_mod.normalize_all_splits(splits=("train",))
    assert normalize_progress_calls == [{"total": 2, "desc": "Normalize train", "unit": "samples"}]

    filtering_progress_calls = _record_progress_calls(monkeypatch, filtering_mod)
    filtering_mod.filter_all_splits(
        tiny_dataset_workspace / DEFAULT_FILTER_CONFIG_RELATIVE_PATH,
        splits=("train",),
    )
    assert filtering_progress_calls == [{"total": 2, "desc": "Filter train", "unit": "records"}]

    manifests_progress_calls = _record_progress_calls(monkeypatch, manifests_mod)
    assumption_report = json.loads(
        (tiny_dataset_workspace / "data/interim/reports/assumption-report.json").read_text(
            encoding="utf-8"
        )
    )
    filter_report = json.loads(
        (tiny_dataset_workspace / "data/interim/reports/filter-report.json").read_text(
            encoding="utf-8"
        )
    )
    manifests_mod.export_final_manifests(
        assumption_report=assumption_report,
        filter_report=filter_report,
        splits=("train",),
    )
    assert manifests_progress_calls == [
        {"total": 1, "desc": "Export processed manifest train", "unit": "records"}
    ]
