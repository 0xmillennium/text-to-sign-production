"""Sprint 3 qualitative panel definition tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import text_to_sign_production.data.utils as utils_mod
from tests.support.builders.manifests import processed_record, write_jsonl_records
from tests.support.builders.samples import write_processed_sample_npz
from text_to_sign_production.modeling.data import read_processed_modeling_manifest
from text_to_sign_production.modeling.inference.qualitative import (
    QUALITATIVE_PANEL_SCHEMA_VERSION,
    QualitativeExportError,
    QualitativePanelDefinition,
    generate_validation_panel_definition,
    load_panel_definition,
    select_panel_records,
    write_panel_definition,
)

pytestmark = pytest.mark.unit


def test_qualitative_panel_module_imports_without_torch() -> None:
    script = """
import importlib.abc
import sys

for module_name in list(sys.modules):
    if module_name == "torch" or module_name.startswith("torch."):
        del sys.modules[module_name]

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return None

sys.meta_path.insert(0, BlockTorch())

from text_to_sign_production.modeling.inference.qualitative import QualitativePanelDefinition

panel = QualitativePanelDefinition(
    split="val",
    sample_ids=("sample-a",),
    selection_rule="import regression",
)
print(panel.to_dict()["split"])
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "val"


def _write_val_manifest(root: Path, sample_ids: list[str]) -> Path:
    records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        sample_path = f"data/processed/v1/samples/val/{sample_id}.npz"
        write_processed_sample_npz(root / sample_path)
        records.append(
            processed_record(
                sample_path,
                split="val",
                sample_id=sample_id,
                text=f"text for {sample_id}",
            )
        )

    manifest_path = root / "data/processed/v1/manifests/val.jsonl"
    write_jsonl_records(manifest_path, records)
    return manifest_path


def test_generated_panel_uses_sorted_validation_sample_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    manifest_path = _write_val_manifest(tmp_path, ["sample-c", "sample-a", "sample-b"])
    records = read_processed_modeling_manifest(manifest_path, split="val")

    panel = generate_validation_panel_definition(records, panel_size=2)

    assert panel.schema_version == QUALITATIVE_PANEL_SCHEMA_VERSION
    assert panel.split == "val"
    assert panel.sample_ids == ("sample-a", "sample-b")
    assert panel.selection_rule == "sorted-by-sample-id:first-2-of-3-validation-records"


def test_generated_panel_falls_back_to_available_validation_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    manifest_path = _write_val_manifest(tmp_path, ["sample-b", "sample-a"])
    records = read_processed_modeling_manifest(manifest_path, split="val")

    panel = generate_validation_panel_definition(records, panel_size=8)

    assert panel.sample_ids == ("sample-a", "sample-b")
    assert panel.selection_rule == "sorted-by-sample-id:first-2-of-2-validation-records"


def test_panel_definition_round_trips_as_json(tmp_path: Path) -> None:
    path = tmp_path / "panel.json"
    panel = QualitativePanelDefinition(
        split="val",
        sample_ids=("sample-a", "sample-b"),
        selection_rule="fixed-test-panel",
    )

    write_panel_definition(path, panel)
    loaded = load_panel_definition(path)

    assert loaded == panel


def test_load_panel_definition_rejects_non_validation_split(tmp_path: Path) -> None:
    path = tmp_path / "panel.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": QUALITATIVE_PANEL_SCHEMA_VERSION,
                "split": "train",
                "sample_ids": ["sample-a"],
                "selection_rule": "bad split",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(QualitativeExportError, match="split must be 'val'"):
        load_panel_definition(path)


def test_load_panel_definition_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    path = tmp_path / "panel.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": QUALITATIVE_PANEL_SCHEMA_VERSION,
                "split": "val",
                "sample_ids": ["sample-a", "sample-a"],
                "selection_rule": "duplicate",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(QualitativeExportError, match="duplicate sample_id"):
        load_panel_definition(path)


def test_load_panel_definition_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "panel.json"
    path.write_text("{broken\n", encoding="utf-8")

    with pytest.raises(QualitativeExportError, match="not valid JSON"):
        load_panel_definition(path)


def test_select_panel_records_rejects_missing_validation_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils_mod, "REPO_ROOT", tmp_path)
    manifest_path = _write_val_manifest(tmp_path, ["sample-a"])
    records = read_processed_modeling_manifest(manifest_path, split="val")
    panel = QualitativePanelDefinition(
        split="val",
        sample_ids=("missing",),
        selection_rule="explicit missing",
    )

    with pytest.raises(QualitativeExportError, match="not found"):
        select_panel_records(panel, records)
