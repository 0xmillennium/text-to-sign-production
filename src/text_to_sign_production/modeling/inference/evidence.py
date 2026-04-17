"""Runtime-adjacent evidence bundle writing for Sprint 3 baseline exports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from text_to_sign_production.data.utils import write_json

from .paths import portable_optional_string_path, portable_path

BASELINE_EVIDENCE_SCHEMA_VERSION = "t2sp-baseline-evidence-v1"


class BaselineEvidenceError(ValueError):
    """Raised when a baseline evidence bundle cannot be written safely."""


def write_baseline_evidence_bundle(
    path: Path,
    *,
    config_path: Path,
    run_summary_path: Path,
    checkpoint_path: Path,
    checkpoint_payload: Mapping[str, Any],
    panel_definition_path: Path,
    panel_summary_path: Path,
    records_path: Path,
    sample_ids: Sequence[str],
    target_channels: Sequence[str],
) -> Path:
    """Write the minimal Phase 6 evidence surface for a qualitative baseline export."""

    run_summary = _read_run_summary(run_summary_path)
    payload = {
        "schema_version": BASELINE_EVIDENCE_SCHEMA_VERSION,
        "config_path": portable_path(config_path),
        "run_summary_path": portable_path(run_summary_path),
        "phase5_checkpoint_references": {
            "checkpoint_output_path": portable_optional_string_path(
                run_summary.get("checkpoint_output_path")
            ),
            "last_checkpoint_path": portable_optional_string_path(
                run_summary.get("last_checkpoint_path")
            ),
            "best_checkpoint_path": portable_optional_string_path(
                run_summary.get("best_checkpoint_path")
            ),
        },
        "export_checkpoint": {
            "path": portable_path(checkpoint_path),
            "role": checkpoint_payload["role"],
            "epoch": checkpoint_payload["epoch"],
            "backbone_name": checkpoint_payload["backbone_name"],
            "seed": checkpoint_payload["seed"],
            "metrics": {
                "train_loss": checkpoint_payload["train_loss"],
                "validation_loss": checkpoint_payload["validation_loss"],
                "validation_metric": checkpoint_payload["validation_metric"],
            },
        },
        "qualitative_panel": {
            "panel_definition_path": portable_path(panel_definition_path),
            "panel_summary_path": portable_path(panel_summary_path),
            "records_path": portable_path(records_path),
            "sample_count": len(sample_ids),
            "sample_ids": list(sample_ids),
            "target_channels": list(target_channels),
        },
        "artifact_role": "runtime-side baseline evidence, not a formal experiment record",
    }
    write_json(path, payload)
    return path


def _read_run_summary(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline run_summary.json does not exist: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineEvidenceError(f"Baseline run_summary.json is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise BaselineEvidenceError(f"Baseline run_summary.json root must be a mapping: {path}")
    return cast(Mapping[str, Any], loaded)
