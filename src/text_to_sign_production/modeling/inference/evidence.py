"""Runtime-adjacent evidence bundle writing for M0 baseline exports."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .paths import portable_optional_string_path, portable_path

BASELINE_EVIDENCE_SCHEMA_VERSION = "t2sp-baseline-evidence-v1"


class BaselineEvidenceError(ValueError):
    """Raised when a baseline evidence bundle cannot be written safely."""


def write_baseline_evidence_bundle(
    path: Path,
    *,
    config_path: Path,
    training_summary_path: Path,
    run_summary_path: Path,
    checkpoint_path: Path,
    checkpoint_payload: Mapping[str, Any],
    panel_definition_path: Path,
    panel_summary_path: Path,
    records_path: Path,
    sample_ids: Sequence[str],
    target_channels: Sequence[str],
    path_formatter: Callable[[Path], str],
) -> Path:
    """Write the minimal Phase 6 evidence surface for a qualitative baseline export."""

    training_summary = _read_training_summary(training_summary_path)
    payload = {
        "schema_version": BASELINE_EVIDENCE_SCHEMA_VERSION,
        "run_mode": training_summary.get("run_mode"),
        "run_mode_statement": training_summary.get("run_mode_statement"),
        "config_path": portable_path(config_path, path_formatter=path_formatter),
        "training_summary_path": portable_path(
            training_summary_path,
            path_formatter=path_formatter,
        ),
        "run_summary_path": portable_path(run_summary_path, path_formatter=path_formatter),
        "phase5_checkpoint_references": {
            "checkpoint_output_path": portable_optional_string_path(
                training_summary.get("checkpoint_output_path"),
                path_formatter=path_formatter,
            ),
            "last_checkpoint_path": portable_optional_string_path(
                training_summary.get("last_checkpoint_path"),
                path_formatter=path_formatter,
            ),
            "best_checkpoint_path": portable_optional_string_path(
                training_summary.get("best_checkpoint_path"),
                path_formatter=path_formatter,
            ),
        },
        "export_checkpoint": {
            "path": portable_path(checkpoint_path, path_formatter=path_formatter),
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
            "panel_definition_path": portable_path(
                panel_definition_path,
                path_formatter=path_formatter,
            ),
            "panel_summary_path": portable_path(
                panel_summary_path,
                path_formatter=path_formatter,
            ),
            "records_path": portable_path(records_path, path_formatter=path_formatter),
            "sample_count": len(sample_ids),
            "sample_ids": list(sample_ids),
            "target_channels": list(target_channels),
        },
        "artifact_role": "runtime-side baseline evidence, not a formal experiment record",
    }
    _write_json_existing_parent(path, payload)
    return path


def _read_training_summary(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline training summary does not exist: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineEvidenceError(f"Baseline training summary is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise BaselineEvidenceError(f"Baseline training summary root must be a mapping: {path}")
    return cast(Mapping[str, Any], loaded)


def _write_json_existing_parent(path: Path, payload: object) -> None:
    if not path.parent.is_dir():
        raise BaselineEvidenceError(f"Evidence parent directory does not exist: {path.parent}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
