"""Qualitative validation-panel export for M0 baseline runs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from text_to_sign_production.modeling.data import (
    M0_TARGET_CHANNEL_SHAPES,
    M0_TARGET_CHANNELS,
    ProcessedModelingManifestRecord,
    ProcessedPoseItem,
    collate_processed_pose_samples,
    load_processed_pose_sample,
    read_processed_modeling_manifest,
)
from text_to_sign_production.modeling.training.config import load_baseline_training_config

from .evidence import write_baseline_evidence_bundle
from .paths import portable_path
from .schemas import (
    build_prediction_sample_payload,
    prediction_sample_policy_metadata,
)

if TYPE_CHECKING:
    from text_to_sign_production.modeling.models import BaselinePoseOutput

QUALITATIVE_PANEL_SCHEMA_VERSION = "t2sp-qualitative-panel-v1"
QUALITATIVE_EXPORT_SCHEMA_VERSION = "t2sp-qualitative-export-v1"
DEFAULT_QUALITATIVE_PANEL_SIZE = 8
QUALITATIVE_PANEL_SPLIT = "val"
PANEL_DEFINITION_FILENAME = "panel_definition.json"
QUALITATIVE_RECORDS_FILENAME = "records.jsonl"
PANEL_SUMMARY_FILENAME = "panel_summary.json"
EVIDENCE_BUNDLE_FILENAME = "baseline_evidence_bundle.json"
REFERENCE_ARTIFACTS_DIRNAME = "references"
PREDICTION_ARTIFACTS_DIRNAME = "predictions"


class QualitativeExportError(ValueError):
    """Raised when qualitative panel export inputs or artifacts are invalid."""


@dataclass(frozen=True, slots=True)
class QualitativePanelDefinition:
    """Fixed validation-panel definition for M0 qualitative export."""

    split: str
    sample_ids: tuple[str, ...]
    selection_rule: str
    schema_version: str = QUALITATIVE_PANEL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""

        return {
            "schema_version": self.schema_version,
            "split": self.split,
            "sample_ids": list(self.sample_ids),
            "selection_rule": self.selection_rule,
        }


@dataclass(frozen=True, slots=True)
class QualitativeExportResult:
    """Paths produced by a qualitative panel export run."""

    output_dir: Path
    panel_definition_path: Path
    records_path: Path
    panel_summary_path: Path
    evidence_bundle_path: Path
    checkpoint_path: Path
    sample_count: int
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _QualitativeArtifactPaths:
    output_dir: Path
    panel_definition_path: Path
    records_path: Path
    panel_summary_path: Path
    evidence_bundle_path: Path


def generate_validation_panel_definition(
    records: Sequence[ProcessedModelingManifestRecord],
    *,
    panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE,
) -> QualitativePanelDefinition:
    """Generate the fixed deterministic validation panel from manifest records."""

    if panel_size <= 0:
        raise QualitativeExportError("panel_size must be positive.")

    validation_records = sorted(
        _validation_records(records),
        key=lambda record: record.sample_id,
    )
    if not validation_records:
        raise QualitativeExportError(
            "Cannot build a qualitative panel from zero validation records."
        )

    selected_records = validation_records[: min(panel_size, len(validation_records))]
    return QualitativePanelDefinition(
        split=QUALITATIVE_PANEL_SPLIT,
        sample_ids=tuple(record.sample_id for record in selected_records),
        selection_rule=(
            "sorted-by-sample-id:first-"
            f"{len(selected_records)}-of-{len(validation_records)}-validation-records"
        ),
    )


def load_panel_definition(path: Path | str) -> QualitativePanelDefinition:
    """Load and validate a qualitative panel definition JSON file."""

    resolved_path = _resolve_path(path)
    try:
        loaded = json.loads(resolved_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QualitativeExportError(f"Could not read panel definition: {resolved_path}") from exc
    except json.JSONDecodeError as exc:
        raise QualitativeExportError(
            f"Panel definition is not valid JSON: {resolved_path}"
        ) from exc

    if not isinstance(loaded, dict):
        raise QualitativeExportError("Panel definition root must be a mapping.")
    return _panel_definition_from_mapping(cast(Mapping[str, object], loaded))


def write_panel_definition(path: Path, panel: QualitativePanelDefinition) -> Path:
    """Write a normalized qualitative panel definition JSON artifact."""

    _write_json_existing_parent(path, panel.to_dict())
    return path


def select_panel_records(
    panel: QualitativePanelDefinition,
    records: Sequence[ProcessedModelingManifestRecord],
) -> list[ProcessedModelingManifestRecord]:
    """Select panel records from validation manifest records, preserving panel order."""

    _validate_panel_split(panel)
    by_sample_id: dict[str, ProcessedModelingManifestRecord] = {}
    for record in records:
        if record.split != QUALITATIVE_PANEL_SPLIT:
            raise QualitativeExportError(
                f"Qualitative panel records must come from split 'val'; "
                f"record {record.sample_id!r} has split {record.split!r}."
            )
        by_sample_id[record.sample_id] = record

    selected: list[ProcessedModelingManifestRecord] = []
    for sample_id in panel.sample_ids:
        selected_record = by_sample_id.get(sample_id)
        if selected_record is None:
            raise QualitativeExportError(
                f"Panel sample_id {sample_id!r} was not found in the validation manifest."
            )
        selected.append(selected_record)
    return selected


def export_qualitative_panel(
    config_path: Path | str,
    *,
    output_dir: Path | str,
    training_summary_path: Path | str,
    run_summary_path: Path | str,
    checkpoint_path: Path | str | None = None,
    checkpoint_output_dir: Path | str | None = None,
    panel_definition_path: Path | str | None = None,
    panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE,
    repo_root: Path | str | None = None,
    path_formatter: Callable[[Path], str],
    progress_reporter: Any | None = None,
) -> QualitativeExportResult:
    """Export the fixed qualitative validation panel for an M0 baseline checkpoint."""

    load_baseline_predictor, predict_baseline_batch = _load_prediction_helpers()
    config = load_baseline_training_config(
        config_path,
        checkpoint_output_dir=checkpoint_output_dir,
        repo_root=repo_root,
    )
    if config.data.val_split != QUALITATIVE_PANEL_SPLIT:
        raise QualitativeExportError(
            "Qualitative export requires data.val_split to be 'val'; "
            f"got {config.data.val_split!r}."
        )

    resolved_training_summary_path = _resolve_path(training_summary_path, repo_root=repo_root)
    resolved_run_summary_path = _resolve_path(run_summary_path, repo_root=repo_root)
    if not resolved_training_summary_path.is_file():
        raise FileNotFoundError(
            f"Baseline training summary does not exist: {resolved_training_summary_path}"
        )

    validation_records = read_processed_modeling_manifest(
        config.data.val_manifest,
        split=QUALITATIVE_PANEL_SPLIT,
    )
    panel = _resolve_export_panel_definition(
        panel_definition_path=panel_definition_path,
        validation_records=validation_records,
        panel_size=panel_size,
        repo_root=repo_root,
    )
    selected_records = select_panel_records(panel, validation_records)
    predictor = load_baseline_predictor(
        config,
        checkpoint_path=checkpoint_path,
        repo_root=repo_root,
    )

    artifact_paths = _qualitative_artifact_paths(output_dir)
    _require_qualitative_output_dirs(artifact_paths.output_dir)
    write_panel_definition(artifact_paths.panel_definition_path, panel)
    artifact_records = _export_qualitative_sample_artifacts(
        selected_records=selected_records,
        output_dir=artifact_paths.output_dir,
        predictor=predictor,
        predict_baseline_batch=predict_baseline_batch,
        path_formatter=path_formatter,
        progress_reporter=progress_reporter,
    )
    _write_qualitative_export_manifests(
        artifact_paths=artifact_paths,
        panel=panel,
        config_path=config.source_path,
        training_summary_path=resolved_training_summary_path,
        run_summary_path=resolved_run_summary_path,
        predictor=predictor,
        artifact_records=artifact_records,
        path_formatter=path_formatter,
    )

    return QualitativeExportResult(
        output_dir=artifact_paths.output_dir,
        panel_definition_path=artifact_paths.panel_definition_path,
        records_path=artifact_paths.records_path,
        panel_summary_path=artifact_paths.panel_summary_path,
        evidence_bundle_path=artifact_paths.evidence_bundle_path,
        checkpoint_path=predictor.checkpoint_path,
        sample_count=len(panel.sample_ids),
        sample_ids=panel.sample_ids,
    )


def _qualitative_artifact_paths(output_dir: Path | str) -> _QualitativeArtifactPaths:
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    return _QualitativeArtifactPaths(
        output_dir=resolved_output_dir,
        panel_definition_path=resolved_output_dir / PANEL_DEFINITION_FILENAME,
        records_path=resolved_output_dir / QUALITATIVE_RECORDS_FILENAME,
        panel_summary_path=resolved_output_dir / PANEL_SUMMARY_FILENAME,
        evidence_bundle_path=resolved_output_dir / EVIDENCE_BUNDLE_FILENAME,
    )


def _resolve_export_panel_definition(
    *,
    panel_definition_path: Path | str | None,
    validation_records: Sequence[ProcessedModelingManifestRecord],
    panel_size: int,
    repo_root: Path | str | None,
) -> QualitativePanelDefinition:
    if panel_definition_path is not None:
        return _load_panel_definition(
            panel_definition_path,
            repo_root=repo_root,
        )
    return generate_validation_panel_definition(validation_records, panel_size=panel_size)


def _export_qualitative_sample_artifacts(
    *,
    selected_records: Sequence[ProcessedModelingManifestRecord],
    output_dir: Path,
    predictor: Any,
    predict_baseline_batch: Any,
    path_formatter: Callable[[Path], str],
    progress_reporter: Any | None = None,
) -> list[dict[str, object]]:
    from text_to_sign_production.core.progress import (
        ItemProgress,
        NoOpProgressReporter,
    )

    progress = ItemProgress(
        label="qualitative panel",
        total=len(selected_records),
        unit="sample",
        reporter=progress_reporter if progress_reporter is not None else NoOpProgressReporter(),
        interval=1,
    )
    artifact_records: list[dict[str, object]] = []
    for index, record in enumerate(selected_records):
        pose_sample = load_processed_pose_sample(record)
        item = ProcessedPoseItem.from_manifest_and_sample(record, pose_sample)
        batch = collate_processed_pose_samples([item])
        prediction = predict_baseline_batch(
            predictor.model,
            batch,
            device=predictor.device,
        )
        artifact_records.append(
            write_qualitative_sample_artifacts(
                output_dir,
                index=index,
                item=item,
                prediction=prediction,
                path_formatter=path_formatter,
            )
        )
        progress.advance(sample_id=record.sample_id)
    progress.finish()
    return artifact_records


def _write_qualitative_export_manifests(
    *,
    artifact_paths: _QualitativeArtifactPaths,
    panel: QualitativePanelDefinition,
    config_path: Path,
    training_summary_path: Path,
    run_summary_path: Path,
    predictor: Any,
    artifact_records: list[dict[str, object]],
    path_formatter: Callable[[Path], str],
) -> None:
    _write_jsonl_existing_parent(artifact_paths.records_path, artifact_records)
    panel_summary = _panel_summary_payload(
        panel=panel,
        records_path=artifact_paths.records_path,
        panel_definition_path=artifact_paths.panel_definition_path,
        checkpoint_path=predictor.checkpoint_path,
        artifact_records=artifact_records,
        path_formatter=path_formatter,
    )
    _write_json_existing_parent(artifact_paths.panel_summary_path, panel_summary)

    write_baseline_evidence_bundle(
        artifact_paths.evidence_bundle_path,
        config_path=config_path,
        training_summary_path=training_summary_path,
        run_summary_path=run_summary_path,
        checkpoint_path=predictor.checkpoint_path,
        checkpoint_payload=predictor.checkpoint_payload,
        panel_definition_path=artifact_paths.panel_definition_path,
        panel_summary_path=artifact_paths.panel_summary_path,
        records_path=artifact_paths.records_path,
        sample_ids=panel.sample_ids,
        target_channels=M0_TARGET_CHANNELS,
        path_formatter=path_formatter,
    )


def write_qualitative_sample_artifacts(
    output_dir: Path,
    *,
    index: int,
    item: ProcessedPoseItem,
    prediction: BaselinePoseOutput,
    path_formatter: Callable[[Path], str],
) -> dict[str, object]:
    """Write one reference/prediction pair and return its metadata record."""

    if index < 0:
        raise QualitativeExportError("artifact index must not be negative.")

    filename = f"{index:04d}__{_sanitize_sample_id(item.sample_id)}.npz"
    reference_path = output_dir / REFERENCE_ARTIFACTS_DIRNAME / filename
    prediction_path = output_dir / PREDICTION_ARTIFACTS_DIRNAME / filename
    _require_existing_dir(reference_path.parent, label="Qualitative reference artifact directory")
    _require_existing_dir(
        prediction_path.parent,
        label="Qualitative prediction artifact directory",
    )

    np.savez_compressed(
        reference_path,
        processed_schema_version=np.asarray(item.processed_schema_version),
        body=item.body.astype(np.float32, copy=False),
        body_confidence=item.body_confidence.astype(np.float32, copy=False),
        left_hand=item.left_hand.astype(np.float32, copy=False),
        left_hand_confidence=item.left_hand_confidence.astype(np.float32, copy=False),
        right_hand=item.right_hand.astype(np.float32, copy=False),
        right_hand_confidence=item.right_hand_confidence.astype(np.float32, copy=False),
        face=item.face.astype(np.float32, copy=False),
        face_confidence=item.face_confidence.astype(np.float32, copy=False),
        frame_valid_mask=item.frame_valid_mask.astype(np.bool_, copy=False),
        people_per_frame=item.people_per_frame,
        selected_person_index=np.asarray(item.selected_person_index, dtype=np.int16),
    )
    prediction_arrays = {
        channel: _prediction_channel_array(
            prediction,
            channel=channel,
            expected_length=item.length,
        )
        for channel in M0_TARGET_CHANNELS
    }
    prediction_payload = build_prediction_sample_payload(
        prediction_arrays,
        frame_valid_mask=item.frame_valid_mask,
        selected_person_index=item.selected_person_index,
        source_processed_schema_version=item.processed_schema_version,
    )
    np.savez_compressed(
        prediction_path,
        **cast(dict[str, Any], prediction_payload),
    )

    policy_metadata = prediction_sample_policy_metadata()
    return {
        "sample_id": item.sample_id,
        "split": item.split,
        "text": item.text,
        "fps": item.fps,
        "num_frames": item.num_frames,
        "frame_valid_count": int(np.count_nonzero(item.frame_valid_mask)),
        "reference_artifact": portable_path(reference_path, path_formatter=path_formatter),
        "prediction_artifact": portable_path(prediction_path, path_formatter=path_formatter),
        "source_processed_sample_path": item.sample_path_value,
        "source_processed_schema_version": item.processed_schema_version,
        "target_channels": list(M0_TARGET_CHANNELS),
        "prediction_schema_version": str(prediction_payload["prediction_schema_version"].item()),
        **policy_metadata,
    }


def _panel_definition_from_mapping(
    payload: Mapping[str, object],
) -> QualitativePanelDefinition:
    required_fields = {"schema_version", "split", "sample_ids", "selection_rule"}
    missing_fields = sorted(required_fields.difference(payload))
    if missing_fields:
        formatted = ", ".join(missing_fields)
        raise QualitativeExportError(f"Panel definition is missing required field(s): {formatted}.")

    schema_version = payload["schema_version"]
    if schema_version != QUALITATIVE_PANEL_SCHEMA_VERSION:
        raise QualitativeExportError(
            "Panel definition schema version mismatch: "
            f"expected {QUALITATIVE_PANEL_SCHEMA_VERSION!r}, got {schema_version!r}."
        )

    split = payload["split"]
    if split != QUALITATIVE_PANEL_SPLIT:
        raise QualitativeExportError(f"Panel definition split must be 'val'; got {split!r}.")

    selection_rule = payload["selection_rule"]
    if not isinstance(selection_rule, str) or not selection_rule.strip():
        raise QualitativeExportError("Panel definition selection_rule must be a non-empty string.")

    sample_ids = _validate_panel_sample_ids(payload["sample_ids"])
    return QualitativePanelDefinition(
        split=QUALITATIVE_PANEL_SPLIT,
        sample_ids=sample_ids,
        selection_rule=selection_rule,
        schema_version=QUALITATIVE_PANEL_SCHEMA_VERSION,
    )


def _validate_panel_sample_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise QualitativeExportError("Panel definition sample_ids must be a list.")
    if not value:
        raise QualitativeExportError("Panel definition sample_ids must not be empty.")

    sample_ids: list[str] = []
    seen: set[str] = set()
    for index, raw_sample_id in enumerate(value):
        if not isinstance(raw_sample_id, str) or not raw_sample_id.strip():
            raise QualitativeExportError(
                f"Panel definition sample_ids[{index}] must be a non-empty string."
            )
        if raw_sample_id != raw_sample_id.strip():
            raise QualitativeExportError(
                f"Panel definition sample_ids[{index}] has leading or trailing whitespace."
            )
        if raw_sample_id in seen:
            raise QualitativeExportError(
                f"Panel definition has duplicate sample_id: {raw_sample_id!r}."
            )
        seen.add(raw_sample_id)
        sample_ids.append(raw_sample_id)
    return tuple(sample_ids)


def _validate_panel_split(panel: QualitativePanelDefinition) -> None:
    if panel.schema_version != QUALITATIVE_PANEL_SCHEMA_VERSION:
        raise QualitativeExportError(
            "Panel definition schema version mismatch: "
            f"expected {QUALITATIVE_PANEL_SCHEMA_VERSION!r}, got {panel.schema_version!r}."
        )
    if panel.split != QUALITATIVE_PANEL_SPLIT:
        raise QualitativeExportError(f"Panel definition split must be 'val'; got {panel.split!r}.")
    if not panel.sample_ids:
        raise QualitativeExportError("Panel definition sample_ids must not be empty.")


def _validation_records(
    records: Sequence[ProcessedModelingManifestRecord],
) -> list[ProcessedModelingManifestRecord]:
    validation_records: list[ProcessedModelingManifestRecord] = []
    for record in records:
        if record.split != QUALITATIVE_PANEL_SPLIT:
            raise QualitativeExportError(
                f"Qualitative panel generation requires validation records only; "
                f"record {record.sample_id!r} has split {record.split!r}."
            )
        validation_records.append(record)
    return validation_records


def _prediction_channel_array(
    prediction: BaselinePoseOutput,
    *,
    channel: str,
    expected_length: int,
) -> npt.NDArray[np.float32]:
    torch = _load_torch()
    tensor = getattr(prediction, channel)
    if not isinstance(tensor, torch.Tensor):
        raise QualitativeExportError(f"Prediction channel {channel!r} must be a tensor.")
    expected_shape = (1, expected_length, *M0_TARGET_CHANNEL_SHAPES[channel])
    if tuple(tensor.shape) != expected_shape:
        raise QualitativeExportError(
            f"Prediction channel {channel!r} has shape {tuple(tensor.shape)}; "
            f"expected {expected_shape}."
        )
    array = tensor.detach().cpu().numpy()[0].astype(np.float32, copy=False)
    return cast(npt.NDArray[np.float32], array)


def _panel_summary_payload(
    *,
    panel: QualitativePanelDefinition,
    records_path: Path,
    panel_definition_path: Path,
    checkpoint_path: Path,
    artifact_records: Sequence[Mapping[str, object]],
    path_formatter: Callable[[Path], str],
) -> dict[str, object]:
    return {
        "schema_version": QUALITATIVE_EXPORT_SCHEMA_VERSION,
        "panel_definition_path": portable_path(
            panel_definition_path,
            path_formatter=path_formatter,
        ),
        "records_path": portable_path(records_path, path_formatter=path_formatter),
        "checkpoint_path": portable_path(checkpoint_path, path_formatter=path_formatter),
        "sample_count": len(artifact_records),
        "sample_ids": list(panel.sample_ids),
        "selection_rule": panel.selection_rule,
        "split": panel.split,
        "target_channels": list(M0_TARGET_CHANNELS),
        "artifacts": list(artifact_records),
    }


def _sanitize_sample_id(sample_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    return sanitized or "sample"


def _load_panel_definition(
    path: Path | str,
    *,
    repo_root: Path | str | None,
) -> QualitativePanelDefinition:
    resolved_path = _resolve_path(path, repo_root=repo_root)
    return load_panel_definition(resolved_path)


def _resolve_path(path: Path | str, *, repo_root: Path | str | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if repo_root is not None:
        root = Path(repo_root).expanduser().resolve()
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root):
            raise QualitativeExportError(f"Path must stay under {root}: {path}")
        return resolved

    raise QualitativeExportError(
        f"Relative path requires explicit repo_root in qualitative export: {path}"
    )


def _require_qualitative_output_dirs(output_dir: Path) -> None:
    _require_existing_dir(output_dir, label="Qualitative output directory")
    _require_existing_dir(
        output_dir / REFERENCE_ARTIFACTS_DIRNAME,
        label="Qualitative reference artifact directory",
    )
    _require_existing_dir(
        output_dir / PREDICTION_ARTIFACTS_DIRNAME,
        label="Qualitative prediction artifact directory",
    )


def _require_existing_dir(path: Path, *, label: str) -> None:
    if not path.exists():
        raise QualitativeExportError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise QualitativeExportError(f"{label} is not a directory: {path}")


def _write_json_existing_parent(path: Path, payload: object) -> None:
    if not path.parent.is_dir():
        raise QualitativeExportError(f"JSON parent directory does not exist: {path.parent}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_existing_parent(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    if not path.parent.is_dir():
        raise QualitativeExportError(f"JSONL parent directory does not exist: {path.parent}")
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _load_prediction_helpers() -> tuple[Any, Any]:
    try:
        from .predict import load_baseline_predictor, predict_baseline_batch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "M0 qualitative export requires torch. "
                "Install the modeling extra or run inside the configured modeling environment."
            ) from exc
        raise

    return load_baseline_predictor, predict_baseline_batch


def _load_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "M0 qualitative prediction artifact writing requires torch. "
                "Install the modeling extra or run inside the configured modeling environment."
            ) from exc
        raise

    return torch
