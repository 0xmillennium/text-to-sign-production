"""Full-split M0 prediction artifact writer."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

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
from text_to_sign_production.modeling.models import BaselinePoseOutput
from text_to_sign_production.modeling.training.config import BaselineTrainingConfig

from .predict import load_baseline_predictor, predict_baseline_batch
from .schemas import (
    build_prediction_manifest_row,
    build_prediction_sample_payload,
    validate_prediction_sample_payload,
)


@dataclass(frozen=True, slots=True)
class SplitPredictionWriteResult:
    """Artifacts produced by one deterministic split prediction export."""

    split: str
    manifest_path: Path
    samples_dir: Path
    sample_paths: tuple[Path, ...]
    sample_count: int
    checkpoint_path: Path


class PredictionWriteError(ValueError):
    """Raised when prediction artifacts cannot be written safely."""


def write_split_predictions(
    config: BaselineTrainingConfig,
    *,
    run_name: str,
    split: str,
    manifest_path: Path | str,
    output_dir: Path | str,
    checkpoint_path: Path | str,
    data_root: Path | str | None = None,
    limit_prediction_samples: int | None = None,
    manifest_path_formatter: Callable[[Path], str],
) -> SplitPredictionWriteResult:
    """Write full-BFH prediction samples and an evaluation-compatible manifest."""

    resolved_limit = _positive_optional_limit(
        limit_prediction_samples,
        label="limit_prediction_samples",
    )
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    samples_dir = resolved_output_dir / "samples"
    output_manifest_path = resolved_output_dir / "manifest.jsonl"
    _require_existing_dir(resolved_output_dir, label="Prediction output directory")
    _require_existing_dir(samples_dir, label="Prediction samples directory")
    _require_absent_file(output_manifest_path, label="Prediction manifest")

    records = _selected_prediction_records(
        manifest_path,
        split=split,
        data_root=data_root,
        limit=resolved_limit,
    )
    if not records:
        raise PredictionWriteError(
            f"No prediction records were selected for split {split!r}; "
            "prediction output would be empty."
        )
    predictor = load_baseline_predictor(config, checkpoint_path=checkpoint_path)

    manifest_rows: list[object] = []
    sample_paths: list[Path] = []
    generated_paths: dict[Path, str] = {}
    for record in records:
        pose_sample = load_processed_pose_sample(record)
        item = ProcessedPoseItem.from_manifest_and_sample(record, pose_sample)
        batch = collate_processed_pose_samples([item])
        prediction = predict_baseline_batch(
            predictor.model,
            batch,
            device=predictor.device,
        )
        prediction_path = samples_dir / f"{_sanitize_sample_id(item.sample_id)}.npz"
        _require_unique_prediction_path(
            prediction_path,
            sample_id=item.sample_id,
            generated_paths=generated_paths,
        )
        _require_absent_file(prediction_path, label=f"Prediction sample {item.sample_id!r}")
        prediction_payload = _prediction_payload(item, prediction)
        validate_prediction_sample_payload(prediction_payload)
        np.savez_compressed(prediction_path, **cast(dict[str, Any], prediction_payload))
        sample_paths.append(prediction_path)
        manifest_rows.append(
            build_prediction_manifest_row(
                run_name=run_name,
                split=split,
                sample_id=item.sample_id,
                text=item.text,
                reference_sample_path=_manifest_path_string(
                    record.sample_path,
                    formatter=manifest_path_formatter,
                ),
                prediction_sample_path=_manifest_path_string(
                    prediction_path,
                    formatter=manifest_path_formatter,
                ),
                reference_num_frames=item.num_frames,
                prediction_num_frames=item.length,
                checkpoint_path=_manifest_path_string(
                    predictor.checkpoint_path,
                    formatter=manifest_path_formatter,
                ),
            )
        )

    _write_jsonl_existing_parent(output_manifest_path, manifest_rows)
    return SplitPredictionWriteResult(
        split=split,
        manifest_path=output_manifest_path,
        samples_dir=samples_dir,
        sample_paths=tuple(sample_paths),
        sample_count=len(sample_paths),
        checkpoint_path=predictor.checkpoint_path,
    )


def _selected_prediction_records(
    manifest_path: Path | str,
    *,
    split: str,
    data_root: Path | str | None,
    limit: int | None,
) -> list[ProcessedModelingManifestRecord]:
    records = sorted(
        read_processed_modeling_manifest(
            manifest_path,
            split=split,
            data_root=data_root,
        ),
        key=lambda record: record.sample_id,
    )
    if limit is None:
        return records
    return records[:limit]


def _prediction_payload(
    item: ProcessedPoseItem,
    prediction: BaselinePoseOutput,
) -> dict[str, npt.NDArray[Any]]:
    prediction_arrays = {
        channel: _prediction_channel_array(
            prediction,
            channel=channel,
            expected_length=item.length,
        )
        for channel in M0_TARGET_CHANNELS
    }
    return build_prediction_sample_payload(
        prediction_arrays,
        frame_valid_mask=item.frame_valid_mask,
        selected_person_index=item.selected_person_index,
        source_processed_schema_version=item.processed_schema_version,
    )


def _prediction_channel_array(
    prediction: BaselinePoseOutput,
    *,
    channel: str,
    expected_length: int,
) -> npt.NDArray[np.float32]:
    tensor = getattr(prediction, channel)
    expected_shape = (1, expected_length, *M0_TARGET_CHANNEL_SHAPES[channel])
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Prediction channel {channel!r} has shape {tuple(tensor.shape)}; "
            f"expected {expected_shape}."
        )
    array = tensor.detach().cpu().numpy()[0].astype(np.float32, copy=False)
    return cast(npt.NDArray[np.float32], array)


def _sanitize_sample_id(sample_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    return sanitized or "sample"


def _positive_optional_limit(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise PredictionWriteError(f"{label} must be positive when set.")
    return value


def _require_existing_dir(path: Path, *, label: str) -> None:
    if not path.exists():
        raise PredictionWriteError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise PredictionWriteError(f"{label} is not a directory: {path}")


def _require_absent_file(path: Path, *, label: str) -> None:
    if path.exists():
        raise PredictionWriteError(f"{label} already exists and will not be overwritten: {path}")


def _require_unique_prediction_path(
    path: Path,
    *,
    sample_id: str,
    generated_paths: dict[Path, str],
) -> None:
    existing_sample_id = generated_paths.get(path)
    if existing_sample_id is not None:
        raise PredictionWriteError(
            "Prediction filename collision after sanitizing sample_id values: "
            f"{existing_sample_id!r} and {sample_id!r} both map to {path.name!r}."
        )
    generated_paths[path] = sample_id


def _manifest_path_string(path: Path, *, formatter: Callable[[Path], str]) -> str:
    value = formatter(path)
    if not isinstance(value, str):
        raise PredictionWriteError(
            f"Prediction manifest path formatter must return str, got {type(value).__name__}."
        )
    stripped = value.strip()
    if not stripped:
        raise PredictionWriteError("Prediction manifest paths must not be blank.")
    portable_path = PurePosixPath(stripped)
    if portable_path.is_absolute() or ".." in portable_path.parts:
        raise PredictionWriteError(
            f"Prediction manifest paths must be project-root-relative POSIX paths; got {value!r}."
        )
    return portable_path.as_posix()


def _write_jsonl_existing_parent(path: Path, records: list[object]) -> None:
    if not path.parent.is_dir():
        raise PredictionWriteError(
            f"Prediction manifest parent directory does not exist: {path.parent}"
        )
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_record() if hasattr(record, "to_record") else record
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


__all__ = [
    "PredictionWriteError",
    "SplitPredictionWriteResult",
    "write_split_predictions",
]
