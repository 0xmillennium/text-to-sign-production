"""PreparedSample payload persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.models import PoseTruth, PreparedSample, SourceTruth
from text_to_sign_production.data.gate.pose import CoordinateSpace
from text_to_sign_production.data.dataset.validate import validate_prepared_sample
from text_to_sign_production.data.gate.sources import SourceIssueCode


def write_prepared_sample_payload(path: str | Path, sample: PreparedSample) -> None:
    """Write one PreparedSample as a compressed NPZ artifact."""
    issues = validate_prepared_sample(sample)
    if issues:
        raise ValueError(f"Invalid prepared sample payload: {issues}")
    payload_path = Path(path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_metadata(sample), sort_keys=True)),
        "valid_frame_mask": np.asarray(sample.pose.valid_frame_mask, dtype=np.bool_),
        "body_xyc": np.asarray(sample.pose.body_xyc, dtype=np.float32),
        "face_xyc": np.asarray(sample.pose.face_xyc, dtype=np.float32),
        "left_hand_xyc": np.asarray(sample.pose.left_hand_xyc, dtype=np.float32),
        "right_hand_xyc": np.asarray(sample.pose.right_hand_xyc, dtype=np.float32),
    }
    _write_deterministic_npz(payload_path, arrays)


def load_prepared_sample_payload(path: str | Path) -> PreparedSample:
    """Load one PreparedSample from a compressed NPZ artifact."""
    payload_path = Path(path)
    try:
        with np.load(payload_path, allow_pickle=False) as loaded:
            metadata = _metadata_from_json(str(np.asarray(loaded["metadata_json"]).item()))
            sample = _sample_from_metadata_and_arrays(metadata, loaded)
    except (BadZipFile, EOFError, OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Prepared sample payload could not be loaded from {payload_path}: {exc}"
        ) from exc
    issues = validate_prepared_sample(sample)
    if issues:
        raise ValueError(f"Invalid prepared sample payload: {issues}")
    return sample


def _write_deterministic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        for key in sorted(arrays):
            info = ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, _npy_bytes(np.asarray(arrays[key])))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _metadata(sample: PreparedSample) -> dict[str, Any]:
    return {
        "schema_version": sample.schema_version,
        "source": {
            "sample_id": sample.source.sample_id,
            "split": sample.source.split.value,
            "text": sample.source.text,
            "canonical_normalized_text": sample.source.canonical_normalized_text,
            "fps": sample.source.fps,
            "source_video_id": sample.source.source_video_id,
            "source_sentence_id": sample.source.source_sentence_id,
            "source_sentence_name": sample.source.source_sentence_name,
            "source_issue_codes": [code.value for code in sample.source.source_issue_codes],
        },
        "pose": {
            "coordinate_space": sample.pose.coordinate_space.value,
            "frame_count": sample.pose.frame_count,
            "selected_person_indices": list(sample.pose.selected_person_indices),
            "tracked_target_missing_frame_count": (sample.pose.tracked_target_missing_frame_count),
            "continuity_break_count": sample.pose.continuity_break_count,
            "reanchor_count": sample.pose.reanchor_count,
            "body_nonzero_frame_count": sample.pose.body_nonzero_frame_count,
            "face_nonzero_frame_count": sample.pose.face_nonzero_frame_count,
            "left_hand_nonzero_frame_count": sample.pose.left_hand_nonzero_frame_count,
            "right_hand_nonzero_frame_count": sample.pose.right_hand_nonzero_frame_count,
        },
    }


def _metadata_from_json(value: str) -> Mapping[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, Mapping):
        raise ValueError("Payload metadata must be a JSON object.")
    if any(not isinstance(key, str) for key in loaded):
        raise ValueError("Payload metadata keys must be strings.")
    return cast(Mapping[str, Any], loaded)


def _sample_from_metadata_and_arrays(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> PreparedSample:
    source = _require_mapping(metadata["source"], "source")
    pose = _require_mapping(metadata["pose"], "pose")
    return PreparedSample(
        schema_version=_text(metadata["schema_version"], "schema_version"),
        source=SourceTruth(
            sample_id=_text(source["sample_id"], "source.sample_id"),
            split=SampleSplit(_text(source["split"], "source.split")),
            text=_text(source["text"], "source.text"),
            canonical_normalized_text=_text(
                source["canonical_normalized_text"],
                "source.canonical_normalized_text",
            ),
            fps=_float(source["fps"], "source.fps"),
            source_video_id=_text(source["source_video_id"], "source.source_video_id"),
            source_sentence_id=_text(
                source["source_sentence_id"],
                "source.source_sentence_id",
            ),
            source_sentence_name=_text(
                source["source_sentence_name"],
                "source.source_sentence_name",
            ),
            source_issue_codes=tuple(
                SourceIssueCode(_text(value, "source.source_issue_codes"))
                for value in _list(source["source_issue_codes"], "source.source_issue_codes")
            ),
        ),
        pose=PoseTruth(
            coordinate_space=CoordinateSpace(
                _text(pose["coordinate_space"], "pose.coordinate_space")
            ),
            frame_count=_int(pose["frame_count"], "pose.frame_count"),
            valid_frame_mask=np.asarray(arrays["valid_frame_mask"], dtype=np.bool_),
            selected_person_indices=tuple(
                None if value is None else _int(value, "pose.selected_person_indices")
                for value in _list(
                    pose["selected_person_indices"],
                    "pose.selected_person_indices",
                )
            ),
            tracked_target_missing_frame_count=_int(
                pose["tracked_target_missing_frame_count"],
                "pose.tracked_target_missing_frame_count",
            ),
            continuity_break_count=_int(
                pose["continuity_break_count"],
                "pose.continuity_break_count",
            ),
            reanchor_count=_int(pose["reanchor_count"], "pose.reanchor_count"),
            body_xyc=np.asarray(arrays["body_xyc"], dtype=np.float32),
            face_xyc=np.asarray(arrays["face_xyc"], dtype=np.float32),
            left_hand_xyc=np.asarray(arrays["left_hand_xyc"], dtype=np.float32),
            right_hand_xyc=np.asarray(arrays["right_hand_xyc"], dtype=np.float32),
            body_nonzero_frame_count=_int(
                pose["body_nonzero_frame_count"],
                "pose.body_nonzero_frame_count",
            ),
            face_nonzero_frame_count=_int(
                pose["face_nonzero_frame_count"],
                "pose.face_nonzero_frame_count",
            ),
            left_hand_nonzero_frame_count=_int(
                pose["left_hand_nonzero_frame_count"],
                "pose.left_hand_nonzero_frame_count",
            ),
            right_hand_nonzero_frame_count=_int(
                pose["right_hand_nonzero_frame_count"],
                "pose.right_hand_nonzero_frame_count",
            ),
        ),
    )


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings.")
    return cast(Mapping[str, Any], value)


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    return value


def _int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    return value


def _float(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


__all__ = [
    "load_prepared_sample_payload",
    "write_prepared_sample_payload",
]
