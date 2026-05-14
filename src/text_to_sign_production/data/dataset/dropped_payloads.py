"""DroppedSample JSON payload serialization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from text_to_sign_production.core.ids import SampleSplit, SampleStatus
from text_to_sign_production.core.models import (
    DroppedSample,
    DroppedSampleGateSnapshot,
    DroppedSamplePoseSnapshot,
    DroppedSampleSourceSnapshot,
    GateDropIssueCode,
    GateDropStage,
    GateIssueCode,
    GateName,
)
from text_to_sign_production.data.dataset.schemas import (
    DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION,
)
from text_to_sign_production.data.dataset.validate import validate_dropped_sample

_ROOT_KEYS = frozenset(
    {"schema_version", "sample_id", "split", "drop_stage", "issue_codes", "source", "pose", "gate"}
)
_SOURCE_KEYS = frozenset(
    {
        "text",
        "source_video_id",
        "source_sentence_id",
        "source_sentence_name",
        "match_status",
        "unmatched_reason",
        "ambiguity_reasons",
        "source_issue_codes",
        "video_match_count",
        "keypoint_match_count",
    }
)
_POSE_KEYS = frozenset(
    {
        "candidate_available",
        "video_path",
        "keypoints_dir",
        "candidate_frame_count",
        "observed_frame_count",
        "video_metadata_readable",
        "video_metadata_error",
        "missing_frame_files",
        "viability_status",
        "viability_issue_codes",
        "viability_issue_messages",
    }
)
_GATE_KEYS = frozenset(
    {
        "prepared_sample_available",
        "frame_count",
        "valid_frame_count",
        "body_nonzero_frame_count",
        "face_nonzero_frame_count",
        "left_hand_nonzero_frame_count",
        "right_hand_nonzero_frame_count",
        "final_status",
        "terminal_gate",
        "failed_gates",
        "decision_issue_codes",
    }
)


def dropped_sample_to_record(sample: DroppedSample) -> dict[str, object]:
    """Serialize a DroppedSample into its JSON document record."""
    return {
        "schema_version": sample.schema_version,
        "sample_id": sample.sample_id,
        "split": sample.split.value,
        "drop_stage": sample.drop_stage.value,
        "issue_codes": [code.value for code in sample.issue_codes],
        "source": _source_to_record(sample.source),
        "pose": None if sample.pose is None else _pose_to_record(sample.pose),
        "gate": None if sample.gate is None else _gate_to_record(sample.gate),
    }


def dropped_sample_from_record(record: Mapping[str, object]) -> DroppedSample:
    """Parse a DroppedSample JSON document record exactly."""
    _require_exact_keys(record, _ROOT_KEYS, "dropped sample")
    schema_version = _text(record["schema_version"], "schema_version")
    if schema_version != DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION:
        raise ValueError(
            "DroppedSample schema_version is unsupported: "
            f"expected {DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION!r}, observed {schema_version!r}."
        )
    return DroppedSample(
        schema_version=schema_version,
        sample_id=_text(record["sample_id"], "sample_id"),
        split=SampleSplit(_text(record["split"], "split")),
        drop_stage=GateDropStage(_text(record["drop_stage"], "drop_stage")),
        issue_codes=tuple(
            GateDropIssueCode(_text(value, "issue_codes"))
            for value in _list(record["issue_codes"], "issue_codes")
        ),
        source=_source_from_record(_mapping(record["source"], "source")),
        pose=(
            None if record["pose"] is None else _pose_from_record(_mapping(record["pose"], "pose"))
        ),
        gate=(
            None if record["gate"] is None else _gate_from_record(_mapping(record["gate"], "gate"))
        ),
    )


def write_dropped_sample_payload(path: str | Path, sample: DroppedSample) -> None:
    """Write one DroppedSample JSON payload deterministically.

    Runs full dataset-layer semantic validation via ``validate_dropped_sample``
    before any filesystem side effect.  A ``ValueError`` is raised — and no
    file is created — if the sample is semantically invalid.
    """
    issues = validate_dropped_sample(sample)
    if issues:
        raise ValueError(
            f"Cannot write invalid DroppedSample for {sample.sample_id!r}: "
            f"{[issue.message for issue in issues]}"
        )
    payload_path = Path(path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(dropped_sample_to_record(sample), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_dropped_sample_payload(path: str | Path) -> DroppedSample:
    """Load one DroppedSample JSON payload."""
    payload_path = Path(path)
    try:
        loaded = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed DroppedSample JSON in {payload_path}: {exc.msg}") from exc
    return dropped_sample_from_record(_mapping(loaded, "dropped sample document"))


def _source_to_record(source: DroppedSampleSourceSnapshot) -> dict[str, object]:
    return {
        "text": source.text,
        "source_video_id": source.source_video_id,
        "source_sentence_id": source.source_sentence_id,
        "source_sentence_name": source.source_sentence_name,
        "match_status": source.match_status,
        "unmatched_reason": source.unmatched_reason,
        "ambiguity_reasons": list(source.ambiguity_reasons),
        "source_issue_codes": list(source.source_issue_codes),
        "video_match_count": source.video_match_count,
        "keypoint_match_count": source.keypoint_match_count,
    }


def _pose_to_record(pose: DroppedSamplePoseSnapshot) -> dict[str, object]:
    return {
        "candidate_available": pose.candidate_available,
        "video_path": pose.video_path,
        "keypoints_dir": pose.keypoints_dir,
        "candidate_frame_count": pose.candidate_frame_count,
        "observed_frame_count": pose.observed_frame_count,
        "video_metadata_readable": pose.video_metadata_readable,
        "video_metadata_error": pose.video_metadata_error,
        "missing_frame_files": pose.missing_frame_files,
        "viability_status": pose.viability_status,
        "viability_issue_codes": list(pose.viability_issue_codes),
        "viability_issue_messages": list(pose.viability_issue_messages),
    }


def _gate_to_record(gate: DroppedSampleGateSnapshot) -> dict[str, object]:
    return {
        "prepared_sample_available": gate.prepared_sample_available,
        "frame_count": gate.frame_count,
        "valid_frame_count": gate.valid_frame_count,
        "body_nonzero_frame_count": gate.body_nonzero_frame_count,
        "face_nonzero_frame_count": gate.face_nonzero_frame_count,
        "left_hand_nonzero_frame_count": gate.left_hand_nonzero_frame_count,
        "right_hand_nonzero_frame_count": gate.right_hand_nonzero_frame_count,
        "final_status": gate.final_status.value,
        "terminal_gate": None if gate.terminal_gate is None else gate.terminal_gate.value,
        "failed_gates": [gate_name.value for gate_name in gate.failed_gates],
        "decision_issue_codes": [code.value for code in gate.decision_issue_codes],
    }


def _source_from_record(record: Mapping[str, object]) -> DroppedSampleSourceSnapshot:
    _require_exact_keys(record, _SOURCE_KEYS, "source")
    return DroppedSampleSourceSnapshot(
        text=_optional_text(record["text"], "source.text"),
        source_video_id=_optional_text(record["source_video_id"], "source.source_video_id"),
        source_sentence_id=_optional_text(
            record["source_sentence_id"],
            "source.source_sentence_id",
        ),
        source_sentence_name=_optional_text(
            record["source_sentence_name"],
            "source.source_sentence_name",
        ),
        match_status=_text(record["match_status"], "source.match_status"),
        unmatched_reason=_optional_text(record["unmatched_reason"], "source.unmatched_reason"),
        ambiguity_reasons=tuple(
            _text(value, "source.ambiguity_reasons")
            for value in _list(record["ambiguity_reasons"], "source.ambiguity_reasons")
        ),
        source_issue_codes=tuple(
            _text(value, "source.source_issue_codes")
            for value in _list(record["source_issue_codes"], "source.source_issue_codes")
        ),
        video_match_count=_int(record["video_match_count"], "source.video_match_count"),
        keypoint_match_count=_int(record["keypoint_match_count"], "source.keypoint_match_count"),
    )


def _pose_from_record(record: Mapping[str, object]) -> DroppedSamplePoseSnapshot:
    _require_exact_keys(record, _POSE_KEYS, "pose")
    return DroppedSamplePoseSnapshot(
        candidate_available=_bool(record["candidate_available"], "pose.candidate_available"),
        video_path=_optional_text(record["video_path"], "pose.video_path"),
        keypoints_dir=_optional_text(record["keypoints_dir"], "pose.keypoints_dir"),
        candidate_frame_count=_optional_int(
            record["candidate_frame_count"],
            "pose.candidate_frame_count",
        ),
        observed_frame_count=_optional_int(
            record["observed_frame_count"],
            "pose.observed_frame_count",
        ),
        video_metadata_readable=_optional_bool(
            record["video_metadata_readable"],
            "pose.video_metadata_readable",
        ),
        video_metadata_error=_optional_text(
            record["video_metadata_error"],
            "pose.video_metadata_error",
        ),
        missing_frame_files=_optional_bool(
            record["missing_frame_files"],
            "pose.missing_frame_files",
        ),
        viability_status=_optional_text(record["viability_status"], "pose.viability_status"),
        viability_issue_codes=tuple(
            _text(value, "pose.viability_issue_codes")
            for value in _list(record["viability_issue_codes"], "pose.viability_issue_codes")
        ),
        viability_issue_messages=tuple(
            _text(value, "pose.viability_issue_messages")
            for value in _list(
                record["viability_issue_messages"],
                "pose.viability_issue_messages",
            )
        ),
    )


def _gate_from_record(record: Mapping[str, object]) -> DroppedSampleGateSnapshot:
    _require_exact_keys(record, _GATE_KEYS, "gate")
    return DroppedSampleGateSnapshot(
        prepared_sample_available=_bool(
            record["prepared_sample_available"],
            "gate.prepared_sample_available",
        ),
        frame_count=_int(record["frame_count"], "gate.frame_count"),
        valid_frame_count=_int(record["valid_frame_count"], "gate.valid_frame_count"),
        body_nonzero_frame_count=_int(
            record["body_nonzero_frame_count"],
            "gate.body_nonzero_frame_count",
        ),
        face_nonzero_frame_count=_int(
            record["face_nonzero_frame_count"],
            "gate.face_nonzero_frame_count",
        ),
        left_hand_nonzero_frame_count=_int(
            record["left_hand_nonzero_frame_count"],
            "gate.left_hand_nonzero_frame_count",
        ),
        right_hand_nonzero_frame_count=_int(
            record["right_hand_nonzero_frame_count"],
            "gate.right_hand_nonzero_frame_count",
        ),
        final_status=SampleStatus(_text(record["final_status"], "gate.final_status")),
        terminal_gate=(
            None
            if record["terminal_gate"] is None
            else GateName(_text(record["terminal_gate"], "gate.terminal_gate"))
        ),
        failed_gates=tuple(
            GateName(_text(value, "gate.failed_gates"))
            for value in _list(record["failed_gates"], "gate.failed_gates")
        ),
        decision_issue_codes=tuple(
            GateIssueCode(_text(value, "gate.decision_issue_codes"))
            for value in _list(record["decision_issue_codes"], "gate.decision_issue_codes")
        ),
    )


def _require_exact_keys(
    record: Mapping[str, object],
    expected_keys: frozenset[str],
    label: str,
) -> None:
    observed_keys = frozenset(record)
    missing = sorted(expected_keys.difference(observed_keys))
    extra = sorted(observed_keys.difference(expected_keys))
    if missing or extra:
        raise ValueError(f"{label} keys do not match: missing={missing}, extra={extra}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings.")
    return cast(Mapping[str, object], value)


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    return value


def _optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _int(value, label)


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a bool.")
    return value


def _optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, label)


__all__ = [
    "DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION",
    "load_dropped_sample_payload",
    "write_dropped_sample_payload",
]
