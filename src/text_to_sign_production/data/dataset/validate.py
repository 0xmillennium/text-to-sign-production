"""Dataset-layer validation for PreparedSample payloads and manifest rows."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.ids import CoordinateSpace, SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    DroppedSample,
    GateDropStage,
    PassedManifestEntry,
    PoseTruth,
    PreparedSample,
    SourceTruth,
)
from text_to_sign_production.data.dataset.schemas import (
    DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION,
)
from text_to_sign_production.data.dataset.types import (
    DatasetValidationIssue,
    DatasetValidationIssueCode,
    ManifestEntry,
)


def validate_prepared_sample(sample: PreparedSample) -> tuple[DatasetValidationIssue, ...]:
    """Validate PreparedSample payload invariants."""
    issues: list[DatasetValidationIssue] = []
    _validate_schema(issues, sample.schema_version, "schema_version")
    _validate_source_truth(issues, sample.source)
    _validate_pose_truth(issues, sample.pose)
    if sample.source.sample_id and sample.pose.frame_count <= 0:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_FRAME_COUNT,
            "Prepared sample pose frame_count must be positive.",
            "pose.frame_count",
        )
    return tuple(issues)


def validate_manifest_entry(entry: ManifestEntry) -> tuple[DatasetValidationIssue, ...]:
    """Validate passed or dropped manifest row invariants."""
    if isinstance(entry, PassedManifestEntry):
        return _validate_passed_entry(entry)
    if isinstance(entry, DroppedManifestEntry):
        return _validate_dropped_entry(entry)
    return (
        DatasetValidationIssue(
            code=DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
            message="Manifest entry must be passed or dropped.",
        ),
    )


def validate_payload_manifest_coherence(
    sample: PreparedSample,
    entry: PassedManifestEntry,
) -> tuple[DatasetValidationIssue, ...]:
    """Validate handoff coherence between a PreparedSample payload and passed row."""
    issues: list[DatasetValidationIssue] = []
    expected = {
        "sample_id": sample.source.sample_id,
        "split": sample.source.split,
        "text": sample.source.text,
        "fps": sample.source.fps,
        "frame_count": sample.pose.frame_count,
        "source_video_id": sample.source.source_video_id,
        "source_sentence_id": sample.source.source_sentence_id,
        "source_sentence_name": sample.source.source_sentence_name,
        "valid_frame_count": int(np.count_nonzero(sample.pose.valid_frame_mask)),
        "body_nonzero_frame_count": sample.pose.body_nonzero_frame_count,
        "face_nonzero_frame_count": sample.pose.face_nonzero_frame_count,
        "left_hand_nonzero_frame_count": sample.pose.left_hand_nonzero_frame_count,
        "right_hand_nonzero_frame_count": sample.pose.right_hand_nonzero_frame_count,
    }
    for field_name, expected_value in expected.items():
        if getattr(entry, field_name) != expected_value:
            _add(
                issues,
                DatasetValidationIssueCode.MANIFEST_PAYLOAD_MISMATCH,
                f"Passed manifest {field_name} does not match payload.",
                field_name,
            )
    return tuple(issues)


def validate_dropped_sample(
    sample: DroppedSample,
) -> tuple[DatasetValidationIssue, ...]:
    """Validate DroppedSample JSON payload invariants."""
    issues: list[DatasetValidationIssue] = []
    if sample.schema_version != DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_SCHEMA_VERSION,
            "DroppedSample schema_version is unsupported.",
            "schema_version",
        )
    _validate_text(
        issues,
        sample.sample_id,
        "sample_id",
        DatasetValidationIssueCode.EMPTY_SAMPLE_ID,
    )
    if sample.drop_stage not in {GateDropStage.SOURCE, GateDropStage.POSE, GateDropStage.GATES}:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
            "DroppedSample drop_stage must be source, pose, or gates.",
            "drop_stage",
        )
    if not sample.issue_codes:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
            "DroppedSample issue_codes must be non-empty.",
            "issue_codes",
        )
    if sample.source is None:
        _add(
            issues,
            DatasetValidationIssueCode.MISSING_SOURCE_TRUTH,
            "DroppedSample source snapshot is required.",
            "source",
        )
    else:
        _validate_dropped_sample_source(issues, sample)
    _validate_dropped_sample_stage_evidence(issues, sample)
    return tuple(issues)


def validate_dropped_manifest_payload_coherence(
    *,
    entry: DroppedManifestEntry,
    sample: DroppedSample,
) -> tuple[DatasetValidationIssue, ...]:
    """Validate identity coherence between a dropped manifest row and JSON payload."""
    issues: list[DatasetValidationIssue] = []
    expected = {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "drop_stage": sample.drop_stage,
        "issue_codes": sample.issue_codes,
    }
    for field_name, expected_value in expected.items():
        if getattr(entry, field_name) != expected_value:
            _add(
                issues,
                DatasetValidationIssueCode.MANIFEST_PAYLOAD_MISMATCH,
                f"Dropped manifest {field_name} does not match payload.",
                field_name,
            )
    _validate_text(
        issues,
        entry.dropped_sample_ref,
        "dropped_sample_ref",
        DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
    )
    if entry.dropped_sample_ref and not entry.dropped_sample_ref.endswith(".json"):
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
            "Dropped manifest dropped_sample_ref must point to a .json payload.",
            "dropped_sample_ref",
        )
    if entry.dropped_sample_ref:
        expected_ref = _canonical_dropped_sample_ref(entry.split.value, entry.sample_id)
        if entry.dropped_sample_ref != expected_ref:
            _add(
                issues,
                DatasetValidationIssueCode.MANIFEST_PAYLOAD_MISMATCH,
                "Dropped manifest dropped_sample_ref does not match canonical sample ref.",
                "dropped_sample_ref",
            )
    return tuple(issues)


def _validate_source_truth(
    issues: list[DatasetValidationIssue],
    source: SourceTruth,
) -> None:
    _validate_text(
        issues, source.sample_id, "source.sample_id", DatasetValidationIssueCode.EMPTY_SAMPLE_ID
    )
    _validate_text(issues, source.text, "source.text", DatasetValidationIssueCode.EMPTY_TEXT)
    _validate_text(
        issues,
        source.source_video_id,
        "source.source_video_id",
        DatasetValidationIssueCode.MISSING_SOURCE_TRUTH,
    )
    _validate_text(
        issues,
        source.source_sentence_id,
        "source.source_sentence_id",
        DatasetValidationIssueCode.MISSING_SOURCE_TRUTH,
    )
    _validate_text(
        issues,
        source.source_sentence_name,
        "source.source_sentence_name",
        DatasetValidationIssueCode.MISSING_SOURCE_TRUTH,
    )
    if source.fps <= 0:
        _add(issues, DatasetValidationIssueCode.INVALID_FPS, "FPS must be positive.", "source.fps")
    if len(set(source.source_issue_codes)) != len(source.source_issue_codes):
        _add(
            issues,
            DatasetValidationIssueCode.DUPLICATE_ISSUE_CODE,
            "Source issue codes cannot contain duplicates.",
            "source.source_issue_codes",
        )


def _validate_pose_truth(issues: list[DatasetValidationIssue], pose: PoseTruth) -> None:
    if pose.coordinate_space is not CoordinateSpace.NORMALIZED_IMAGE:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_COORDINATE_SPACE,
            "Pose coordinate_space must be normalized_image.",
            "pose.coordinate_space",
        )
    if pose.frame_count <= 0:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_FRAME_COUNT,
            "Pose frame_count must be positive.",
            "pose.frame_count",
        )
    if pose.valid_frame_mask.shape != (pose.frame_count,):
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_FRAME_MASK,
            "valid_frame_mask shape must match frame_count.",
            "pose.valid_frame_mask",
        )
    if pose.valid_frame_mask.dtype != np.dtype(np.bool_):
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_FRAME_MASK,
            "valid_frame_mask must use bool dtype.",
            "pose.valid_frame_mask",
        )
    if len(pose.selected_person_indices) != pose.frame_count:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_TRACKING_SUMMARY,
            "selected_person_indices length must match frame_count.",
            "pose.selected_person_indices",
        )
    for path, tensor in (
        ("pose.body_xyc", pose.body_xyc),
        ("pose.face_xyc", pose.face_xyc),
        ("pose.left_hand_xyc", pose.left_hand_xyc),
        ("pose.right_hand_xyc", pose.right_hand_xyc),
    ):
        _validate_xyc_tensor(issues, tensor, pose.frame_count, path)
    for path, count in (
        ("pose.tracked_target_missing_frame_count", pose.tracked_target_missing_frame_count),
        ("pose.continuity_break_count", pose.continuity_break_count),
        ("pose.reanchor_count", pose.reanchor_count),
        ("pose.body_nonzero_frame_count", pose.body_nonzero_frame_count),
        ("pose.face_nonzero_frame_count", pose.face_nonzero_frame_count),
        ("pose.left_hand_nonzero_frame_count", pose.left_hand_nonzero_frame_count),
        ("pose.right_hand_nonzero_frame_count", pose.right_hand_nonzero_frame_count),
    ):
        if count < 0 or count > pose.frame_count:
            _add(
                issues,
                DatasetValidationIssueCode.INVALID_CHANNEL_COUNTS,
                "Pose counts must be between 0 and frame_count.",
                path,
            )


def _validate_xyc_tensor(
    issues: list[DatasetValidationIssue],
    tensor: np.ndarray,
    frame_count: int,
    path: str,
) -> None:
    shape_valid = True
    if tensor.ndim == 0 or tensor.shape[0] != frame_count:
        shape_valid = False
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_POSE_ARRAY_SHAPE,
            "Pose tensor first dimension must match frame_count.",
            path,
        )
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        shape_valid = False
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_POSE_ARRAY_SHAPE,
            "Pose tensor must have shape (frames, landmarks, xyc).",
            path,
        )
    if tensor.dtype != np.dtype(np.float32):
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_POSE_ARRAY_DTYPE,
            "Pose tensor must be float32.",
            path,
        )
    if not shape_valid:
        return
    if not np.all(np.isfinite(tensor)):
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_POSE_ARRAY_VALUE,
            "Pose tensor must contain only finite values.",
            path,
        )
        return
    confidence = tensor[..., 2]
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_POSE_CONFIDENCE_RANGE,
            "Pose tensor confidence values must be within [0.0, 1.0].",
            path,
        )


def _validate_passed_entry(entry: PassedManifestEntry) -> tuple[DatasetValidationIssue, ...]:
    issues: list[DatasetValidationIssue] = []
    _validate_schema(issues, entry.schema_version, "schema_version")
    for value, path, code in (
        (entry.sample_id, "sample_id", DatasetValidationIssueCode.EMPTY_SAMPLE_ID),
        (entry.payload_ref, "payload_ref", DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY),
        (entry.text, "text", DatasetValidationIssueCode.EMPTY_TEXT),
        (
            entry.source_video_id,
            "source_video_id",
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
        ),
        (
            entry.source_sentence_id,
            "source_sentence_id",
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
        ),
        (
            entry.source_sentence_name,
            "source_sentence_name",
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
        ),
    ):
        _validate_text(issues, value, path, code)
    if entry.fps <= 0:
        _add(issues, DatasetValidationIssueCode.INVALID_FPS, "FPS must be positive.", "fps")
    if entry.frame_count <= 0:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_FRAME_COUNT,
            "frame_count must be positive.",
            "frame_count",
        )
    for path, count in (
        ("valid_frame_count", entry.valid_frame_count),
        ("body_nonzero_frame_count", entry.body_nonzero_frame_count),
        ("face_nonzero_frame_count", entry.face_nonzero_frame_count),
        ("left_hand_nonzero_frame_count", entry.left_hand_nonzero_frame_count),
        ("right_hand_nonzero_frame_count", entry.right_hand_nonzero_frame_count),
    ):
        if count < 0 or count > entry.frame_count:
            _add(
                issues,
                DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
                "Manifest frame counts must be between 0 and frame_count.",
                path,
            )
    return tuple(issues)


def _validate_dropped_entry(entry: DroppedManifestEntry) -> tuple[DatasetValidationIssue, ...]:
    issues: list[DatasetValidationIssue] = []
    _validate_schema(issues, entry.schema_version, "schema_version")
    _validate_text(
        issues,
        entry.sample_id,
        "sample_id",
        DatasetValidationIssueCode.EMPTY_SAMPLE_ID,
    )
    if not entry.issue_codes:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
            "Dropped manifest entries require at least one issue code.",
            "issue_codes",
        )
    if entry.drop_stage not in {GateDropStage.SOURCE, GateDropStage.POSE, GateDropStage.GATES}:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
            "Dropped manifest drop_stage must be source, pose, or gates.",
            "drop_stage",
        )
    _validate_text(
        issues,
        entry.dropped_sample_ref,
        "dropped_sample_ref",
        DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
    )
    if entry.dropped_sample_ref and not entry.dropped_sample_ref.endswith(".json"):
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
            "Dropped manifest dropped_sample_ref must point to a .json payload.",
            "dropped_sample_ref",
        )
    if len(set(entry.issue_codes)) != len(entry.issue_codes):
        _add(
            issues,
            DatasetValidationIssueCode.DUPLICATE_ISSUE_CODE,
            "Dropped manifest issue codes cannot contain duplicates.",
            "issue_codes",
        )
    return tuple(issues)


def _validate_dropped_sample_source(
    issues: list[DatasetValidationIssue],
    sample: DroppedSample,
) -> None:
    source = sample.source
    for path, count in (
        ("source.video_match_count", source.video_match_count),
        ("source.keypoint_match_count", source.keypoint_match_count),
    ):
        if count < 0:
            _add(
                issues,
                DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
                "DroppedSample source counts cannot be negative.",
                path,
            )
    if len(set(sample.issue_codes)) != len(sample.issue_codes):
        _add(
            issues,
            DatasetValidationIssueCode.DUPLICATE_ISSUE_CODE,
            "DroppedSample issue codes cannot contain duplicates.",
            "issue_codes",
        )


def _validate_dropped_sample_stage_evidence(
    issues: list[DatasetValidationIssue],
    sample: DroppedSample,
) -> None:
    if sample.drop_stage is GateDropStage.SOURCE:
        if sample.pose is not None or sample.gate is not None:
            _add(
                issues,
                DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
                "Source-stage dropped samples cannot carry pose or gate evidence.",
                "drop_stage",
            )
        return
    if sample.drop_stage is GateDropStage.POSE:
        if sample.pose is None or sample.gate is not None:
            _add(
                issues,
                DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
                "Pose-stage dropped samples require pose evidence only.",
                "drop_stage",
            )
        if sample.pose is not None:
            _validate_dropped_sample_pose_counts(issues, sample)
        return
    if sample.drop_stage is GateDropStage.GATES:
        if sample.pose is None or sample.gate is None:
            _add(
                issues,
                DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
                "Gate-stage dropped samples require pose and gate evidence.",
                "drop_stage",
            )
        if sample.pose is not None:
            _validate_dropped_sample_pose_counts(issues, sample)
        if sample.gate is not None:
            _validate_dropped_sample_gate(issues, sample)


def _validate_dropped_sample_pose_counts(
    issues: list[DatasetValidationIssue],
    sample: DroppedSample,
) -> None:
    if sample.pose is None:
        return
    for path, count in (
        ("pose.candidate_frame_count", sample.pose.candidate_frame_count),
        ("pose.observed_frame_count", sample.pose.observed_frame_count),
    ):
        if count is not None and count < 0:
            _add(
                issues,
                DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
                "DroppedSample pose counts cannot be negative.",
                path,
            )


def _validate_dropped_sample_gate(
    issues: list[DatasetValidationIssue],
    sample: DroppedSample,
) -> None:
    if sample.gate is None:
        return
    for path, count in (
        ("gate.frame_count", sample.gate.frame_count),
        ("gate.valid_frame_count", sample.gate.valid_frame_count),
        ("gate.body_nonzero_frame_count", sample.gate.body_nonzero_frame_count),
        ("gate.face_nonzero_frame_count", sample.gate.face_nonzero_frame_count),
        ("gate.left_hand_nonzero_frame_count", sample.gate.left_hand_nonzero_frame_count),
        ("gate.right_hand_nonzero_frame_count", sample.gate.right_hand_nonzero_frame_count),
    ):
        if count < 0:
            _add(
                issues,
                DatasetValidationIssueCode.INVALID_CHANNEL_COUNTS,
                "DroppedSample gate counts cannot be negative.",
                path,
            )
    if sample.gate.final_status is not SampleStatus.DROPPED:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
            "Gate-stage DroppedSample final_status must be dropped.",
            "gate.final_status",
        )
    if not sample.gate.failed_gates and not sample.gate.decision_issue_codes:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_MANIFEST_ENTRY,
            "Gate-stage DroppedSample must carry failed gates or decision issue codes.",
            "gate",
        )


def _canonical_dropped_sample_ref(split: str, sample_id: str) -> str:
    return f"dropped/{split}/{sample_id}.json"


def _validate_schema(
    issues: list[DatasetValidationIssue],
    value: str,
    path: str,
) -> None:
    _validate_text(issues, value, path, DatasetValidationIssueCode.INVALID_SCHEMA_VERSION)


def _validate_text(
    issues: list[DatasetValidationIssue],
    value: str,
    path: str,
    code: DatasetValidationIssueCode,
) -> None:
    if not isinstance(value, str) or not value.strip():
        _add(issues, code, f"{path} must be a non-blank string.", path)


def _add(
    issues: list[DatasetValidationIssue],
    code: DatasetValidationIssueCode,
    message: str,
    field_path: str | None = None,
) -> None:
    issues.append(DatasetValidationIssue(code=code, message=message, field_path=field_path))


__all__ = [
    "ManifestEntry",
    "DatasetValidationIssue",
    "DatasetValidationIssueCode",
    "validate_dropped_manifest_payload_coherence",
    "validate_dropped_sample",
    "validate_manifest_entry",
    "validate_payload_manifest_coherence",
    "validate_prepared_sample",
]
