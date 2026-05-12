"""Dataset-layer validation for PreparedSample payloads and manifest rows."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.ids import CoordinateSpace
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    PassedManifestEntry,
    PoseTruth,
    PreparedSample,
    SourceTruth,
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
    if tensor.shape[0] != frame_count:
        _add(
            issues,
            DatasetValidationIssueCode.INVALID_POSE_ARRAY_SHAPE,
            "Pose tensor first dimension must match frame_count.",
            path,
        )
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
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
    if len(set(entry.issue_codes)) != len(entry.issue_codes):
        _add(
            issues,
            DatasetValidationIssueCode.DUPLICATE_ISSUE_CODE,
            "Dropped manifest issue codes cannot contain duplicates.",
            "issue_codes",
        )
    return tuple(issues)


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
    "validate_manifest_entry",
    "validate_payload_manifest_coherence",
    "validate_prepared_sample",
]
