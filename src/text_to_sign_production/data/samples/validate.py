"""Typed semantic validation for sample payload and manifest contracts."""

from __future__ import annotations

from text_to_sign_production.data.samples._shared.validate import (
    validate_drop_details as _validate_drop_details,
    validate_drop_reasons as _validate_drop_reasons,
    validate_frame_quality_summary as _validate_frame_quality_summary,
    validate_pose_payload as _validate_pose_payload,
    validate_selected_person_metadata as _validate_selected_person_metadata,
)
from text_to_sign_production.data.samples._shared.validation_primitives import (
    add_issue as _add_issue,
    validate_frame_count_value as _validate_frame_count_value,
    validate_non_empty_text as _validate_non_empty_text,
    validate_optional_fps_value as _validate_optional_fps_value,
    validate_schema_version_value as _validate_schema_version_value,
)
from text_to_sign_production.data.samples.types import (
    DroppedDebugMaterializationOutcome,
    DroppedManifestEntry,
    DroppedMaterializationLifecycle,
    ManifestEntry,
    PassedManifestEntry,
    ProcessedSamplePayload,
    SampleStatus,
    SampleValidationIssue,
)

__all__ = ("validate_manifest_entry", "validate_payload")


def validate_payload(payload: ProcessedSamplePayload) -> list[SampleValidationIssue]:
    """Validate semantic invariants of a typed processed sample payload."""
    issues: list[SampleValidationIssue] = []

    _validate_schema_version_value(issues, payload.schema_version, label="Payload")
    _validate_non_empty_text(issues, payload.sample_id, label="Payload sample_id")
    _validate_non_empty_text(issues, payload.text, label="Payload text")
    num_frames = _validate_frame_count_value(
        issues,
        payload.num_frames,
        label="Payload num_frames",
        allow_zero=False,
    )
    _validate_optional_fps_value(issues, payload.fps, label="Payload fps")
    _validate_selected_person_metadata(
        issues,
        payload.selected_person,
        num_frames=num_frames,
        label="Payload selected_person",
    )
    _validate_frame_quality_summary(
        issues,
        payload.frame_quality,
        num_frames=num_frames,
        label="Payload frame_quality",
    )
    _validate_pose_payload(issues, payload.pose, num_frames=num_frames, label="Payload pose")

    return issues


def validate_manifest_entry(entry: ManifestEntry) -> list[SampleValidationIssue]:
    """Validate semantic invariants of a typed manifest entry."""
    issues: list[SampleValidationIssue] = []

    if isinstance(entry, PassedManifestEntry):
        _validate_passed_entry(issues, entry)
    elif isinstance(entry, DroppedManifestEntry):
        _validate_dropped_entry(issues, entry)
    else:
        _add_issue(
            issues,
            "invalid_manifest_entry_type",
            f"Manifest entry must be typed as passed or dropped, got {type(entry).__name__}.",
        )

    return issues


def _validate_passed_entry(
    issues: list[SampleValidationIssue],
    entry: PassedManifestEntry,
) -> None:
    _validate_schema_version_value(issues, entry.schema_version, label="Passed manifest entry")
    if entry.status != SampleStatus.PASSED:
        _add_issue(
            issues,
            "invalid_passed_entry_status",
            f"PassedManifestEntry status must be 'passed', got {entry.status!r}.",
        )

    for value, label in (
        (entry.sample_id, "Passed manifest sample_id"),
        (entry.text, "Passed manifest text"),
        (entry.sample_path, "Passed manifest sample_path"),
        (entry.source_video_id, "Passed manifest source_video_id"),
        (entry.source_sentence_id, "Passed manifest source_sentence_id"),
        (entry.source_sentence_name, "Passed manifest source_sentence_name"),
    ):
        _validate_non_empty_text(issues, value, label=label)

    num_frames = _validate_frame_count_value(
        issues,
        entry.num_frames,
        label="Passed manifest num_frames",
        allow_zero=False,
    )
    _validate_optional_fps_value(issues, entry.fps, label="Passed manifest fps")
    _validate_selected_person_metadata(
        issues,
        entry.selected_person,
        num_frames=num_frames,
        label="Passed manifest selected_person",
    )
    _validate_frame_quality_summary(
        issues,
        entry.frame_quality,
        num_frames=num_frames,
        label="Passed manifest frame_quality",
    )


def _validate_dropped_entry(
    issues: list[SampleValidationIssue],
    entry: DroppedManifestEntry,
) -> None:
    _validate_schema_version_value(issues, entry.schema_version, label="Dropped manifest entry")
    if entry.status != SampleStatus.DROPPED:
        _add_issue(
            issues,
            "invalid_dropped_entry_status",
            f"DroppedManifestEntry status must be 'dropped', got {entry.status!r}.",
        )

    _validate_non_empty_text(issues, entry.sample_id, label="Dropped manifest sample_id")
    _validate_non_empty_text(issues, entry.drop_stage, label="Dropped manifest drop_stage")
    _validate_drop_reasons(issues, entry.drop_reasons)
    _validate_drop_details(issues, entry.drop_details)
    _validate_dropped_materialization(issues, entry.materialization)

    if entry.text is not None:
        _validate_non_empty_text(issues, entry.text, label="Dropped manifest text")

    num_frames = None
    if entry.num_frames is not None:
        num_frames = _validate_frame_count_value(
            issues,
            entry.num_frames,
            label="Dropped manifest num_frames",
            allow_zero=True,
        )
    elif entry.materialization.payload_exists:
        _add_issue(
            issues,
            "materialized_dropped_sample_missing_num_frames",
            "Dropped entries with materialized payloads must include num_frames.",
        )

    _validate_optional_fps_value(issues, entry.fps, label="Dropped manifest fps")

    if entry.selected_person is not None:
        _validate_selected_person_metadata(
            issues,
            entry.selected_person,
            num_frames=num_frames,
            label="Dropped manifest selected_person",
        )

    if entry.frame_quality is not None:
        if num_frames is None:
            _add_issue(
                issues,
                "dropped_frame_quality_missing_num_frames",
                "Dropped entries with frame_quality must include num_frames.",
            )
        _validate_frame_quality_summary(
            issues,
            entry.frame_quality,
            num_frames=num_frames,
            label="Dropped manifest frame_quality",
        )


def _validate_dropped_materialization(
    issues: list[SampleValidationIssue],
    materialization: DroppedMaterializationLifecycle,
) -> None:
    outcome = materialization.debug_materialization_outcome
    if not materialization.debug_materialization_eligible:
        if materialization.debug_materialization_attempted:
            _add_issue(
                issues,
                "ineligible_dropped_materialization_attempted",
                "Ineligible dropped entries must not attempt debug materialization.",
            )
        if outcome is not DroppedDebugMaterializationOutcome.NOT_ATTEMPTED:
            _add_issue(
                issues,
                "ineligible_dropped_materialization_has_attempt_outcome",
                "Ineligible dropped entries must have materialization outcome 'not_attempted'.",
            )

    if materialization.debug_materialization_attempted:
        if outcome is DroppedDebugMaterializationOutcome.NOT_ATTEMPTED:
            _add_issue(
                issues,
                "attempted_dropped_materialization_missing_outcome",
                "Attempted dropped debug materialization must have succeeded or failed.",
            )
    elif outcome is not DroppedDebugMaterializationOutcome.NOT_ATTEMPTED:
        _add_issue(
            issues,
            "unattempted_dropped_materialization_has_attempt_outcome",
            "Unattempted dropped debug materialization must have outcome 'not_attempted'.",
        )

    if materialization.payload_path is not None:
        _validate_non_empty_text(
            issues,
            materialization.payload_path,
            label="Dropped materialization payload_path",
        )
    elif materialization.payload_exists:
        _add_issue(
            issues,
            "dropped_payload_exists_without_path",
            "Dropped materialization payload_exists=true requires payload_path.",
        )

    if materialization.archive_publishable and not materialization.payload_exists:
        _add_issue(
            issues,
            "dropped_archive_publishable_without_payload",
            "Dropped archive publishability requires an existing dropped payload artifact.",
        )

    if outcome is DroppedDebugMaterializationOutcome.SUCCEEDED:
        if not materialization.debug_materialization_eligible:
            _add_issue(
                issues,
                "succeeded_dropped_materialization_not_eligible",
                "Succeeded dropped debug materialization requires eligibility.",
            )
        if not materialization.debug_materialization_attempted:
            _add_issue(
                issues,
                "succeeded_dropped_materialization_not_attempted",
                "Succeeded dropped debug materialization requires an attempt.",
            )
        if not materialization.payload_exists or materialization.payload_path is None:
            _add_issue(
                issues,
                "succeeded_dropped_materialization_missing_payload",
                "Succeeded dropped debug materialization requires an existing payload path.",
            )
        if not materialization.archive_publishable:
            _add_issue(
                issues,
                "succeeded_dropped_materialization_not_archive_publishable",
                "Succeeded dropped debug materialization must be archive-publishable.",
            )
        if materialization.failure_reason is not None:
            _add_issue(
                issues,
                "succeeded_dropped_materialization_has_failure",
                "Succeeded dropped debug materialization must not include a failure reason.",
            )

    if outcome is DroppedDebugMaterializationOutcome.FAILED:
        if not materialization.debug_materialization_eligible:
            _add_issue(
                issues,
                "failed_dropped_materialization_not_eligible",
                "Failed dropped debug materialization requires eligibility.",
            )
        if not materialization.debug_materialization_attempted:
            _add_issue(
                issues,
                "failed_dropped_materialization_not_attempted",
                "Failed dropped debug materialization requires an attempt.",
            )
        if materialization.payload_exists or materialization.payload_path is not None:
            _add_issue(
                issues,
                "failed_dropped_materialization_has_payload",
                "Failed dropped debug materialization must not expose a payload path.",
            )
        if materialization.archive_publishable:
            _add_issue(
                issues,
                "failed_dropped_materialization_archive_publishable",
                "Failed dropped debug materialization must not be archive-publishable.",
            )
        if materialization.failure_reason is None:
            _add_issue(
                issues,
                "failed_dropped_materialization_missing_failure",
                "Failed dropped debug materialization requires a failure reason.",
            )
        else:
            _validate_non_empty_text(
                issues,
                materialization.failure_reason,
                label="Dropped materialization failure_reason",
            )

    if outcome is DroppedDebugMaterializationOutcome.NOT_ATTEMPTED:
        if materialization.payload_exists or materialization.payload_path is not None:
            _add_issue(
                issues,
                "unattempted_dropped_materialization_has_payload",
                "Unattempted dropped debug materialization must not expose a payload path.",
            )
        if materialization.archive_publishable:
            _add_issue(
                issues,
                "unattempted_dropped_materialization_archive_publishable",
                "Unattempted dropped debug materialization must not be archive-publishable.",
            )
