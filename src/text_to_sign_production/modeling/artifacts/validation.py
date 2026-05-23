"""Validation helpers for generated-pose artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data.bfh_schema import FULL_BFH_CHANNEL_POLICY


class GeneratedPoseArtifactError(ValueError):
    """Raised when generated-pose payload or manifest contracts are violated."""


@dataclass(frozen=True, slots=True)
class GeneratedPoseValidationIssue:
    """Structured generated-pose validation issue."""

    code: str
    message: str
    field: str | None = None
    sample_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise GeneratedPoseArtifactError("code must be non-empty.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise GeneratedPoseArtifactError("message must be non-empty.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable validation issue."""

        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "sample_id": self.sample_id,
        }


def validate_generated_pose_sample(sample: Any) -> tuple[GeneratedPoseValidationIssue, ...]:
    """Return validation issues for a generated-pose sample."""

    issues: list[GeneratedPoseValidationIssue] = []
    _require_text(issues, getattr(sample, "sample_id", None), "sample_id")
    _require_text(issues, getattr(sample, "producer_key", None), "producer_key")
    _require_text(issues, getattr(sample, "run_name", None), "run_name")
    _require_text(issues, getattr(sample, "text", None), "text", sample_id=getattr(sample, "sample_id", None))
    _positive_int(issues, getattr(sample, "phase_number", None), "phase_number")
    generation_index = getattr(sample, "generation_index", None)
    candidate_count = getattr(sample, "num_candidates_for_sample", None)
    _non_negative_int(issues, generation_index, "generation_index")
    _positive_int(issues, candidate_count, "num_candidates_for_sample")
    if isinstance(generation_index, int) and isinstance(candidate_count, int):
        if generation_index >= candidate_count:
            issues.append(
                _issue(
                    "generation_index_out_of_range",
                    "generation_index must be less than num_candidates_for_sample.",
                    "generation_index",
                    getattr(sample, "sample_id", None),
                )
            )
    pose = getattr(sample, "pose", None)
    if pose is None:
        issues.append(_issue("missing_pose", "pose is required.", "pose"))
    return tuple(issues)


def validate_generated_pose_manifest_entry(
    entry: Any,
) -> tuple[GeneratedPoseValidationIssue, ...]:
    """Return validation issues for one generated-pose manifest entry."""

    issues: list[GeneratedPoseValidationIssue] = []
    sample_id = getattr(entry, "sample_id", None)
    for field_name in (
        "producer_key",
        "canonical_id",
        "research_role",
        "run_name",
        "sample_id",
        "source_video_id",
        "source_sentence_id",
        "source_sentence_name",
        "text",
        "reference_payload_ref",
    ):
        _require_text(issues, getattr(entry, field_name, None), field_name, sample_id=sample_id)
    if getattr(entry, "channel_policy", None) != FULL_BFH_CHANNEL_POLICY:
        issues.append(
            _issue(
                "invalid_channel_policy",
                "channel_policy must be full_bfh.",
                "channel_policy",
                sample_id,
            )
        )
    _positive_int(issues, getattr(entry, "phase_number", None), "phase_number", sample_id=sample_id)
    generation_index = getattr(entry, "generation_index", None)
    candidate_count = getattr(entry, "num_candidates_for_sample", None)
    _non_negative_int(issues, generation_index, "generation_index", sample_id=sample_id)
    _positive_int(issues, candidate_count, "num_candidates_for_sample", sample_id=sample_id)
    if isinstance(generation_index, int) and isinstance(candidate_count, int):
        if generation_index >= candidate_count:
            issues.append(
                _issue(
                    "generation_index_out_of_range",
                    "generation_index must be less than num_candidates_for_sample.",
                    "generation_index",
                    sample_id,
                )
            )
    failure_reason = getattr(entry, "failure_reason", None)
    generated_payload_ref = getattr(entry, "generated_payload_ref", None)
    if failure_reason is None:
        _require_text(issues, generated_payload_ref, "generated_payload_ref", sample_id=sample_id)
        _positive_int(issues, getattr(entry, "frame_count", None), "frame_count", sample_id=sample_id)
        valid_frame_count = getattr(entry, "valid_frame_count", None)
        _non_negative_int(issues, valid_frame_count, "valid_frame_count", sample_id=sample_id)
        frame_count = getattr(entry, "frame_count", None)
        if isinstance(frame_count, int) and isinstance(valid_frame_count, int):
            if valid_frame_count > frame_count:
                issues.append(
                    _issue(
                        "valid_frame_count_exceeds_frame_count",
                        "valid_frame_count cannot exceed frame_count.",
                        "valid_frame_count",
                        sample_id,
                    )
                )
    return tuple(issues)


def validate_generated_pose_manifest_entries(
    entries: Iterable[Any],
    *,
    expected_split: SampleSplit | str | None = None,
) -> tuple[GeneratedPoseValidationIssue, ...]:
    """Return validation issues for a generated-pose manifest entry collection."""

    expected = None if expected_split is None else SampleSplit(expected_split)
    issues: list[GeneratedPoseValidationIssue] = []
    seen: set[tuple[str, str, int]] = set()
    for entry in entries:
        issues.extend(validate_generated_pose_manifest_entry(entry))
        split = getattr(entry, "split", None)
        sample_id = getattr(entry, "sample_id", None)
        generation_index = getattr(entry, "generation_index", None)
        if expected is not None and split != expected:
            issues.append(
                _issue(
                    "unexpected_split",
                    f"entry split must be {expected.value}.",
                    "split",
                    sample_id,
                )
            )
        if isinstance(split, SampleSplit) and isinstance(sample_id, str) and isinstance(generation_index, int):
            key = (split.value, sample_id, generation_index)
            if key in seen:
                issues.append(
                    _issue(
                        "duplicate_manifest_key",
                        "duplicate (split, sample_id, generation_index).",
                        "generation_index",
                        sample_id,
                    )
                )
            seen.add(key)
    return tuple(issues)


def _require_text(
    issues: list[GeneratedPoseValidationIssue],
    value: object,
    field_name: str,
    sample_id: str | None = None,
) -> None:
    if not isinstance(value, str) or not value.strip():
        issues.append(_issue("missing_text", f"{field_name} must be non-empty.", field_name, sample_id))


def _positive_int(
    issues: list[GeneratedPoseValidationIssue],
    value: object,
    field_name: str,
    sample_id: str | None = None,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        issues.append(_issue("invalid_integer", f"{field_name} must be positive.", field_name, sample_id))


def _non_negative_int(
    issues: list[GeneratedPoseValidationIssue],
    value: object,
    field_name: str,
    sample_id: str | None = None,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        issues.append(
            _issue("invalid_integer", f"{field_name} must be non-negative.", field_name, sample_id)
        )


def _issue(
    code: str,
    message: str,
    field: str | None = None,
    sample_id: str | None = None,
) -> GeneratedPoseValidationIssue:
    return GeneratedPoseValidationIssue(
        code=code,
        message=message,
        field=field,
        sample_id=sample_id,
    )


__all__ = [
    "GeneratedPoseArtifactError",
    "GeneratedPoseValidationIssue",
    "validate_generated_pose_manifest_entries",
    "validate_generated_pose_manifest_entry",
    "validate_generated_pose_sample",
]
