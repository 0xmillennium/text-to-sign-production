"""Schema validation for PreparedSample-derived quality facts."""

from __future__ import annotations

from text_to_sign_production.data.tier.facts.types import (
    CANONICAL_QUALITY_CHANNELS,
    FactsValidationIssue,
    FactsValidationIssueCode,
    QualityFacts,
)


def validate_quality_facts_schema(facts: QualityFacts) -> tuple[FactsValidationIssue, ...]:
    """Validate quality facts as a PreparedSample-derived product."""
    issues: list[FactsValidationIssue] = []
    if not facts.source.sample_id:
        _add(
            issues,
            FactsValidationIssueCode.EMPTY_SAMPLE_ID,
            "sample_id is empty.",
            "source.sample_id",
        )
    identity_checks: tuple[tuple[str, str], ...] = (
        (facts.source.source_video_id, "source.source_video_id"),
        (facts.source.source_sentence_id, "source.source_sentence_id"),
        (facts.source.source_sentence_name, "source.source_sentence_name"),
    )
    for identity_value, path in identity_checks:
        if not identity_value:
            _add(issues, FactsValidationIssueCode.EMPTY_SOURCE_IDENTITY, f"{path} is empty.", path)
    if facts.source.source_fps <= 0:
        _add(
            issues,
            FactsValidationIssueCode.INVALID_SOURCE_FPS,
            "source_fps must be positive.",
            "source.source_fps",
        )
    if facts.frame.frame_count <= 0:
        _add(
            issues,
            FactsValidationIssueCode.NEGATIVE_FRAME_COUNT,
            "frame_count must be positive.",
            "frame.frame_count",
        )
    if facts.frame.valid_frame_count < 0 or facts.frame.invalid_frame_count < 0:
        _add(
            issues,
            FactsValidationIssueCode.NEGATIVE_VALID_FRAME_COUNT,
            "frame counts cannot be negative.",
            "frame",
        )
    ratio_checks: tuple[tuple[float, str], ...] = (
        (facts.frame.valid_frame_ratio, "frame.valid_frame_ratio"),
        (facts.frame.invalid_frame_ratio, "frame.invalid_frame_ratio"),
        (facts.tracking.missing_frame_ratio, "tracking.missing_frame_ratio"),
        (facts.tracking.continuity_break_ratio, "tracking.continuity_break_ratio"),
        (facts.tracking.reanchor_ratio, "tracking.reanchor_ratio"),
    )
    for ratio_value, path in ratio_checks:
        if not 0.0 <= ratio_value <= 1.0:
            _add(
                issues,
                FactsValidationIssueCode.INVALID_VALID_FRAME_RATIO,
                "ratio out of range.",
                path,
            )
    if tuple(fact.channel for fact in facts.channel) != CANONICAL_QUALITY_CHANNELS:
        _add(
            issues,
            FactsValidationIssueCode.CHANNEL_FACT_COUNT_MISMATCH,
            "channel facts must follow canonical order.",
            "channel",
        )
    for index, channel in enumerate(facts.channel):
        if channel.coordinate_space != "normalized_image":
            _add(
                issues,
                FactsValidationIssueCode.INVALID_COORDINATE_SPACE,
                "coordinate space must be normalized_image.",
                f"channel.{index}.coordinate_space",
            )
        channel_ratio_checks: tuple[tuple[float, str], ...] = (
            (channel.nonzero_frame_ratio, f"channel.{index}.nonzero_frame_ratio"),
            (channel.zero_frame_ratio, f"channel.{index}.zero_frame_ratio"),
        )
        for ratio_value, path in channel_ratio_checks:
            if not 0.0 <= ratio_value <= 1.0:
                _add(
                    issues,
                    FactsValidationIssueCode.INVALID_CHANNEL_RATIO,
                    "channel ratio out of range.",
                    path,
                )
    if facts.text_length.duration_seconds <= 0:
        _add(
            issues,
            FactsValidationIssueCode.INVALID_DURATION,
            "duration_seconds must be positive.",
            "text_length.duration_seconds",
        )
    return tuple(issues)


def _add(
    issues: list[FactsValidationIssue],
    code: FactsValidationIssueCode,
    message: str,
    field_path: str | None = None,
) -> None:
    issues.append(FactsValidationIssue(code=code, message=message, field_path=field_path))


__all__ = ["validate_quality_facts_schema"]
