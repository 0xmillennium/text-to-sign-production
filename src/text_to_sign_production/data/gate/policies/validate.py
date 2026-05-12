"""Gate admission-gate validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import GateDecision, GateDecisionBundle, GateStatus
from text_to_sign_production.data.gate.policies.types import (
    GateValidationIssue,
    GateValidationIssueCode,
)

if TYPE_CHECKING:
    from text_to_sign_production.data.gate.policies.config import GatesConfig


def validate_gates_config(config: GatesConfig) -> tuple[GateValidationIssue, ...]:
    """Validate typed gate admission-gate threshold consistency."""
    issues: list[GateValidationIssue] = []
    for name, value in (
        ("source.min_character_count", config.source.min_character_count),
        ("source.min_token_count", config.source.min_token_count),
    ):
        if value < 0:
            issues.append(
                _issue(
                    GateValidationIssueCode.INVALID_CONFIG_THRESHOLD,
                    f"{name} must be non-negative.",
                )
            )
    if config.frames.min_frame_count < 1:
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_CONFIG_THRESHOLD,
                "frames.min_frame_count must be positive.",
            )
        )
    if config.frames.min_valid_frame_count < 1:
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_CONFIG_THRESHOLD,
                "frames.min_valid_frame_count must be positive.",
            )
        )
    if config.frames.min_valid_frame_count > config.frames.min_frame_count:
        issues.append(
            _issue(
                GateValidationIssueCode.CONTRADICTORY_THRESHOLDS,
                "frames.min_valid_frame_count cannot exceed frames.min_frame_count.",
            )
        )
    if config.frames.min_duration_seconds <= 0:
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_CONFIG_THRESHOLD,
                "frames.min_duration_seconds must be positive.",
            )
        )
    if (
        config.frames.max_duration_seconds is not None
        and config.frames.max_duration_seconds < config.frames.min_duration_seconds
    ):
        issues.append(
            _issue(
                GateValidationIssueCode.CONTRADICTORY_THRESHOLDS,
                "frames.max_duration_seconds cannot be below frames.min_duration_seconds.",
            )
        )
    for section_name in ("body", "hand", "face"):
        section = getattr(config, section_name)
        for field_name in section.__dataclass_fields__:
            value = getattr(section, field_name)
            if value is None:
                continue
            if isinstance(value, int) and not isinstance(value, bool) and value < 0:
                issues.append(
                    _issue(
                        GateValidationIssueCode.INVALID_CONFIG_THRESHOLD,
                        f"{section_name}.{field_name} must be non-negative.",
                    )
                )
            if isinstance(value, float) and not 0 <= value <= 1:
                issues.append(
                    _issue(
                        GateValidationIssueCode.INVALID_CONFIG_THRESHOLD,
                        f"{section_name}.{field_name} must be within [0, 1].",
                    )
                )
    return tuple(issues)


def validate_gate_decision(decision: GateDecision) -> tuple[GateValidationIssue, ...]:
    """Validate a single root gate decision."""
    issues: list[GateValidationIssue] = []
    if decision.status is GateStatus.FAIL and not decision.issue_codes:
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_GATE_DECISION,
                "Failed gate decisions must carry issue codes.",
            )
        )
    if decision.status is GateStatus.PASS and decision.issue_codes:
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_GATE_DECISION,
                "Passed gate decisions must not carry issue codes.",
            )
        )
    if len(set(decision.issue_codes)) != len(decision.issue_codes):
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_GATE_DECISION,
                "Gate issue codes cannot contain duplicates.",
            )
        )
    return tuple(issues)


def validate_gate_result(decision: GateDecision) -> tuple[GateValidationIssue, ...]:
    """Compatibility alias for validating one gate decision."""
    return validate_gate_decision(decision)


def validate_gate_decision_bundle(
    bundle: GateDecisionBundle,
) -> tuple[GateValidationIssue, ...]:
    """Validate final gate admission decision structure."""
    issues: list[GateValidationIssue] = []
    for decision in bundle.decisions:
        issues.extend(validate_gate_decision(decision))
    expected_failed = tuple(
        decision.gate for decision in bundle.decisions if decision.status is GateStatus.FAIL
    )
    if bundle.failed_gates != expected_failed:
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_DECISION_BUNDLE,
                "failed_gates must match failed decisions.",
            )
        )
    expected_terminal = expected_failed[0] if expected_failed else None
    if bundle.terminal_gate != expected_terminal:
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_DECISION_BUNDLE,
                "terminal_gate must be the first failed gate.",
            )
        )
    if bundle.final_status is SampleStatus.PASSED and expected_failed:
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_DECISION_BUNDLE,
                "Passed bundles cannot have failed gates.",
            )
        )
    if bundle.final_status is SampleStatus.DROPPED and not expected_failed:
        issues.append(
            _issue(
                GateValidationIssueCode.INVALID_DECISION_BUNDLE,
                "Dropped bundles require failed gates.",
            )
        )
    return tuple(issues)


def _issue(code: GateValidationIssueCode, message: str) -> GateValidationIssue:
    return GateValidationIssue(code=code, message=message)


__all__ = [
    "validate_gate_decision",
    "validate_gate_decision_bundle",
    "validate_gate_result",
    "validate_gates_config",
]
