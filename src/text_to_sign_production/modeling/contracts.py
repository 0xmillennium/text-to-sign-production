"""Source contracts for the M0 direct text-to-full-BFH baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from text_to_sign_production.data.samples.schema import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.modeling.data.schemas import (
    M0_CHANNEL_POLICY,
    M0_TARGET_CHANNEL_SHAPES,
    M0_TARGET_CHANNELS,
)

BASELINE_ID: Final[str] = "m0-direct-text-to-full-bfh"
BASELINE_NAME: Final[str] = "M0 Direct Text-to-Full-BFH Baseline"
BASELINE_ROLE: Final[str] = "m0_comparison_floor"

LENGTH_POLICY: Final[str] = "reference_length"
CONFIDENCE_POLICY: Final[str] = "synthetic_validity_not_model_uncertainty"
PREDICTION_SCHEMA_VERSION: Final[str] = "t2sp-baseline-prediction-v1"
PREDICTION_MANIFEST_SCHEMA_VERSION: Final[str] = "t2sp-baseline-prediction-manifest-v1"

REFERENCE_CONFIDENCE_MASK_POLICY: Final[str] = (
    "strictly_positive_reference_confidence_marks_target_supervision"
)
PEOPLE_PER_FRAME_POLICY: Final[str] = "synthetic_single_person_validity"
SELECTED_PERSON_INDEX_POLICY: Final[str] = "copy_reference_selected_person_index"

DEFAULT_CHANNEL_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "body": 1.0,
        "left_hand": 1.5,
        "right_hand": 1.5,
        "face": 1.25,
    }
)

EXCLUDED_MECHANISMS: Final[tuple[str, ...]] = (
    "learned_pose_bottleneck",
    "vector_quantized_pose_bottleneck",
    "stochastic_denoising_generator",
    "manual_component_factor_groups_as_contribution",
    "auxiliary_consistency_objective",
    "nearest_neighbor_or_segment_assembly",
    "public_annotation_supervision_or_annotation_to_pose",
    "formal_notation_to_pose",
    "three_dimensional_character_model",
    "parametric_body_model",
    "rendering",
)

RISK_CONTROL_STATEMENT: Final[str] = (
    "M0 readiness does not imply contribution strength. "
    "M0 readiness does not imply strong task-solving quality. "
    "Automatic metrics are not sufficient evidence of sign intelligibility."
)


@dataclass(frozen=True, slots=True)
class FailureModeDefinition:
    """Expected M0 baseline failure-mode definition for later report generation."""

    identifier: str
    label: str
    expected_baseline_risk: str
    expected_qualitative_observation: bool
    solved_by_m0: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable failure-mode record."""

        return {
            "identifier": self.identifier,
            "label": self.label,
            "expected_baseline_risk": self.expected_baseline_risk,
            "expected_qualitative_observation": self.expected_qualitative_observation,
            "solved_by_m0": self.solved_by_m0,
        }


M0_FAILURE_MODES: Final[tuple[FailureModeDefinition, ...]] = (
    FailureModeDefinition(
        identifier="oversmoothing",
        label="Oversmoothing",
        expected_baseline_risk=(
            "Direct regression may average plausible motions into low-amplitude trajectories."
        ),
        expected_qualitative_observation=True,
    ),
    FailureModeDefinition(
        identifier="temporal_jitter",
        label="Temporal jitter",
        expected_baseline_risk=(
            "Frame-independent decoding pressure can produce unstable local motion."
        ),
        expected_qualitative_observation=True,
    ),
    FailureModeDefinition(
        identifier="timing_misalignment",
        label="Timing misalignment",
        expected_baseline_risk=(
            "Reference-length decoding does not learn robust sign timing or duration."
        ),
        expected_qualitative_observation=True,
    ),
    FailureModeDefinition(
        identifier="hand_articulation_loss",
        label="Hand articulation loss",
        expected_baseline_risk=(
            "Small hand poses may be weakly recovered by a simple text-conditioned regressor."
        ),
        expected_qualitative_observation=True,
    ),
    FailureModeDefinition(
        identifier="non_manual_loss",
        label="Non-manual loss",
        expected_baseline_risk=(
            "Face and other non-manual cues may be poorly modeled even though face is predicted."
        ),
        expected_qualitative_observation=True,
    ),
    FailureModeDefinition(
        identifier="cross_channel_inconsistency",
        label="Cross-channel inconsistency",
        expected_baseline_risk=(
            "Separate output heads may disagree across body, hand, and face channels."
        ),
        expected_qualitative_observation=True,
    ),
    FailureModeDefinition(
        identifier="semantic_mismatch",
        label="Semantic mismatch",
        expected_baseline_risk=(
            "A direct text regressor may produce motion that is not intelligibly aligned to text."
        ),
        expected_qualitative_observation=True,
    ),
    FailureModeDefinition(
        identifier="mode_averaging",
        label="Mode averaging",
        expected_baseline_risk=(
            "Multiple plausible sign realizations can collapse into an average trajectory."
        ),
        expected_qualitative_observation=True,
    ),
    FailureModeDefinition(
        identifier="sequence_drift",
        label="Sequence drift",
        expected_baseline_risk=(
            "Longer sequences may accumulate trajectory errors without structural correction."
        ),
        expected_qualitative_observation=True,
    ),
)


@dataclass(frozen=True, slots=True)
class BaselineArchitectureSpec:
    """Serializable M0 architecture contract for provenance and reports."""

    baseline_id: str
    baseline_name: str
    role: str
    architecture_summary: str
    excluded_mechanisms: tuple[str, ...]
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]
    length_policy: str
    channel_policy: str
    confidence_policy: str
    loss_policy: Mapping[str, object]
    risk_control_statement: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable architecture specification."""

        return {
            "baseline_id": self.baseline_id,
            "baseline_name": self.baseline_name,
            "role": self.role,
            "architecture_summary": self.architecture_summary,
            "excluded_mechanisms": list(self.excluded_mechanisms),
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "length_policy": self.length_policy,
            "channel_policy": self.channel_policy,
            "confidence_policy": self.confidence_policy,
            "loss_policy": dict(self.loss_policy),
            "risk_control_statement": self.risk_control_statement,
        }


def baseline_architecture_spec(
    *,
    channel_weights: Mapping[str, float] | None = None,
) -> BaselineArchitectureSpec:
    """Build the source-level M0 architecture contract."""

    resolved_weights = validate_channel_weights(
        DEFAULT_CHANNEL_WEIGHTS if channel_weights is None else channel_weights
    )
    return BaselineArchitectureSpec(
        baseline_id=BASELINE_ID,
        baseline_name=BASELINE_NAME,
        role=BASELINE_ROLE,
        architecture_summary=(
            "Config-driven pretrained text encoder, pooled text representation, normalized "
            "frame position plus sinusoidal frame features, residual MLP temporal decoder, "
            "and linear full-BFH channel heads."
        ),
        excluded_mechanisms=EXCLUDED_MECHANISMS,
        input_schema={
            "text": "transcript string from processed-v1 manifest",
            "target_length_source": LENGTH_POLICY,
            "source_processed_schema_version": PROCESSED_SCHEMA_VERSION,
        },
        output_schema={
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "channel_policy": M0_CHANNEL_POLICY,
            "channels": _channel_shape_records(M0_TARGET_CHANNELS),
        },
        length_policy=LENGTH_POLICY,
        channel_policy=M0_CHANNEL_POLICY,
        confidence_policy=CONFIDENCE_POLICY,
        loss_policy={
            "name": "channel_balanced_masked_mse",
            "channel_weights": resolved_weights,
            "frame_mask": "frame_valid_mask_and_not_padding",
            "reference_confidence_mask_policy": REFERENCE_CONFIDENCE_MASK_POLICY,
        },
        risk_control_statement=RISK_CONTROL_STATEMENT,
    )


def failure_mode_inventory_records() -> list[dict[str, object]]:
    """Return the M0 failure-mode inventory as JSON-serializable records."""

    return [failure_mode.to_dict() for failure_mode in M0_FAILURE_MODES]


def _channel_shape_records(channels: Sequence[str]) -> dict[str, dict[str, object]]:
    return {
        channel: {
            "coordinates": ["T", M0_TARGET_CHANNEL_SHAPES[channel][0], 2],
            "confidence": ["T", M0_TARGET_CHANNEL_SHAPES[channel][0]],
        }
        for channel in channels
    }


def validate_channel_weights(channel_weights: Mapping[str, float]) -> dict[str, float]:
    missing_channels = sorted(set(M0_TARGET_CHANNELS).difference(channel_weights))
    extra_channels = sorted(set(channel_weights).difference(M0_TARGET_CHANNELS))
    if missing_channels or extra_channels:
        raise ValueError(
            "channel_weights must match the M0 full-BFH channels exactly: "
            f"missing={missing_channels}, extra={extra_channels}."
        )

    resolved: dict[str, float] = {}
    for channel in M0_TARGET_CHANNELS:
        weight = float(channel_weights[channel])
        if weight < 0.0:
            raise ValueError(f"channel weight for {channel!r} must not be negative.")
        resolved[channel] = weight
    if not any(weight > 0.0 for weight in resolved.values()):
        raise ValueError("At least one M0 channel weight must be positive.")
    return resolved


__all__ = [
    "BASELINE_ID",
    "BASELINE_NAME",
    "BASELINE_ROLE",
    "CONFIDENCE_POLICY",
    "DEFAULT_CHANNEL_WEIGHTS",
    "EXCLUDED_MECHANISMS",
    "FailureModeDefinition",
    "LENGTH_POLICY",
    "M0_FAILURE_MODES",
    "PEOPLE_PER_FRAME_POLICY",
    "PREDICTION_MANIFEST_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_VERSION",
    "REFERENCE_CONFIDENCE_MASK_POLICY",
    "RISK_CONTROL_STATEMENT",
    "SELECTED_PERSON_INDEX_POLICY",
    "baseline_architecture_spec",
    "BaselineArchitectureSpec",
    "failure_mode_inventory_records",
    "validate_channel_weights",
]
