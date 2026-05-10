"""Samples admission-gate evaluation entrypoint."""

from __future__ import annotations

from text_to_sign_production.core.ids import SampleSplit, SampleStatus
from text_to_sign_production.core.models import (
    GateDecision,
    GateDecisionBundle,
    GateName,
    GateStatus,
    PreparedSample,
)
from text_to_sign_production.data.gate.policies.body import evaluate_body_gate
from text_to_sign_production.data.gate.policies.config import GatesConfig
from text_to_sign_production.data.gate.policies.face import evaluate_face_gate
from text_to_sign_production.data.gate.policies.frames import evaluate_frames_gate
from text_to_sign_production.data.gate.policies.hand import evaluate_hand_gate
from text_to_sign_production.data.gate.policies.source import evaluate_source_gate

GATE_EVALUATION_ORDER: tuple[GateName, ...] = (
    GateName.SOURCE,
    GateName.FRAMES,
    GateName.BODY,
    GateName.HAND,
    GateName.FACE,
)


def evaluate_sample_gates(
    sample: PreparedSample,
    config: GatesConfig,
) -> GateDecisionBundle:
    """Evaluate whether a PreparedSample is admissible into the checkpoint."""
    decisions = (
        evaluate_source_gate(sample, config.source),
        evaluate_frames_gate(sample, config.frames),
        evaluate_body_gate(sample, config.body),
        evaluate_hand_gate(sample, config.hand),
        evaluate_face_gate(sample, config.face),
    )
    return assemble_gate_decision_bundle(
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        decisions=decisions,
    )


def evaluate_quality_gates(
    sample: PreparedSample,
    config: GatesConfig,
) -> GateDecisionBundle:
    """Compatibility alias for the samples admission gate entrypoint."""
    return evaluate_sample_gates(sample, config)


def assemble_gate_decision_bundle(
    *,
    sample_id: str,
    split: SampleSplit,
    decisions: tuple[GateDecision, ...],
) -> GateDecisionBundle:
    """Assemble ordered per-gate decisions into a final root decision bundle."""
    failed_gates = tuple(
        decision.gate for decision in decisions if decision.status is GateStatus.FAIL
    )
    return GateDecisionBundle(
        sample_id=sample_id,
        split=split,
        final_status=SampleStatus.DROPPED if failed_gates else SampleStatus.PASSED,
        terminal_gate=failed_gates[0] if failed_gates else None,
        decisions=decisions,
        failed_gates=failed_gates,
    )


__all__ = [
    "GATE_EVALUATION_ORDER",
    "assemble_gate_decision_bundle",
    "evaluate_quality_gates",
    "evaluate_sample_gates",
]
