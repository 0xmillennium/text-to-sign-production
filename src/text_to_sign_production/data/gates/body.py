"""Body channel structural gate."""

from __future__ import annotations

from text_to_sign_production.data.gates.config import GatesConfig
from text_to_sign_production.data.gates.types import GateResult, GateStatus
from text_to_sign_production.data.pose.types import PoseBuildOutput


def evaluate_body_gate(pose_output: PoseBuildOutput, config: GatesConfig) -> GateResult:
    """Evaluate if the body channel carries minimally usable canonical evidence."""
    reasons: list[str] = []

    nonzero = pose_output.frame_quality.channel_nonzero_frames.get("body", 0)
    min_req = config.channel_configs["body"].min_nonzero_frames

    if nonzero < min_req:
        reasons.append(f"insufficient_body_evidence:{nonzero}<{min_req}")

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=reasons)

    return GateResult(status=GateStatus.PASSED)
