"""Hand channel structural gate."""

from __future__ import annotations

from text_to_sign_production.data.gates.config import GatesConfig
from text_to_sign_production.data.gates.types import GateResult, GateStatus
from text_to_sign_production.data.pose.types import PoseBuildOutput


def evaluate_hand_gate(pose_output: PoseBuildOutput, config: GatesConfig) -> GateResult:
    """Evaluate if either hand channel carries minimal structural evidence."""
    reasons = []

    channel_nonzero_frames = pose_output.frame_quality.channel_nonzero_frames
    for channel in ("left_hand", "right_hand"):
        if channel not in channel_nonzero_frames:
            reasons.append(f"missing_channel_nonzero_frames:{channel}")

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    nonzero_left = channel_nonzero_frames["left_hand"]
    nonzero_right = channel_nonzero_frames["right_hand"]
    any_hand_nonzero = max(nonzero_left, nonzero_right)
    min_req = config.min_any_hand_nonzero_frames
    if any_hand_nonzero < min_req:
        reasons.append(f"insufficient_any_hand_evidence:{any_hand_nonzero}<{min_req}")

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    return GateResult(status=GateStatus.PASSED)
