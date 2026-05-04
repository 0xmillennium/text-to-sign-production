"""Hand channel structural gate."""

from __future__ import annotations

from text_to_sign_production.data.gates.config import GatesConfig
from text_to_sign_production.data.gates.types import GateResult, GateStatus
from text_to_sign_production.data.pose.types import PoseBuildOutput


def evaluate_hand_gate(pose_output: PoseBuildOutput, config: GatesConfig) -> GateResult:
    """Evaluate if the hand channels carry minimally usable canonical evidence."""
    reasons = []

    channel_nonzero_frames = pose_output.frame_quality.channel_nonzero_frames
    for channel in ("left_hand", "right_hand"):
        if channel not in channel_nonzero_frames:
            reasons.append(f"missing_channel_nonzero_frames:{channel}")

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    nonzero_left = channel_nonzero_frames["left_hand"]
    min_req_left = config.channel_configs["left_hand"].min_nonzero_frames
    if nonzero_left < min_req_left:
        reasons.append(f"insufficient_left_hand_evidence:{nonzero_left}<{min_req_left}")

    nonzero_right = channel_nonzero_frames["right_hand"]
    min_req_right = config.channel_configs["right_hand"].min_nonzero_frames
    if nonzero_right < min_req_right:
        reasons.append(f"insufficient_right_hand_evidence:{nonzero_right}<{min_req_right}")

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    return GateResult(status=GateStatus.PASSED)
