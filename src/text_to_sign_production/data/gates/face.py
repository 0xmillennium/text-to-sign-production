"""Face channel structural gate."""

from __future__ import annotations

from text_to_sign_production.data.gates.config import GatesConfig
from text_to_sign_production.data.gates.types import GateResult, GateStatus
from text_to_sign_production.data.pose.types import PoseBuildOutput


def evaluate_face_gate(pose_output: PoseBuildOutput, config: GatesConfig) -> GateResult:
    """Evaluate if the face channel carries minimally usable canonical evidence.

    This implements canonical BFH semantics for face: totally unusable/absent
    evidence causes structural failure, but weak quality is deferred.
    """
    reasons = []

    if "face" not in pose_output.frame_quality.channel_nonzero_frames:
        reasons.append("missing_channel_nonzero_frames:face")
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    nonzero = pose_output.frame_quality.channel_nonzero_frames["face"]
    min_req = config.channel_configs["face"].min_nonzero_frames

    if nonzero < min_req:
        reasons.append(f"insufficient_face_evidence:{nonzero}<{min_req}")

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    return GateResult(status=GateStatus.PASSED)
