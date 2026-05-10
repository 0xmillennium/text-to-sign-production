"""Frame availability structural gate."""

from __future__ import annotations

from text_to_sign_production.legacy_data.gates.types import GateResult, GatesConfig, GateStatus
from text_to_sign_production.legacy_data.pose.types import FrameFileListing
from text_to_sign_production.legacy_data.sources.types import SourceCandidate


def evaluate_frames_gate(
    listing: FrameFileListing, candidate: SourceCandidate, config: GatesConfig
) -> GateResult:
    """Evaluate if sufficient frames were discovered and counts match source candidate."""
    reasons = []
    integrity = config.integrity

    if listing.missing:
        reasons.append("frame_directory_missing")
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    if not listing.files:
        reasons.append("no_frame_files_found")
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    discovered_count = len(listing.files)
    expected_count = candidate.frame_count

    if discovered_count < integrity.min_num_frames:
        reasons.append(f"insufficient_frames:{discovered_count}<{integrity.min_num_frames}")

    if discovered_count != expected_count:
        reasons.append(f"frame_count_mismatch:expected_{expected_count}_got_{discovered_count}")

    duration_seconds = candidate.end_time - candidate.start_time
    if duration_seconds < integrity.min_duration_seconds:
        reasons.append(
            f"insufficient_duration_seconds:{duration_seconds:.3f}<{integrity.min_duration_seconds}"
        )

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    return GateResult(status=GateStatus.PASSED)
