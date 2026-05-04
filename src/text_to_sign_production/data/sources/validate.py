"""Source-level validation helpers."""

from __future__ import annotations

from text_to_sign_production.data.sources.types import SourceCandidate


def validate_candidate(candidate: SourceCandidate) -> list[str]:
    """Validate a candidate's source-level properties.

    Checks basic properties like non-empty text and correct splits.
    Returns a list of issue codes.
    """
    issues: list[str] = []

    if not candidate.sample_id:
        issues.append("empty_sample_id")

    if not candidate.text or not candidate.text.strip():
        issues.append("empty_text")

    if (
        candidate.start_time < 0
        or candidate.end_time < 0
        or candidate.start_time >= candidate.end_time
    ):
        issues.append("invalid_timestamps")

    if candidate.split not in {"train", "val", "test"}:
        issues.append(f"invalid_split:{candidate.split}")

    if candidate.frame_count <= 0:
        issues.append("invalid_frame_count")

    if not candidate.video_metadata.is_readable:
        issues.append(f"unreadable_video:{candidate.video_metadata.error}")

    if not str(candidate.keypoints_dir).strip() or not candidate.keypoints_dir.name:
        issues.append("invalid_keypoint_directory")

    if not str(candidate.video_path).strip() or not candidate.video_path.name:
        issues.append("invalid_video_path")

    return issues
