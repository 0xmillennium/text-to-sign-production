"""Source-level validation helpers."""

from __future__ import annotations

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.sources.types import SourceCandidate, SourceValidationIssue


def validate_candidate(candidate: SourceCandidate) -> list[SourceValidationIssue]:
    """Validate a candidate's source-level properties.

    Checks basic properties like non-empty text and correct splits.
    Returns typed validation issues for persistent source invariants.
    """
    issues: list[SourceValidationIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(SourceValidationIssue(code=code, message=message))

    if not candidate.sample_id:
        add("empty_sample_id", "Source candidate sample_id must be non-empty.")

    if not candidate.text or not candidate.text.strip():
        add("empty_text", "Source candidate text must be non-empty.")

    if (
        candidate.start_time < 0
        or candidate.end_time < 0
        or candidate.start_time >= candidate.end_time
    ):
        add("invalid_timestamps", "Source candidate timestamps must be ordered and non-negative.")

    if not isinstance(candidate.split, SampleSplit):
        add("invalid_split", f"Source candidate split is invalid: {candidate.split!r}.")

    if candidate.frame_count <= 0:
        add("invalid_frame_count", "Source candidate frame_count must be positive.")

    if not candidate.video_metadata.is_readable:
        add(
            "unreadable_video",
            f"Source candidate video metadata is unreadable: {candidate.video_metadata.error}.",
        )

    if not str(candidate.keypoints_dir).strip() or not candidate.keypoints_dir.name:
        add("invalid_keypoint_directory", "Source candidate keypoints_dir must be named.")

    if not str(candidate.video_path).strip() or not candidate.video_path.name:
        add("invalid_video_path", "Source candidate video_path must be named.")

    for issue_code in candidate.source_issues:
        add(f"source_issue:{issue_code}", f"Source-side issue present: {issue_code}.")

    return issues
