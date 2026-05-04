"""Pose-level functionality for raw frame ingestion and tensor building."""

from text_to_sign_production.data.pose.files import discover_frame_files
from text_to_sign_production.data.pose.parser import parse_frame
from text_to_sign_production.data.pose.people import (
    DEFAULT_PERSON_SELECTION_POLICY,
    build_person_metadata,
    person_selection_policy_name,
    resolve_person_selection,
    resolve_target_person_index,
)
from text_to_sign_production.data.pose.tensors import build_pose_tensors
from text_to_sign_production.data.pose.types import (
    FrameFileListing,
    ParsedFrameResult,
    PersonSelectionCandidateScore,
    PersonSelectionPolicy,
    PersonSelectionResult,
    PoseBuildDiagnostics,
    PoseBuildInput,
    PoseBuildOutput,
    PoseValidationIssue,
)
from text_to_sign_production.data.pose.validate import validate_pose_build

__all__ = [
    "DEFAULT_PERSON_SELECTION_POLICY",
    "FrameFileListing",
    "ParsedFrameResult",
    "PersonSelectionCandidateScore",
    "PersonSelectionPolicy",
    "PersonSelectionResult",
    "PoseBuildDiagnostics",
    "PoseBuildInput",
    "PoseBuildOutput",
    "PoseValidationIssue",
    "build_person_metadata",
    "build_pose_tensors",
    "discover_frame_files",
    "person_selection_policy_name",
    "parse_frame",
    "resolve_person_selection",
    "resolve_target_person_index",
    "validate_pose_build",
]
