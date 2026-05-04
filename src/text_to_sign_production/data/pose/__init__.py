"""Pose-level functionality for raw frame ingestion and tensor building."""

from text_to_sign_production.data.pose.files import discover_frame_files
from text_to_sign_production.data.pose.parser import parse_frame
from text_to_sign_production.data.pose.people import (
    build_person_metadata,
    resolve_target_person_index,
)
from text_to_sign_production.data.pose.tensors import build_pose_tensors
from text_to_sign_production.data.pose.types import (
    FrameFileListing,
    ParsedFrameResult,
    PoseBuildDiagnostics,
    PoseBuildInput,
    PoseBuildOutput,
)
from text_to_sign_production.data.pose.validate import validate_pose_build

__all__ = [
    "FrameFileListing",
    "ParsedFrameResult",
    "PoseBuildDiagnostics",
    "PoseBuildInput",
    "PoseBuildOutput",
    "build_person_metadata",
    "build_pose_tensors",
    "discover_frame_files",
    "parse_frame",
    "resolve_target_person_index",
    "validate_pose_build",
]
