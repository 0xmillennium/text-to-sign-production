"""Processed-v1 full-BFH modeling input contracts for the M0 baseline."""

from .collate import collate_processed_pose_samples
from .dataset import (
    ProcessedModelingDataError,
    ProcessedPoseDataset,
    load_processed_pose_sample,
    read_processed_modeling_manifest,
)
from .schemas import (
    M0_CHANNEL_POLICY,
    M0_CONFIDENCE_CHANNELS,
    M0_TARGET_CHANNEL_SHAPES,
    M0_TARGET_CHANNELS,
    ConfidenceArray,
    IntegerArray,
    MaskArray,
    PoseArray,
    ProcessedModelingManifestRecord,
    ProcessedPoseBatch,
    ProcessedPoseItem,
    ProcessedPoseSample,
)

__all__ = [
    "ProcessedModelingDataError",
    "ProcessedModelingManifestRecord",
    "ProcessedPoseBatch",
    "ProcessedPoseDataset",
    "ProcessedPoseItem",
    "ProcessedPoseSample",
    "M0_CHANNEL_POLICY",
    "M0_CONFIDENCE_CHANNELS",
    "M0_TARGET_CHANNELS",
    "M0_TARGET_CHANNEL_SHAPES",
    "ConfidenceArray",
    "IntegerArray",
    "MaskArray",
    "PoseArray",
    "collate_processed_pose_samples",
    "load_processed_pose_sample",
    "read_processed_modeling_manifest",
]
