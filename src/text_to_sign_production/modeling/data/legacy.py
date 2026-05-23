"""Explicit legacy M0 processed-world adapter namespace."""

from __future__ import annotations

from text_to_sign_production.modeling.data.collate import collate_processed_pose_samples
from text_to_sign_production.modeling.data.dataset import (
    ProcessedModelingDataError,
    ProcessedPoseDataset,
    load_processed_pose_sample,
    processed_modeling_record_from_entry,
    read_processed_modeling_manifest,
)
from text_to_sign_production.modeling.data.schemas import (
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
    "processed_modeling_record_from_entry",
    "read_processed_modeling_manifest",
]
