"""Modeling data package.

Canonical dataset truth lives in ``text_to_sign_production.data.dataset``.
The M0 processed-world adapter remains available under
``text_to_sign_production.modeling.data.legacy``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ModelingDataError",
    "FULL_BFH_CHANNEL_POLICY",
    "FULL_BFH_CHANNELS",
    "BFH_CHANNEL_SPECS",
    "BfhChannelSpec",
    "BfhPoseArrays",
    "pose_arrays_from_prepared_sample",
    "channel_coordinates",
    "channel_confidence",
    "ModelingManifest",
    "ModelingManifestFamily",
    "ModelingManifestSample",
    "parse_modeling_manifest_family",
    "resolve_modeling_manifest_path",
    "read_modeling_manifest",
    "resolve_prepared_payload_path",
    "load_manifest_sample",
    "load_manifest_samples",
    "ReferencePairingKey",
    "GeneratedPairingKey",
    "SourceIdentity",
    "reference_pairing_key",
    "generated_pairing_key",
    "source_identity_from_reference",
    "source_identity_from_generated",
    "build_reference_index",
    "build_generated_index",
    "valid_frame_mask",
    "channel_observation_mask",
    "channel_frame_presence_mask",
    "GeneratedPoseSurfaceSample",
    "resolve_generated_payload_path",
    "load_generated_pose_surface",
    "TEMPORAL_WINDOW_SPEC_SCHEMA_VERSION",
    "BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION",
    "TEMPORAL_GRANULARITY_FRAME",
    "TEMPORAL_GRANULARITY_WINDOW",
    "TEMPORAL_COVERAGE_POLICY_EXACT",
    "TEMPORAL_COVERAGE_POLICY_COVER_ALL",
    "TEMPORAL_PADDING_POLICY_NONE",
    "TEMPORAL_PADDING_POLICY_RIGHT_ZERO",
    "WINDOW_MERGE_POLICY_MEAN_VALID_OBSERVATIONS",
    "TemporalWindowSpec",
    "BfhPoseWindowSequence",
    "BfhPoseWindowMergeResult",
    "temporal_window_starts",
    "extract_bfh_pose_windows",
    "flatten_bfh_pose_windows",
    "flatten_bfh_pose_window_validity",
    "merge_bfh_pose_windows",
]

_EXPORT_MODULES = {
    "ModelingDataError": "text_to_sign_production.modeling.data.errors",
    "FULL_BFH_CHANNEL_POLICY": "text_to_sign_production.modeling.data.bfh_schema",
    "FULL_BFH_CHANNELS": "text_to_sign_production.modeling.data.bfh_schema",
    "BFH_CHANNEL_SPECS": "text_to_sign_production.modeling.data.bfh_schema",
    "BfhChannelSpec": "text_to_sign_production.modeling.data.bfh_schema",
    "BfhPoseArrays": "text_to_sign_production.modeling.data.bfh_schema",
    "pose_arrays_from_prepared_sample": "text_to_sign_production.modeling.data.bfh_schema",
    "channel_coordinates": "text_to_sign_production.modeling.data.bfh_schema",
    "channel_confidence": "text_to_sign_production.modeling.data.bfh_schema",
    "ModelingManifest": "text_to_sign_production.modeling.data.manifest_families",
    "ModelingManifestFamily": "text_to_sign_production.modeling.data.manifest_families",
    "parse_modeling_manifest_family": "text_to_sign_production.modeling.data.manifest_families",
    "resolve_modeling_manifest_path": "text_to_sign_production.modeling.data.manifest_families",
    "read_modeling_manifest": "text_to_sign_production.modeling.data.manifest_families",
    "ModelingManifestSample": "text_to_sign_production.modeling.data.prepared_sample_loader",
    "resolve_prepared_payload_path": "text_to_sign_production.modeling.data.prepared_sample_loader",
    "load_manifest_sample": "text_to_sign_production.modeling.data.prepared_sample_loader",
    "load_manifest_samples": "text_to_sign_production.modeling.data.prepared_sample_loader",
    "ReferencePairingKey": "text_to_sign_production.modeling.data.pairing_keys",
    "GeneratedPairingKey": "text_to_sign_production.modeling.data.pairing_keys",
    "SourceIdentity": "text_to_sign_production.modeling.data.pairing_keys",
    "reference_pairing_key": "text_to_sign_production.modeling.data.pairing_keys",
    "generated_pairing_key": "text_to_sign_production.modeling.data.pairing_keys",
    "source_identity_from_reference": "text_to_sign_production.modeling.data.pairing_keys",
    "source_identity_from_generated": "text_to_sign_production.modeling.data.pairing_keys",
    "build_reference_index": "text_to_sign_production.modeling.data.pairing_keys",
    "build_generated_index": "text_to_sign_production.modeling.data.pairing_keys",
    "valid_frame_mask": "text_to_sign_production.modeling.data.masks",
    "channel_observation_mask": "text_to_sign_production.modeling.data.masks",
    "channel_frame_presence_mask": "text_to_sign_production.modeling.data.masks",
    "GeneratedPoseSurfaceSample": "text_to_sign_production.modeling.data.generated_pose_loader",
    "resolve_generated_payload_path": "text_to_sign_production.modeling.data.generated_pose_loader",
    "load_generated_pose_surface": "text_to_sign_production.modeling.data.generated_pose_loader",
    "TEMPORAL_WINDOW_SPEC_SCHEMA_VERSION": "text_to_sign_production.modeling.data.temporal_windows",
    "BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION": "text_to_sign_production.modeling.data.temporal_windows",
    "TEMPORAL_GRANULARITY_FRAME": "text_to_sign_production.modeling.data.temporal_windows",
    "TEMPORAL_GRANULARITY_WINDOW": "text_to_sign_production.modeling.data.temporal_windows",
    "TEMPORAL_COVERAGE_POLICY_EXACT": "text_to_sign_production.modeling.data.temporal_windows",
    "TEMPORAL_COVERAGE_POLICY_COVER_ALL": "text_to_sign_production.modeling.data.temporal_windows",
    "TEMPORAL_PADDING_POLICY_NONE": "text_to_sign_production.modeling.data.temporal_windows",
    "TEMPORAL_PADDING_POLICY_RIGHT_ZERO": "text_to_sign_production.modeling.data.temporal_windows",
    "WINDOW_MERGE_POLICY_MEAN_VALID_OBSERVATIONS": "text_to_sign_production.modeling.data.temporal_windows",
    "TemporalWindowSpec": "text_to_sign_production.modeling.data.temporal_windows",
    "BfhPoseWindowSequence": "text_to_sign_production.modeling.data.temporal_windows",
    "BfhPoseWindowMergeResult": "text_to_sign_production.modeling.data.temporal_windows",
    "temporal_window_starts": "text_to_sign_production.modeling.data.temporal_windows",
    "extract_bfh_pose_windows": "text_to_sign_production.modeling.data.temporal_windows",
    "flatten_bfh_pose_windows": "text_to_sign_production.modeling.data.temporal_windows",
    "flatten_bfh_pose_window_validity": "text_to_sign_production.modeling.data.temporal_windows",
    "merge_bfh_pose_windows": "text_to_sign_production.modeling.data.temporal_windows",
}


def __getattr__(name: str) -> Any:
    """Lazily expose modeling data contracts without forcing artifact imports."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
