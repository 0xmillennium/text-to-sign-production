"""Shared modeling backbones and lazy M0 backbone exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import TextBackbone, TextBackboneOutput
from .bfh_masks import (
    BfhChannelPartition,
    apply_channel_mask,
    build_bfh_channel_partitions,
    channel_feature_slice,
    channel_joint_mask,
)
from .bfh_standardization import (
    BFH_STANDARDIZATION_SCHEMA_VERSION,
    BfhStandardizationStats,
    apply_bfh_standardization,
    compute_bfh_standardization_stats,
    invert_bfh_standardization,
    read_bfh_standardization_stats_json,
    write_bfh_standardization_stats_json,
)
from .bfh_vectorization import (
    BFH_VECTORIZATION_SCHEMA_VERSION,
    BfhTensorLayout,
    BfhVectorizedPose,
    bfh_pose_arrays_from_vectorized,
    default_bfh_tensor_layout,
    flatten_bfh_vectorized_pose,
    unflatten_bfh_pose_values,
    vectorize_bfh_pose_arrays,
)
from .text_encoder import (
    TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH,
    TEXT_ENCODER_BACKEND_HUGGINGFACE,
    TEXT_ENCODER_SCHEMA_VERSION,
    TextEncoder,
    TextEncodingCache,
    TextEncodingRequest,
    TextEncoderConfig,
    TextEncoderOutput,
    build_text_encoder,
    encode_text_requests,
    text_encoder_metadata,
)

if TYPE_CHECKING:
    from .flan_t5 import DEFAULT_FLAN_T5_MODEL_NAME, FlanT5TextBackbone

__all__ = [
    "DEFAULT_FLAN_T5_MODEL_NAME",
    "BFH_STANDARDIZATION_SCHEMA_VERSION",
    "BFH_VECTORIZATION_SCHEMA_VERSION",
    "TEXT_ENCODER_SCHEMA_VERSION",
    "TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH",
    "TEXT_ENCODER_BACKEND_HUGGINGFACE",
    "BfhChannelPartition",
    "BfhStandardizationStats",
    "BfhTensorLayout",
    "BfhVectorizedPose",
    "FlanT5TextBackbone",
    "TextBackbone",
    "TextBackboneOutput",
    "TextEncodingCache",
    "TextEncodingRequest",
    "TextEncoder",
    "TextEncoderConfig",
    "TextEncoderOutput",
    "apply_bfh_standardization",
    "apply_channel_mask",
    "bfh_pose_arrays_from_vectorized",
    "build_bfh_channel_partitions",
    "build_text_encoder",
    "encode_text_requests",
    "channel_feature_slice",
    "channel_joint_mask",
    "compute_bfh_standardization_stats",
    "default_bfh_tensor_layout",
    "flatten_bfh_vectorized_pose",
    "invert_bfh_standardization",
    "read_bfh_standardization_stats_json",
    "text_encoder_metadata",
    "unflatten_bfh_pose_values",
    "vectorize_bfh_pose_arrays",
    "write_bfh_standardization_stats_json",
]


def __getattr__(name: str) -> Any:
    if name == "DEFAULT_FLAN_T5_MODEL_NAME":
        from .flan_t5 import DEFAULT_FLAN_T5_MODEL_NAME

        return DEFAULT_FLAN_T5_MODEL_NAME
    if name == "FlanT5TextBackbone":
        from .flan_t5 import FlanT5TextBackbone

        return FlanT5TextBackbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
