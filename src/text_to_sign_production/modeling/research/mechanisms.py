"""Mechanism tags used to constrain research-backed modeling artifacts."""

from __future__ import annotations

import enum


class MechanismTag(enum.StrEnum):
    """Controlled mechanism vocabulary for models, objectives, and comparators."""

    DIRECT_TEXT_TO_POSE = "direct_text_to_pose"
    LEARNED_POSE_TOKEN = "learned_pose_token"
    COMPACT_MOTION_CODE = "compact_motion_code"
    LATENT_DIFFUSION = "latent_diffusion"
    LATENT_GENERATION = "latent_generation"
    ARTICULATOR_AWARE_STRUCTURE = "articulator_aware_structure"
    SEMANTIC_CONSISTENCY_OBJECTIVE = "semantic_consistency_objective"
    RETRIEVAL_REUSE = "retrieval_reuse"
    POSE_KEYPOINT_EVALUATION = "pose_keypoint_evaluation"
    EMBEDDING_CHECKS = "embedding_checks"
    QUALITATIVE_INSPECTION = "qualitative_inspection"
    GLOSS_SUPERVISION = "gloss_supervision"
    HAMNOSYS_OR_NOTATION = "hamnosys_or_notation"
    AVATAR_OR_3D = "avatar_or_3d"
    RENDERING = "rendering"
    DICTIONARY_STITCHING = "dictionary_stitching"
    AUDIO_CONDITIONING = "audio_conditioning"
    CFM_OR_SPARSE_KEYFRAME = "cfm_or_sparse_keyframe"
    NON_MANUAL_LINGUISTIC_ANNOTATION = "non_manual_linguistic_annotation"
    MANUAL_GLOSS_SEGMENTATION = "manual_gloss_segmentation"
    PRIMARY_LEARNED_GENERATOR = "primary_learned_generator"


__all__ = [
    "MechanismTag",
]
