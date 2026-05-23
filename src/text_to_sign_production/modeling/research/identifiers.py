"""Stable identifiers for research-backed modeling artifacts."""

from __future__ import annotations

import enum


class ModelKey(enum.StrEnum):
    """Canonical identifiers for model candidates."""

    BASE_DIRECT = "base_direct"
    LEARNED_POSE_TOKEN = "learned_pose_token"
    LATENT_DIFFUSION = "latent_diffusion"
    ARTICULATOR_AWARE = "articulator_aware"


class ObjectiveKey(enum.StrEnum):
    """Canonical identifiers for auxiliary objectives."""

    SEMANTIC_CONSISTENCY = "semantic_consistency"


class ComparatorKey(enum.StrEnum):
    """Canonical identifiers for comparator artifacts."""

    RETRIEVAL_POSE = "retrieval_pose"


class EvaluationProtocolKey(enum.StrEnum):
    """Canonical identifiers for evaluation protocol artifacts."""

    PHASE5_CORE = "phase5_core"
    PHASE9_SEMANTIC_ABLATION = "phase9_semantic_ablation"
    PHASE9_RETRIEVAL_COMPARATOR = "phase9_retrieval_comparator"


__all__ = [
    "ComparatorKey",
    "EvaluationProtocolKey",
    "ModelKey",
    "ObjectiveKey",
]
