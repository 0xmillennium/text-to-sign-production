"""Shared provider-stage artifact role vocabulary."""

from __future__ import annotations

from enum import StrEnum


class ProviderStageArtifactRole(StrEnum):
    """Roles declared by provider stage results, distinct from provenance kinds."""

    CHECKPOINT_BEST = "checkpoint_best"
    CHECKPOINT_LAST = "checkpoint_last"
    GENERATED_POSE_MANIFEST = "generated_pose_manifest"
    GENERATED_POSE_SAMPLE_ROOT = "generated_pose_sample_root"
    TRAINING_METRICS = "training_metrics"
    VALIDATION_METRICS = "validation_metrics"
    REPRESENTATION_ARTIFACT = "representation_artifact"
    LATENT_CACHE = "latent_cache"
    CHANNEL_POLICY = "channel_policy"


__all__ = ["ProviderStageArtifactRole"]
