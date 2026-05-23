"""Gloss-free latent diffusion package.

This package intentionally does not auto-register its provider. Use the explicit
registration module or candidate bootstrap for provider registration.
"""

from __future__ import annotations

from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    DiffusionScheduleConfig,
    LatentCheckpointConfig,
    LatentDenoiserConfig,
    LatentDiffusionConfig,
    LatentDiffusionDataConfig,
    LatentDiffusionIdentityConfig,
    LatentDiffusionReportConfig,
    LatentGenerationConfig,
    LatentLengthConfig,
    LatentTargetConfig,
    LatentTrainingConfig,
    latent_diffusion_config_from_mapping,
    load_latent_diffusion_config,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.dataset import (
    LatentFrameTrainingSample,
    LatentSourceSample,
    build_latent_source_samples,
    build_latent_target_spec,
    cache_latent_sequences,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.io import (
    read_latent_manifest_jsonl,
    read_latent_sequence_npz,
    read_latent_target_spec_json,
    write_latent_manifest_jsonl,
    write_latent_sequence_npz,
    write_latent_target_spec_json,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.latents import (
    LATENT_MANIFEST_SCHEMA_VERSION,
    LATENT_SEQUENCE_SCHEMA_VERSION,
    LATENT_TARGET_SPEC_SCHEMA_VERSION,
    LatentManifestEntry,
    LatentSequence,
    LatentTargetSpec,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.schedule import (
    DIFFUSION_SCHEDULE_SCHEMA_VERSION,
    DiffusionSchedule,
    build_linear_beta_schedule,
    q_sample,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.spec import (
    LATENT_DIFFUSION_CANONICAL_ID,
    LATENT_DIFFUSION_MODEL_KEY,
    LATENT_DIFFUSION_PHASE_NUMBER,
    LATENT_DIFFUSION_RESEARCH_ROLE,
)

__all__ = [
    "DIFFUSION_SCHEDULE_SCHEMA_VERSION",
    "LATENT_DIFFUSION_CANONICAL_ID",
    "LATENT_DIFFUSION_MODEL_KEY",
    "LATENT_DIFFUSION_PHASE_NUMBER",
    "LATENT_DIFFUSION_RESEARCH_ROLE",
    "LATENT_MANIFEST_SCHEMA_VERSION",
    "LATENT_SEQUENCE_SCHEMA_VERSION",
    "LATENT_TARGET_SPEC_SCHEMA_VERSION",
    "DiffusionSchedule",
    "DiffusionScheduleConfig",
    "LatentCheckpointConfig",
    "LatentDenoiserConfig",
    "LatentDiffusionConfig",
    "LatentDiffusionDataConfig",
    "LatentDiffusionError",
    "LatentDiffusionIdentityConfig",
    "LatentDiffusionReportConfig",
    "LatentFrameTrainingSample",
    "LatentGenerationConfig",
    "LatentLengthConfig",
    "LatentManifestEntry",
    "LatentSequence",
    "LatentSourceSample",
    "LatentTargetConfig",
    "LatentTargetSpec",
    "LatentTrainingConfig",
    "build_latent_source_samples",
    "build_latent_target_spec",
    "build_linear_beta_schedule",
    "cache_latent_sequences",
    "latent_diffusion_config_from_mapping",
    "load_latent_diffusion_config",
    "q_sample",
    "read_latent_manifest_jsonl",
    "read_latent_sequence_npz",
    "read_latent_target_spec_json",
    "write_latent_manifest_jsonl",
    "write_latent_sequence_npz",
    "write_latent_target_spec_json",
]
