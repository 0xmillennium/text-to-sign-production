"""Learned pose-token representation foundation.

This package intentionally does not register a model provider. It contains the
pose -> token/code -> reconstructed pose foundations that a later provider stage
can build on.
"""

from __future__ import annotations

from text_to_sign_production.modeling.candidates.learned_pose_token.codebook import (
    CODEBOOK_METADATA_SCHEMA_VERSION,
    CODEBOOK_STABILITY_SCHEMA_VERSION,
    PoseCodebookMetadata,
    PoseCodebookStabilityReport,
    compute_codebook_stability,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    LearnedPoseCodebookConfig,
    LearnedPoseCheckpointConfig,
    LearnedPoseGenerationConfig,
    LearnedPoseTokenConfig,
    LearnedPoseTokenDataConfig,
    LearnedPoseTokenExportConfig,
    LearnedPoseTokenIdentityConfig,
    LearnedPoseTokenReportConfig,
    LearnedPoseTokenRepresentationConfig,
    LearnedPoseTextToTokenConfig,
    LearnedPoseTokenizerConfig,
    LearnedPoseTokenizerTrainingConfig,
    learned_pose_token_config_from_effective_dict,
    load_learned_pose_token_config,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.dataset import (
    PoseTokenSourceSample,
    PoseTokenTrainingSample,
    build_pose_token_training_samples,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.io import (
    read_pose_token_manifest_jsonl,
    read_pose_token_sequence_npz,
    write_pose_token_manifest_jsonl,
    write_pose_token_sequence_npz,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.spec import (
    LEARNED_POSE_TOKEN_CANONICAL_ID,
    LEARNED_POSE_TOKEN_MODEL_KEY,
    LEARNED_POSE_TOKEN_PHASE_NUMBER,
    LEARNED_POSE_TOKEN_RESEARCH_ROLE,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.tokens import (
    POSE_TOKEN_MANIFEST_SCHEMA_VERSION,
    POSE_TOKEN_SCHEMA_VERSION,
    PoseTokenManifestEntry,
    PoseTokenSequence,
)

__all__ = [
    "CODEBOOK_METADATA_SCHEMA_VERSION",
    "CODEBOOK_STABILITY_SCHEMA_VERSION",
    "LEARNED_POSE_TOKEN_CANONICAL_ID",
    "LEARNED_POSE_TOKEN_MODEL_KEY",
    "LEARNED_POSE_TOKEN_PHASE_NUMBER",
    "LEARNED_POSE_TOKEN_RESEARCH_ROLE",
    "POSE_TOKEN_MANIFEST_SCHEMA_VERSION",
    "POSE_TOKEN_SCHEMA_VERSION",
    "LearnedPoseCodebookConfig",
    "LearnedPoseCheckpointConfig",
    "LearnedPoseGenerationConfig",
    "LearnedPoseTextToTokenConfig",
    "LearnedPoseTokenConfig",
    "LearnedPoseTokenDataConfig",
    "LearnedPoseTokenError",
    "LearnedPoseTokenExportConfig",
    "LearnedPoseTokenIdentityConfig",
    "LearnedPoseTokenReportConfig",
    "LearnedPoseTokenRepresentationConfig",
    "LearnedPoseTokenizerConfig",
    "LearnedPoseTokenizerTrainingConfig",
    "PoseCodebookMetadata",
    "PoseCodebookStabilityReport",
    "PoseTokenManifestEntry",
    "PoseTokenSequence",
    "PoseTokenSourceSample",
    "PoseTokenTrainingSample",
    "build_pose_token_training_samples",
    "compute_codebook_stability",
    "load_learned_pose_token_config",
    "learned_pose_token_config_from_effective_dict",
    "read_pose_token_manifest_jsonl",
    "read_pose_token_sequence_npz",
    "write_pose_token_manifest_jsonl",
    "write_pose_token_sequence_npz",
]
