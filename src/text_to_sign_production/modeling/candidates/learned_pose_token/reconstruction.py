"""Helpers for representing tokenizer reconstructions as generated-pose artifacts."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    bfh_pose_arrays_from_vectorized,
    unflatten_bfh_pose_values,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.data.bfh_schema import BfhPoseArrays
from text_to_sign_production.modeling.inference.generated_pose import (
    GeneratedPoseSourceIdentity,
    generated_pose_sample_from_bfh_arrays,
)
from text_to_sign_production.modeling.artifacts import GeneratedPoseSample


def reconstruct_bfh_pose_arrays_from_vectorized_frames(
    *,
    reconstructed_values: np.ndarray,
    reference_vectorized: BfhVectorizedPose,
) -> BfhPoseArrays:
    """Restore reconstructed flat frame vectors to BFH arrays using reference masks."""

    if not isinstance(reference_vectorized, BfhVectorizedPose):
        raise LearnedPoseTokenError("reference_vectorized must be a BfhVectorizedPose.")
    flat = np.asarray(reconstructed_values, dtype=np.float32)
    expected = (
        reference_vectorized.frame_count,
        reference_vectorized.layout.total_feature_dim,
    )
    if flat.shape != expected:
        raise LearnedPoseTokenError(f"reconstructed_values must have shape {expected}; got {flat.shape}.")
    if not np.all(np.isfinite(flat)):
        raise LearnedPoseTokenError("reconstructed_values must contain only finite values.")
    values = unflatten_bfh_pose_values(
        flat,
        layout=reference_vectorized.layout,
        frame_count=reference_vectorized.frame_count,
    )
    vectorized = BfhVectorizedPose(
        layout=reference_vectorized.layout,
        values=values,
        validity_mask=reference_vectorized.validity_mask,
        frame_validity_mask=reference_vectorized.frame_validity_mask,
        confidence_values=reference_vectorized.confidence_values,
        frame_count=reference_vectorized.frame_count,
        source_sample_id=reference_vectorized.source_sample_id,
    )
    return bfh_pose_arrays_from_vectorized(vectorized)


def reconstructed_sample_to_generated_pose(
    *,
    identity: GeneratedPoseSourceIdentity,
    split: SampleSplit,
    generation_index: int,
    arrays: BfhPoseArrays,
    sample_id: str | None = None,
    producer_key: str = "learned_pose_token",
    producer_stage: str = "evaluate_reconstruction",
) -> GeneratedPoseSample:
    """Convert a reconstruction sample to the generated-pose contract."""

    if producer_stage != "evaluate_reconstruction":
        raise LearnedPoseTokenError("producer_stage must be 'evaluate_reconstruction'.")
    if not isinstance(identity, GeneratedPoseSourceIdentity):
        raise LearnedPoseTokenError("identity must be a GeneratedPoseSourceIdentity.")
    return generated_pose_sample_from_bfh_arrays(
        sample_id=sample_id or identity.source_sentence_name,
        sentence_name=identity.source_sentence_name,
        split=split,
        generation_index=generation_index,
        arrays=arrays,
        producer_key=producer_key,
        producer_stage=producer_stage,
        identity=GeneratedPoseSourceIdentity(
            canonical_id=identity.canonical_id,
            phase_number=identity.phase_number,
            research_role=identity.research_role,
            run_name=identity.run_name,
            text=identity.text,
            source_video_id=identity.source_video_id,
            source_sentence_id=identity.source_sentence_id,
            source_sentence_name=identity.source_sentence_name,
            reference_payload_ref=identity.reference_payload_ref,
            num_candidates_for_sample=identity.num_candidates_for_sample,
            producer_type=identity.producer_type,
            generation_mode="reconstruction",
            length_policy=identity.length_policy,
            confidence_policy=identity.confidence_policy,
            seed=identity.seed,
            failure_reason=identity.failure_reason,
        ),
    )


__all__ = [
    "reconstruct_bfh_pose_arrays_from_vectorized_frames",
    "reconstructed_sample_to_generated_pose",
]
