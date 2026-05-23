"""Token decoding and generated-pose export helpers for learned_pose_token."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseSplitWriteResult,
    GeneratedPoseStreamWriteResult,
    diagnostic_generated_pose_paths,
    generated_manifest_entry_from_sample,
    generated_manifest_entry_to_record,
    validate_diagnostic_generated_pose_paths,
    validate_generated_pose_manifest_entries,
    validate_generated_pose_sample,
    write_generated_pose_payload,
    write_generated_pose_split,
    write_generated_pose_split_to_explicit_root,
)
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    BfhStandardizationStats,
    invert_bfh_standardization,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    bfh_pose_arrays_from_vectorized,
    default_bfh_tensor_layout,
    unflatten_bfh_pose_values,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.dataset import (
    PoseTokenSourceSample,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.model import (
    LearnedPoseTokenizer,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.tokens import (
    PoseTokenSequence,
    generated_frame_count_from_token_count,
    pose_token_temporal_spec,
)
from text_to_sign_production.modeling.data import (
    BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION,
    BfhPoseWindowSequence,
    merge_bfh_pose_windows,
    temporal_window_starts,
)
from text_to_sign_production.modeling.data.bfh_schema import BfhPoseArrays
from text_to_sign_production.modeling.inference import (
    GeneratedPoseSourceIdentity,
    generated_pose_sample_from_bfh_arrays,
)
from text_to_sign_production.modeling.registry import require_model_spec
from text_to_sign_production.modeling.research import ModelKey


@dataclass(frozen=True, slots=True)
class DecodedPoseSample:
    source: PoseTokenSourceSample
    token_sequence: PoseTokenSequence
    pose: BfhPoseArrays


def decode_token_sequence_to_bfh_pose(
    *,
    tokenizer: LearnedPoseTokenizer,
    token_sequence: PoseTokenSequence,
    standardization_stats: BfhStandardizationStats,
    source: PoseTokenSourceSample,
    confidence_policy: str,
) -> BfhPoseArrays:
    """Decode token ids to full-BFH pose without copying reference pose values."""

    if not isinstance(tokenizer, LearnedPoseTokenizer):
        raise LearnedPoseTokenError("tokenizer must be a LearnedPoseTokenizer.")
    if not isinstance(token_sequence, PoseTokenSequence):
        raise LearnedPoseTokenError("token_sequence must be a PoseTokenSequence.")
    if not isinstance(standardization_stats, BfhStandardizationStats):
        raise LearnedPoseTokenError("standardization_stats must be BfhStandardizationStats.")
    if not isinstance(source, PoseTokenSourceSample):
        raise LearnedPoseTokenError("source must be a PoseTokenSourceSample.")
    if confidence_policy != "synthetic_validity":
        raise LearnedPoseTokenError("confidence_policy must be 'synthetic_validity'.")
    if token_sequence.sample_id != source.sample_id:
        raise LearnedPoseTokenError("token_sequence sample_id must match source sample_id.")
    ids = np.asarray(token_sequence.token_ids, dtype=np.int64).copy()
    if ids.shape != (token_sequence.token_count,):
        raise LearnedPoseTokenError("token_sequence token_ids shape is invalid.")
    device = next(tokenizer.parameters()).device
    tokenizer.eval()
    with torch.no_grad():
        token_tensor = torch.as_tensor(ids, dtype=torch.long, device=device)
        quantized = tokenizer.quantizer.embedding(token_tensor)
        decoded = tokenizer.decoder(quantized).detach().cpu().numpy().astype(np.float32)
    frame_feature_dim = standardization_stats.layout.total_feature_dim
    expected_feature_dim = token_sequence.window_size * frame_feature_dim
    if token_sequence.temporal_granularity == "frame":
        expected_feature_dim = frame_feature_dim
    if decoded.shape != (token_sequence.token_count, expected_feature_dim):
        unit_name = "window values" if token_sequence.temporal_granularity == "window" else "frames"
        raise LearnedPoseTokenError(
            f"decoded token {unit_name} must have shape "
            f"({token_sequence.token_count}, {expected_feature_dim}); got {decoded.shape}."
        )
    if not np.all(np.isfinite(decoded)):
        raise LearnedPoseTokenError("decoded token frames contain non-finite values.")
    layout = default_bfh_tensor_layout()
    if layout != standardization_stats.layout:
        raise LearnedPoseTokenError("decoded BFH layout does not match standardization stats.")
    if token_sequence.temporal_granularity == "window":
        return _decode_window_token_sequence(
            decoded=decoded,
            token_sequence=token_sequence,
            standardization_stats=standardization_stats,
            layout=layout,
            source=source,
        )
    values = unflatten_bfh_pose_values(
        decoded,
        layout=layout,
        frame_count=token_sequence.frame_count,
    )
    frame_mask = np.ones((token_sequence.frame_count,), dtype=np.bool_)
    confidence = np.ones(
        (token_sequence.frame_count, layout.total_joint_count),
        dtype=np.float32,
    )
    vectorized = BfhVectorizedPose(
        layout=layout,
        values=values,
        validity_mask=np.ones(
            (token_sequence.frame_count, layout.total_joint_count),
            dtype=np.bool_,
        ),
        frame_validity_mask=frame_mask,
        confidence_values=confidence,
        frame_count=token_sequence.frame_count,
        source_sample_id=source.sample_id,
    )
    restored = invert_bfh_standardization(vectorized, standardization_stats)
    return bfh_pose_arrays_from_vectorized(restored)


def _decode_window_token_sequence(
    *,
    decoded: np.ndarray,
    token_sequence: PoseTokenSequence,
    standardization_stats: BfhStandardizationStats,
    layout,
    source: PoseTokenSourceSample,
) -> BfhPoseArrays:
    frame_feature_dim = layout.total_feature_dim
    expected = (token_sequence.token_count, token_sequence.window_size * frame_feature_dim)
    if decoded.shape != expected:
        raise LearnedPoseTokenError(
            "decoded window values must have shape "
            f"({token_sequence.token_count}, {token_sequence.window_size} * total_feature_dim)."
        )
    spec = pose_token_temporal_spec(
        token_sequence.temporal_granularity,
        token_sequence.window_size,
        token_sequence.stride,
    )
    starts = temporal_window_starts(frame_count=token_sequence.frame_count, spec=spec)
    if len(starts) != token_sequence.token_count:
        raise LearnedPoseTokenError(
            "window token_count must equal len(temporal_window_starts(frame_count, spec))."
        )
    window_values = decoded.reshape(
        token_sequence.token_count,
        token_sequence.window_size,
        layout.total_joint_count,
        layout.coordinate_dimensions,
    ).copy()
    frame_shape = (token_sequence.token_count, token_sequence.window_size)
    real_frame_mask = np.zeros(frame_shape, dtype=np.bool_)
    frame_validity_mask = np.zeros(frame_shape, dtype=np.bool_)
    source_frame_indices = np.full(frame_shape, -1, dtype=np.int64)
    for window_index, start in enumerate(starts):
        for offset in range(token_sequence.window_size):
            source_index = start + offset
            if source_index >= token_sequence.frame_count:
                continue
            real_frame_mask[window_index, offset] = True
            frame_validity_mask[window_index, offset] = True
            source_frame_indices[window_index, offset] = source_index
    window_values[~real_frame_mask] = 0.0
    confidence = np.where(
        real_frame_mask[..., None],
        1.0,
        0.0,
    ).astype(np.float32)
    confidence = np.repeat(confidence, layout.total_joint_count, axis=2)
    validity = (
        real_frame_mask[..., None]
        & frame_validity_mask[..., None]
        & (confidence > 0.0)
    )
    windows = BfhPoseWindowSequence(
        schema_version=BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION,
        spec=spec,
        layout=layout,
        source_sample_id=source.sample_id,
        source_frame_count=token_sequence.frame_count,
        values=window_values,
        validity_mask=validity,
        confidence_values=confidence,
        frame_validity_mask=frame_validity_mask,
        real_frame_mask=real_frame_mask,
        source_frame_indices=source_frame_indices,
    )
    merged = merge_bfh_pose_windows(windows).vectorized_pose
    restored = invert_bfh_standardization(merged, standardization_stats)
    expected_frame_count = generated_frame_count_from_token_count(
        token_sequence.token_count,
        token_sequence.temporal_granularity,
        token_sequence.window_size,
        token_sequence.stride,
    )
    if token_sequence.frame_count != expected_frame_count and source.frame_count != token_sequence.frame_count:
        raise LearnedPoseTokenError("decoded window sequence frame_count is inconsistent.")
    return bfh_pose_arrays_from_vectorized(restored)


def export_predicted_pose_samples(
    *,
    decoded_samples: Sequence[DecodedPoseSample],
    topology: ArtifactTopology | None = None,
    output_root: Path | None = None,
    diagnostic_root: Path | None = None,
    producer_key: str,
    producer_stage: str,
    run_name: str,
    split: SampleSplit,
    generation_mode: GeneratedPoseGenerationMode | str,
    seed: int | None,
) -> GeneratedPoseStreamWriteResult:
    """Export decoded samples through the generated-pose contract."""

    materialized = tuple(decoded_samples)
    resolved_generation_mode = GeneratedPoseGenerationMode(generation_mode)
    if not materialized:
        raise LearnedPoseTokenError("decoded_samples must be non-empty.")
    spec = require_model_spec(ModelKey.LEARNED_POSE_TOKEN)
    generated = []
    for index, decoded in enumerate(materialized):
        if decoded.source.split is not SampleSplit(split):
            raise LearnedPoseTokenError("decoded sample split must match export split.")
        identity = GeneratedPoseSourceIdentity(
            canonical_id=spec.canonical_id,
            phase_number=spec.phase_number,
            research_role=spec.research_role.value,
            run_name=run_name,
            text=decoded.source.text,
            source_video_id=decoded.source.source_video_id,
            source_sentence_id=decoded.source.source_sentence_id,
            source_sentence_name=decoded.source.source_sentence_name,
            reference_payload_ref=decoded.source.reference_payload_ref,
            num_candidates_for_sample=1,
            producer_type="model",
            generation_mode=resolved_generation_mode.value,
            length_policy=GeneratedPoseLengthPolicy.PREDICTED_LENGTH.value,
            confidence_policy="synthetic_validity",
            seed=seed,
            failure_reason=None,
        )
        generated.append(
            generated_pose_sample_from_bfh_arrays(
                sample_id=decoded.source.sample_id,
                sentence_name=decoded.source.source_sentence_name,
                split=split,
                generation_index=0,
                arrays=decoded.pose,
                producer_key=producer_key,
                producer_stage=producer_stage,
                identity=identity,
            )
        )
    if topology is not None:
        if diagnostic_root is not None:
            raise LearnedPoseTokenError("diagnostic_root cannot be used with topology.")
        return write_generated_pose_split(
            topology,
            producer_key=producer_key,
            run_name=run_name,
            split=split,
            samples=generated,
        )
    if diagnostic_root is not None:
        paths = diagnostic_generated_pose_paths(Path(diagnostic_root))
        validate_diagnostic_generated_pose_paths(paths)
        return write_generated_pose_split_to_explicit_root(
            manifest_path=paths.manifest_path,
            samples_root=paths.samples_root,
            payload_ref_root=paths.payload_ref_root,
            split=split,
            samples=generated,
        )
    if output_root is None:
        raise LearnedPoseTokenError("output_root is required when topology is not provided.")
    root = Path(output_root)
    return write_generated_pose_split_to_explicit_root(
        manifest_path=root / "manifest.jsonl",
        samples_root=root / "samples",
        payload_ref_root=root.parent,
        split=split,
        samples=generated,
    )


def export_predicted_pose_samples_incremental(
    *,
    decoded_samples,
    topology: ArtifactTopology | None = None,
    output_root: Path | None = None,
    diagnostic_root: Path | None = None,
    producer_key: str,
    producer_stage: str,
    run_name: str,
    split: SampleSplit,
    generation_mode: GeneratedPoseGenerationMode | str,
    seed: int | None,
) -> GeneratedPoseSplitWriteResult:
    """Write decoded pose samples as they are yielded."""

    resolved_split = SampleSplit(split)
    resolved_generation_mode = GeneratedPoseGenerationMode(generation_mode)
    spec = require_model_spec(ModelKey.LEARNED_POSE_TOKEN)
    if topology is not None:
        if diagnostic_root is not None or output_root is not None:
            raise LearnedPoseTokenError("topology cannot be combined with explicit output roots.")
        manifest_path = topology.evaluations.generated_pose_manifest(
            producer_key,
            run_name,
            resolved_split,
        ).path
        samples_root = None
        payload_ref_root = topology.repo_root
    elif diagnostic_root is not None:
        paths = diagnostic_generated_pose_paths(Path(diagnostic_root))
        validate_diagnostic_generated_pose_paths(paths)
        manifest_path = paths.manifest_path
        samples_root = paths.samples_root
        payload_ref_root = paths.payload_ref_root
    else:
        if output_root is None:
            raise LearnedPoseTokenError("output_root is required when topology is not provided.")
        root = Path(output_root)
        manifest_path = root / "manifest.jsonl"
        samples_root = root / "samples"
        payload_ref_root = root.parent
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if samples_root is not None:
        samples_root.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, int]] = set()
    preview_entries = []
    preview_payload_paths = []
    sample_count = 0
    preview_limit = 10
    with manifest_path.open("w", encoding="utf-8") as handle:
        for decoded in decoded_samples:
            if decoded.source.split is not resolved_split:
                raise LearnedPoseTokenError("decoded sample split must match export split.")
            identity = GeneratedPoseSourceIdentity(
                canonical_id=spec.canonical_id,
                phase_number=spec.phase_number,
                research_role=spec.research_role.value,
                run_name=run_name,
                text=decoded.source.text,
                source_video_id=decoded.source.source_video_id,
                source_sentence_id=decoded.source.source_sentence_id,
                source_sentence_name=decoded.source.source_sentence_name,
                reference_payload_ref=decoded.source.reference_payload_ref,
                num_candidates_for_sample=1,
                producer_type="model",
                generation_mode=resolved_generation_mode.value,
                length_policy=GeneratedPoseLengthPolicy.PREDICTED_LENGTH.value,
                confidence_policy="synthetic_validity",
                seed=seed,
                failure_reason=None,
            )
            generated = generated_pose_sample_from_bfh_arrays(
                sample_id=decoded.source.sample_id,
                sentence_name=decoded.source.source_sentence_name,
                split=resolved_split,
                generation_index=0,
                arrays=decoded.pose,
                producer_key=producer_key,
                producer_stage=producer_stage,
                identity=identity,
            )
            issues = validate_generated_pose_sample(generated)
            if issues:
                raise LearnedPoseTokenError(
                    "Invalid generated-pose sample: "
                    + "; ".join(issue.message for issue in issues)
                )
            if topology is not None:
                payload_path = topology.evaluations.generated_pose_sample_file(
                    producer_key,
                    run_name,
                    resolved_split,
                    generated.sample_id,
                    generated.generation_index,
                ).path
                payload_ref = payload_path.relative_to(payload_ref_root).as_posix()
            else:
                if samples_root is None:
                    raise LearnedPoseTokenError("samples_root was not resolved.")
                payload_path = samples_root / _generated_pose_sample_filename(
                    generated.sample_id,
                    generated.generation_index,
                )
                payload_ref = payload_path.relative_to(payload_ref_root).as_posix()
            entry = generated_manifest_entry_from_sample(
                generated,
                generated_payload_ref=payload_ref,
            )
            issues = validate_generated_pose_manifest_entries(
                (entry,),
                expected_split=resolved_split,
            )
            if issues:
                raise LearnedPoseTokenError(
                    "Invalid generated-pose manifest: "
                    + "; ".join(issue.message for issue in issues)
                )
            key = (entry.split.value, entry.sample_id, entry.generation_index)
            if key in seen:
                raise LearnedPoseTokenError("Invalid generated-pose manifest: duplicate manifest key.")
            seen.add(key)
            write_generated_pose_payload(payload_path, generated)
            handle.write(
                json.dumps(
                    generated_manifest_entry_to_record(entry),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
            if len(preview_entries) < preview_limit:
                preview_entries.append(entry)
                preview_payload_paths.append(payload_path)
            sample_count += 1
    if sample_count <= 0:
        raise LearnedPoseTokenError("decoded_samples must be non-empty.")
    resolved_sample_root = samples_root
    if resolved_sample_root is None:
        resolved_sample_root = manifest_path.parent / "samples"
    return GeneratedPoseStreamWriteResult(
        split=resolved_split,
        manifest_path=manifest_path,
        sample_root=resolved_sample_root,
        sample_count=sample_count,
        payload_count=sample_count,
        preview_entries=tuple(preview_entries),
        preview_payload_paths=tuple(preview_payload_paths),
        preview_limit=preview_limit,
    )


def _generated_pose_sample_filename(sample_id: str, generation_index: int) -> str:
    token = str(sample_id)
    if not token.strip() or token in {".", ".."} or "/" in token or "\\" in token or "." in token:
        raise LearnedPoseTokenError("generated-pose sample_id must be a concrete safe path token.")
    return f"{token}__g{int(generation_index)}.npz"


__all__ = [
    "DecodedPoseSample",
    "decode_token_sequence_to_bfh_pose",
    "export_predicted_pose_samples",
    "export_predicted_pose_samples_incremental",
]
