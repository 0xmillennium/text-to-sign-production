"""Deterministic full-BFH export for the articulator-aware provider."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    GeneratedPoseSample,
    GeneratedPoseSplitWriteResult,
    GeneratedPoseStreamWriteResult,
    generated_manifest_entry_from_sample,
    generated_manifest_entry_to_record,
    validate_generated_pose_manifest_entries,
    validate_generated_pose_sample,
    write_generated_pose_payload,
    write_generated_pose_split,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    bfh_pose_arrays_from_vectorized,
    unflatten_bfh_pose_values,
)
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP,
    ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL,
    ArticulatorAwareConfig,
)
from text_to_sign_production.modeling.candidates.articulator_aware.dataset import (
    ArticulatorSourceSample,
)
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)
from text_to_sign_production.modeling.candidates.articulator_aware.model import (
    ArticulatorFactorizedTemporalModel,
    ArticulatorLengthPredictor,
    ChannelFusionPoseModel,
    clamp_articulator_predicted_frame_count,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    ArticulatorChannelPartitionPolicy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.spec import (
    ARTICULATOR_AWARE_CANONICAL_ID,
    ARTICULATOR_AWARE_MODEL_KEY,
    ARTICULATOR_AWARE_PHASE_NUMBER,
    ARTICULATOR_AWARE_RESEARCH_ROLE,
)
from text_to_sign_production.modeling.candidates.articulator_aware.trainer import (
    ARTICULATOR_CHECKPOINT_SCHEMA_VERSION,
)
from text_to_sign_production.modeling.data.bfh_schema import BfhPoseArrays
from text_to_sign_production.modeling.inference.generated_pose import (
    GeneratedPoseSourceIdentity,
    generated_pose_sample_from_bfh_arrays,
)


@dataclass(frozen=True, slots=True)
class ArticulatorGeneratedSample:
    source: ArticulatorSourceSample
    pose: BfhPoseArrays
    predicted_frame_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, ArticulatorSourceSample):
            raise ArticulatorAwareError("source must be an ArticulatorSourceSample.")
        if not isinstance(self.pose, BfhPoseArrays):
            raise ArticulatorAwareError("pose must be a BfhPoseArrays instance.")
        if self.predicted_frame_count != self.pose.frame_count or self.predicted_frame_count < 1:
            raise ArticulatorAwareError("predicted_frame_count must match generated pose.")


def load_articulator_checkpoint_models(
    *,
    checkpoint_path: Path,
    config: ArticulatorAwareConfig,
    partition_policy: ArticulatorChannelPartitionPolicy,
    device: torch.device,
    run_name: str,
    restored_model_config: Mapping[str, object],
    restored_length_predictor_config: Mapping[str, object],
    text_encoder_output_dim: int | None = None,
) -> tuple[ChannelFusionPoseModel | ArticulatorFactorizedTemporalModel, ArticulatorLengthPredictor]:
    """Load the selected checkpoint without fallback to another role."""

    if not Path(checkpoint_path).is_file():
        raise ArticulatorAwareError(
            f"selected articulator_aware checkpoint is missing: {checkpoint_path}. "
            "Restore the selected checkpoint before inference."
        )
    try:
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArticulatorAwareError(
            f"selected articulator_aware checkpoint could not be loaded: {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema_version") != ARTICULATOR_CHECKPOINT_SCHEMA_VERSION:
        raise ArticulatorAwareError("articulator_aware checkpoint schema_version is unsupported.")
    if checkpoint.get("model_key") != ARTICULATOR_AWARE_MODEL_KEY:
        raise ArticulatorAwareError("articulator_aware checkpoint model_key is invalid.")
    if checkpoint.get("run_name") != run_name:
        raise ArticulatorAwareError(
            "articulator_aware checkpoint run_name does not match the requested restored run."
        )
    if checkpoint.get("model_config") != dict(restored_model_config):
        raise ArticulatorAwareError(
            "articulator_aware checkpoint model_config does not match restored model_config.json."
        )
    if checkpoint.get("length_predictor_config") != dict(restored_length_predictor_config):
        raise ArticulatorAwareError(
            "articulator_aware checkpoint length_predictor_config does not match restored "
            "length_predictor_config.json."
        )
    text_dim = int(restored_model_config["text_embedding_dim"])
    _require_restore_match(
        restored_model_config,
        field_name="architecture",
        expected=config.structure_variant.architecture,
    )
    _require_restore_match(
        restored_model_config,
        field_name="fusion",
        expected=config.structure_variant.fusion,
    )
    _require_restore_match(
        restored_model_config,
        field_name="temporal_layers",
        expected=config.structure_variant.temporal_layers,
    )
    _require_restore_match(
        restored_model_config,
        field_name="fusion_hidden_dim",
        expected=config.structure_variant.fusion_hidden_dim,
    )
    if checkpoint.get("provider_config") != config.to_dict():
        raise ArticulatorAwareError(
            "articulator_aware checkpoint provider_config does not match the effective config."
        )
    if checkpoint.get("selection_metric") != config.checkpoints.selection_metric:
        raise ArticulatorAwareError(
            "articulator_aware checkpoint selection_metric does not match the effective config."
        )
    if text_encoder_output_dim is not None and text_dim != text_encoder_output_dim:
        raise ArticulatorAwareError(
            f"checkpoint text_embedding_dim={text_dim} but current text encoder "
            f"output_dim={text_encoder_output_dim}. Use the original text encoder config or retrain."
        )
    length_predictor = ArticulatorLengthPredictor(
        text_embedding_dim=text_dim,
        hidden_dim=config.structure_variant.hidden_dim,
    )
    if config.structure_variant.architecture == ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP:
        model: ChannelFusionPoseModel | ArticulatorFactorizedTemporalModel = ChannelFusionPoseModel(
            partition_policy=partition_policy,
            text_embedding_dim=text_dim,
            hidden_dim=config.structure_variant.hidden_dim,
            dropout=config.structure_variant.dropout,
            max_positions=config.length.max_positions,
            length_predictor=length_predictor,
        ).to(device)
    elif config.structure_variant.architecture == ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL:
        model = ArticulatorFactorizedTemporalModel(
            partition_policy=partition_policy,
            text_embedding_dim=text_dim,
            hidden_dim=config.structure_variant.hidden_dim,
            fusion_hidden_dim=int(config.structure_variant.fusion_hidden_dim),
            temporal_layers=config.structure_variant.temporal_layers,
            dropout=config.structure_variant.dropout,
            max_positions=config.length.max_positions,
            length_predictor=length_predictor,
        ).to(device)
    else:
        raise ArticulatorAwareError(
            "structure_variant.architecture must be one of "
            "{'channel_fusion_mlp', 'articulator_factorized_temporal'}."
        )
    try:
        model.load_state_dict(cast(Any, checkpoint["model_state_dict"]))
    except (KeyError, RuntimeError) as exc:
        raise ArticulatorAwareError(
            "articulator_aware checkpoint model state is missing or incompatible with the restored config."
        ) from exc
    model.eval()
    return model, model.length_predictor


def generate_articulator_pose_for_source(
    *,
    model: ChannelFusionPoseModel | ArticulatorFactorizedTemporalModel,
    length_predictor: ArticulatorLengthPredictor,
    text_embedding: np.ndarray | torch.Tensor,
    source: ArticulatorSourceSample,
    config: ArticulatorAwareConfig,
    partition_policy: ArticulatorChannelPartitionPolicy,
) -> tuple[BfhPoseArrays, dict[str, object]]:
    """Generate a sequence using predicted length, never the reference pose length."""

    if model.partition_policy != partition_policy:
        raise ArticulatorAwareError("generation partition policy must match the loaded model.")
    device = next(model.parameters()).device
    text = torch.tensor(
        np.asarray(text_embedding, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    if text.shape != (model.text_embedding_dim,):
        raise ArticulatorAwareError(
            f"text_embedding must have shape ({model.text_embedding_dim},)."
        )
    with torch.no_grad():
        predicted_length = float(length_predictor(text.unsqueeze(0)).cpu().item())
    frame_count = clamp_articulator_predicted_frame_count(
        predicted_length,
        min_generated_frames=config.length.min_generated_frames,
        max_generated_frames=config.length.max_generated_frames,
        max_positions=config.length.max_positions,
    )
    positions = torch.arange(frame_count, dtype=torch.long, device=device)
    with torch.no_grad():
        if isinstance(model, ArticulatorFactorizedTemporalModel):
            flat = model(text.unsqueeze(0), positions.unsqueeze(0)).full_pose_values.squeeze(0)
        else:
            flat = model(text.unsqueeze(0).repeat(frame_count, 1), positions).full_pose_values
    generated_flat = flat.detach().cpu().numpy()
    if generated_flat.shape != (frame_count, partition_policy.layout.total_feature_dim):
        raise ArticulatorAwareError("generated pose does not match full-BFH flattened shape.")
    vectorized = BfhVectorizedPose(
        layout=partition_policy.layout,
        values=unflatten_bfh_pose_values(
            generated_flat,
            layout=partition_policy.layout,
            frame_count=frame_count,
        ),
        validity_mask=np.ones(
            (frame_count, partition_policy.layout.total_joint_count),
            dtype=np.bool_,
        ),
        frame_validity_mask=np.ones((frame_count,), dtype=np.bool_),
        confidence_values=np.ones(
            (frame_count, partition_policy.layout.total_joint_count),
            dtype=np.float32,
        ),
        frame_count=frame_count,
        source_sample_id=source.sample_id,
    )
    return bfh_pose_arrays_from_vectorized(vectorized), {
        "predicted_frame_count": frame_count,
        "predicted_length_value": predicted_length,
        "length_policy": config.length.policy,
        "confidence_policy": config.generation.confidence_policy,
        "generation_mode": config.generation.generation_mode,
    }


def _require_restore_match(
    restored_model_config: Mapping[str, object],
    *,
    field_name: str,
    expected: object,
) -> None:
    observed = restored_model_config.get(field_name)
    if observed != expected:
        if field_name == "architecture":
            raise ArticulatorAwareError(
                f"checkpoint architecture={observed!r} but current config requires "
                f"{expected!r}. Use the original config or retrain."
            )
        raise ArticulatorAwareError(
            f"checkpoint {field_name}={observed!r} but current config requires "
            f"{field_name}={expected!r}. Use the original config or retrain."
        )


def generated_pose_sample_for_articulator(
    *,
    generated: ArticulatorGeneratedSample,
    config: ArticulatorAwareConfig,
    run_name: str,
    seed: int | None,
    producer_stage: str,
) -> GeneratedPoseSample:
    identity = GeneratedPoseSourceIdentity(
        canonical_id=ARTICULATOR_AWARE_CANONICAL_ID,
        phase_number=ARTICULATOR_AWARE_PHASE_NUMBER,
        research_role=ARTICULATOR_AWARE_RESEARCH_ROLE,
        run_name=run_name,
        text=generated.source.text,
        source_video_id=generated.source.source_video_id,
        source_sentence_id=generated.source.source_sentence_id,
        source_sentence_name=generated.source.source_sentence_name,
        reference_payload_ref=generated.source.reference_payload_ref,
        num_candidates_for_sample=1,
        producer_type="model",
        generation_mode=config.generation.generation_mode,
        length_policy=config.length.policy,
        confidence_policy=config.generation.confidence_policy,
        seed=seed,
    )
    return generated_pose_sample_from_bfh_arrays(
        sample_id=generated.source.sample_id,
        sentence_name=generated.source.source_sentence_name,
        split=generated.source.split,
        generation_index=0,
        arrays=generated.pose,
        producer_key=ARTICULATOR_AWARE_MODEL_KEY,
        producer_stage=producer_stage,
        identity=identity,
    )


def export_articulator_generated_pose_split(
    *,
    topology: ArtifactTopology,
    run_name: str,
    split: SampleSplit,
    generated_samples: Sequence[GeneratedPoseSample],
) -> GeneratedPoseSplitWriteResult:
    """Write validation outputs through the shared generated-pose surface."""

    samples = tuple(generated_samples)
    if not samples:
        raise ArticulatorAwareError("export_generated_pose requires generated samples.")
    return write_generated_pose_split(
        topology,
        producer_key=ARTICULATOR_AWARE_MODEL_KEY,
        run_name=run_name,
        split=split,
        samples=samples,
    )


def export_articulator_generated_pose_stream(
    *,
    topology: ArtifactTopology,
    run_name: str,
    split: SampleSplit,
    generated_samples: Iterable[GeneratedPoseSample],
    on_sample_written=None,
) -> GeneratedPoseStreamWriteResult:
    """Write generated poses one sample at a time without materializing the split."""

    resolved_split = SampleSplit(split)
    manifest_path = topology.evaluations.generated_pose_manifest(
        ARTICULATOR_AWARE_MODEL_KEY,
        run_name,
        resolved_split,
    ).path
    sample_root = topology.evaluations.generated_pose_samples_root(
        ARTICULATOR_AWARE_MODEL_KEY,
        run_name,
        resolved_split,
    ).path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    preview_entries = []
    preview_payload_paths = []
    seen: set[tuple[str, str, int]] = set()
    sample_count = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in generated_samples:
            issues = validate_generated_pose_sample(sample)
            if issues:
                raise ArticulatorAwareError(
                    "Invalid generated-pose sample: "
                    + "; ".join(issue.message for issue in issues)
                )
            if sample.producer_key != ARTICULATOR_AWARE_MODEL_KEY or sample.run_name != run_name:
                raise ArticulatorAwareError("sample producer_key/run_name does not match request.")
            if sample.split is not resolved_split:
                raise ArticulatorAwareError("sample split does not match request.")
            payload_path = topology.evaluations.generated_pose_sample_file(
                ARTICULATOR_AWARE_MODEL_KEY,
                run_name,
                resolved_split,
                sample.sample_id,
                sample.generation_index,
            ).path
            payload_ref = payload_path.relative_to(topology.repo_root).as_posix()
            entry = generated_manifest_entry_from_sample(
                sample,
                generated_payload_ref=payload_ref,
            )
            write_generated_pose_payload(payload_path, sample)
            issues = validate_generated_pose_manifest_entries((entry,), expected_split=resolved_split)
            key = (entry.split.value, entry.sample_id, entry.generation_index)
            if key in seen:
                raise ArticulatorAwareError("Invalid generated-pose manifest: duplicate manifest key.")
            seen.add(key)
            if issues:
                raise ArticulatorAwareError(
                    "Invalid generated-pose manifest: "
                    + "; ".join(issue.message for issue in issues)
                )
            handle.write(
                json.dumps(
                    generated_manifest_entry_to_record(entry),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            if len(preview_entries) < 10:
                preview_entries.append(entry)
                preview_payload_paths.append(payload_path)
            sample_count += 1
            if on_sample_written is not None:
                on_sample_written()
    if sample_count == 0:
        raise ArticulatorAwareError("export_generated_pose requires generated samples.")
    return GeneratedPoseStreamWriteResult(
        split=resolved_split,
        manifest_path=manifest_path,
        sample_root=sample_root,
        sample_count=sample_count,
        payload_count=sample_count,
        preview_entries=tuple(preview_entries),
        preview_payload_paths=tuple(preview_payload_paths),
        preview_limit=10,
    )


__all__ = [
    "ArticulatorGeneratedSample",
    "export_articulator_generated_pose_split",
    "export_articulator_generated_pose_stream",
    "generate_articulator_pose_for_source",
    "generated_pose_sample_for_articulator",
    "load_articulator_checkpoint_models",
]
