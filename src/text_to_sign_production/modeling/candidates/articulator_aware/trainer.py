"""Channel-weighted training for the articulator-aware provider."""

from __future__ import annotations

import math
import random
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.io import write_json
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    flatten_bfh_vectorized_pose,
    unflatten_bfh_pose_values,
)
from text_to_sign_production.modeling.backbones.text_encoder import (
    build_text_encoder,
    text_encoder_metadata,
)
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL,
    ArticulatorAwareConfig,
    ChannelMaskStrategyConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticConsistencyObjectiveConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.training import (
    compute_semantic_training_loss,
)
from text_to_sign_production.modeling.candidates.articulator_aware.dataset import (
    ArticulatorSourceSample,
    build_articulator_frame_training_samples,
)
from text_to_sign_production.modeling.candidates.articulator_aware.diagnostics import (
    CHANNEL_DIAGNOSTIC_RECORD_SCHEMA_VERSION,
    ChannelDiagnosticRecord,
    aggregate_channel_diagnostics,
    build_channel_diagnostic_records,
)
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)
from text_to_sign_production.modeling.candidates.articulator_aware.io import (
    write_channel_diagnostic_records_jsonl,
    write_channel_diagnostic_summary_json,
    write_channel_loss_records_jsonl,
)
from text_to_sign_production.modeling.candidates.articulator_aware.losses import (
    ChannelLossRecord,
    ChannelLossWeightingPolicy,
    compute_channel_velocity_loss_torch,
    compute_channel_weighted_sequence_reconstruction_loss_torch,
    compute_channel_weighted_reconstruction_loss_numpy,
    compute_cross_channel_temporal_consistency_loss_torch,
    compute_hand_motion_coordination_loss_torch,
)
from text_to_sign_production.modeling.candidates.articulator_aware.masks import (
    build_channel_loss_masks,
    summarize_channel_masks,
)
from text_to_sign_production.modeling.candidates.articulator_aware.model import (
    ArticulatorFactorizedTemporalModel,
    ArticulatorLengthPredictor,
    ChannelFusionPoseModel,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    ArticulatorChannelPartitionPolicy,
)
from text_to_sign_production.modeling.training.checkpoints import (
    CHECKPOINT_METADATA_SCHEMA_VERSION,
    CheckpointRole,
    ModelCheckpointMetadata,
    write_checkpoint_metadata_json,
)
from text_to_sign_production.modeling.training.metrics import (
    TRAINING_METRIC_SCHEMA_VERSION,
    TrainingMetricRecord,
    write_training_metric_records_jsonl,
)
from text_to_sign_production.modeling.training.precision import (
    ResolvedPrecisionPolicy,
    autocast_context,
    resolve_precision_policy,
)
from text_to_sign_production.modeling.data_surfaces import ModelDataSurface
from text_to_sign_production.modeling.data_surfaces import ModelDataSurfaceReader
from text_to_sign_production.modeling.data_surfaces import surface_batch_count

ARTICULATOR_CHECKPOINT_SCHEMA_VERSION = "t2sp-articulator-aware-checkpoint-v1"
ARTICULATOR_MODEL_CONFIG_SCHEMA_VERSION = "t2sp-articulator-model-config-v1"
ARTICULATOR_LENGTH_CONFIG_SCHEMA_VERSION = "t2sp-articulator-length-predictor-config-v1"


@dataclass(frozen=True, slots=True)
class ArticulatorTrainingResult:
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_metadata_path: Path
    last_metadata_path: Path
    training_metrics_path: Path
    train_channel_loss_records_path: Path
    validation_channel_loss_records_path: Path
    train_channel_diagnostics_path: Path
    validation_channel_diagnostics_path: Path
    train_channel_diagnostics_summary_path: Path
    validation_channel_diagnostics_summary_path: Path
    model_config_path: Path
    length_predictor_config_path: Path
    final_train_loss: float
    validation_channel_weighted_loss: float
    validation_length_loss: float
    completed_epoch: int
    global_step: int
    runtime_trace: dict[str, object] | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name.endswith("_path"):
                object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "final_train_loss",
            "validation_channel_weighted_loss",
            "validation_length_loss",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ArticulatorAwareError(f"{name} must be finite.")


@dataclass(frozen=True, slots=True)
class ArticulatorSequenceBatch:
    text_embeddings: torch.Tensor
    positions: torch.Tensor
    target_values: torch.Tensor
    frame_mask: torch.Tensor
    channel_masks: Mapping[PoseChannel, torch.Tensor]
    source_lengths: torch.Tensor
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.text_embeddings.ndim != 2:
            raise ArticulatorAwareError("sequence batch text_embeddings must be rank 2.")
        if self.positions.ndim != 2:
            raise ArticulatorAwareError("sequence batch positions must be rank 2.")
        if self.target_values.ndim != 3:
            raise ArticulatorAwareError("sequence batch target_values must be rank 3.")
        batch, frames, features = self.target_values.shape
        if self.positions.shape != (batch, frames) or self.frame_mask.shape != (batch, frames):
            raise ArticulatorAwareError("sequence batch masks/positions must align with targets.")
        if self.source_lengths.shape != (batch,):
            raise ArticulatorAwareError("sequence batch source_lengths must have shape (batch,).")
        if len(self.sample_ids) != batch:
            raise ArticulatorAwareError("sequence batch sample_ids must match batch size.")
        for channel, mask in self.channel_masks.items():
            PoseChannel(channel)
            if not isinstance(mask, torch.Tensor) or mask.shape != (batch, frames, features):
                raise ArticulatorAwareError("sequence batch channel masks must align with targets.")


def compute_channel_weighted_reconstruction_loss_torch(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    channel_masks: Mapping[PoseChannel, torch.Tensor],
    weighting: ChannelLossWeightingPolicy,
) -> tuple[torch.Tensor, Mapping[PoseChannel, torch.Tensor | None]]:
    """Compute full-vector masked MSE with skipped channels represented by None."""

    if predicted.ndim != 2 or target.shape != predicted.shape:
        raise ArticulatorAwareError("predicted and target must share shape (batch, feature_dim).")
    if not isinstance(weighting, ChannelLossWeightingPolicy):
        raise ArticulatorAwareError("weighting must be ChannelLossWeightingPolicy.")
    losses: dict[PoseChannel, torch.Tensor | None] = {}
    active: list[tuple[PoseChannel, torch.Tensor]] = []
    for channel in weighting.normalized_channel_weights:
        mask = channel_masks.get(channel)
        if not isinstance(mask, torch.Tensor) or mask.shape != predicted.shape:
            raise ArticulatorAwareError(
                f"channel mask for {channel.value!r} must match predicted full-BFH shape."
            )
        valid = mask.to(dtype=torch.bool)
        if not torch.any(valid):
            losses[channel] = None
            continue
        value = torch.mean(torch.square(predicted[valid] - target[valid]))
        if not torch.isfinite(value):
            raise ArticulatorAwareError(f"{channel.value} torch channel loss is not finite.")
        losses[channel] = value
        active.append((channel, value))
    if not active:
        raise ArticulatorAwareError("no valid channels remain for torch reconstruction loss.")
    weight_sum = sum(weighting.normalized_channel_weights[channel] for channel, _ in active)
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        raise ArticulatorAwareError("valid channel weights must have a positive finite sum.")
    total = sum(
        value * (weighting.normalized_channel_weights[channel] / weight_sum)
        for channel, value in active
    )
    if not torch.isfinite(total):
        raise ArticulatorAwareError("weighted torch reconstruction loss is not finite.")
    return total, losses


def compute_articulator_length_loss(
    *,
    predicted_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute finite source-length regression loss."""

    if predicted_lengths.shape != target_lengths.shape or predicted_lengths.ndim != 1:
        raise ArticulatorAwareError("predicted_lengths and target_lengths must share shape (batch,).")
    value = torch.nn.functional.mse_loss(predicted_lengths, target_lengths)
    if not torch.isfinite(value):
        raise ArticulatorAwareError("articulator length loss is not finite.")
    return value


def train_articulator_model(
    *,
    config: ArticulatorAwareConfig,
    train_samples: Sequence[ArticulatorSourceSample],
    validation_samples: Sequence[ArticulatorSourceSample],
    partition_policy: ArticulatorChannelPartitionPolicy,
    mask_config: ChannelMaskStrategyConfig,
    weighting: ChannelLossWeightingPolicy,
    output_root: Path,
    run_name: str,
    seed: int | None,
    semantic_config: SemanticConsistencyObjectiveConfig | None = None,
    progress=None,
    precision_policy_name: str = "auto",
) -> ArticulatorTrainingResult:
    """Train a deterministic channel-head model and materialize real support artifacts."""

    if config.structure_variant.architecture == ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL:
        return _train_factorized_temporal_model(
            config=config,
            train_samples=train_samples,
            validation_samples=validation_samples,
            partition_policy=partition_policy,
            mask_config=mask_config,
            weighting=weighting,
            output_root=output_root,
            run_name=run_name,
            seed=seed,
            semantic_config=semantic_config,
            progress=progress,
            precision_policy_name=precision_policy_name,
        )
    if semantic_config is not None:
        raise ArticulatorAwareError(
            "semantic_consistency training objective for articulator_aware requires "
            "articulator_factorized_temporal; channel_fusion_mlp only supports post-generation "
            "proxy diagnostics."
        )
    train_sources = tuple(train_samples)
    validation_sources = tuple(validation_samples)
    if not train_sources or not validation_sources:
        raise ArticulatorAwareError("training requires non-empty train and validation samples.")
    _validate_source_lengths((*train_sources, *validation_sources), config=config)
    with _progress_task(
        progress,
        operation="build_frame_training_samples",
        label="articulator build frame training samples",
        unit="sample",
        total=len(train_sources),
        total_semantics="train source samples expanded to frame targets",
        allowed_counters=("frames",),
    ) as task:
        train_frames = build_articulator_frame_training_samples(
            source_samples=train_sources,
            partition_policy=partition_policy,
            mask_config=mask_config,
            on_source_built=lambda index, total, sample_id, frames: task.advance(
                1, counters={"frames": frames}
            ),
        )
    build_articulator_frame_training_samples(
        source_samples=validation_sources,
        partition_policy=partition_policy,
        mask_config=mask_config,
    )
    run_seed = config.training.seed if seed is None else seed
    if run_seed is None:
        run_seed = 0
    _set_seed(run_seed)
    device = _resolve_device(config.training.device)
    precision_policy = resolve_precision_policy(precision_policy_name, device)
    text_encoder = build_text_encoder(config.text_encoder)
    text_dim = text_encoder.output_dim
    length_predictor = ArticulatorLengthPredictor(
        text_embedding_dim=text_dim,
        hidden_dim=config.structure_variant.hidden_dim,
    )
    model = ChannelFusionPoseModel(
        partition_policy=partition_policy,
        text_embedding_dim=text_dim,
        hidden_dim=config.structure_variant.hidden_dim,
        dropout=config.structure_variant.dropout,
        max_positions=config.length.max_positions,
        length_predictor=length_predictor,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    embedding_sources = (*train_sources, *validation_sources)
    with _progress_task(
        progress,
        operation="encode_text",
        label="articulator encode train and validation text",
        unit="sample",
        total=len(embedding_sources),
        total_semantics="train and validation text embeddings encoded",
    ) as task:
        encoded_text = text_encoder.encode_batch(
            [source.text for source in embedding_sources],
            sentence_names=[source.source_sentence_name for source in embedding_sources],
        )
        task.advance(len(embedding_sources))
    embeddings = {
        source.sample_id: np.asarray(output.pooled_embedding, dtype=np.float32)
        for source, output in zip(embedding_sources, encoded_text, strict=True)
    }
    metrics: list[TrainingMetricRecord] = []
    global_step = 0
    final_train_loss = 0.0
    frames_by_source = {
        source.sample_id: tuple(
            frame for frame in train_frames if frame.source.sample_id == source.sample_id
        )
        for source in train_sources
    }
    for epoch in range(config.training.max_epochs):
        model.train()
        losses: list[float] = []
        source_batches = _batches(train_sources, config.training.batch_size)
        with _progress_task(
            progress,
            operation=f"train_epoch_{epoch + 1}",
            label=f"articulator train epoch {epoch + 1}/{config.training.max_epochs}",
            unit="batch",
            total=len(source_batches),
            total_semantics="articulator source batches in epoch",
            allowed_counters=("loss",),
        ) as task:
            for source_batch in source_batches:
                source_frames = tuple(
                    frame
                    for source in source_batch
                    for frame in frames_by_source[source.sample_id]
                )
                optimizer.zero_grad(set_to_none=True)
                reconstruction_value = _backprop_frame_chunk_reconstruction(
                    model=model,
                    frames=source_frames,
                    frame_batch_size=config.training.frame_batch_size,
                    embeddings=embeddings,
                    device=device,
                    weighting=weighting,
                    precision_policy=precision_policy,
                )
                source_text = torch.as_tensor(
                    np.stack([embeddings[source.sample_id] for source in source_batch]),
                    dtype=torch.float32,
                    device=device,
                )
                target_lengths = torch.as_tensor(
                    [float(source.frame_count) for source in source_batch],
                    dtype=torch.float32,
                    device=device,
                )
                with autocast_context(precision_policy):
                    length_loss = compute_articulator_length_loss(
                        predicted_lengths=model.length_predictor(source_text),
                        target_lengths=target_lengths,
                    )
                weighted_length_loss = float(config.length.loss_weight) * length_loss
                if not torch.isfinite(weighted_length_loss):
                    raise ArticulatorAwareError("articulator training length loss is not finite.")
                weighted_length_loss.backward()
                optimizer.step()
                value = reconstruction_value + float(weighted_length_loss.detach().cpu().item())
                if not math.isfinite(value):
                    raise ArticulatorAwareError("articulator training total loss is not finite.")
                losses.append(value)
                metrics.append(_metric("train", "total_loss", value, epoch, global_step))
                task.advance(1, counters={"loss": f"{value:.6g}"})
                global_step += 1
        final_train_loss = _finite_mean(losses, "final_train_loss")
    with _progress_task(
        progress,
        operation="evaluate_train_sources",
        label="articulator evaluate train sources",
        unit="sample",
        total=len(train_sources),
        total_semantics="train sources evaluated",
    ) as task:
        train_channel, train_length, train_records = _evaluate_sources(
            model,
            train_sources,
            embeddings=embeddings,
            partition_policy=partition_policy,
            weighting=weighting,
            device=device,
            precision_policy=precision_policy,
        )
        task.advance(len(train_sources))
    with _progress_task(
        progress,
        operation="evaluate_validation_sources",
        label="articulator evaluate validation sources",
        unit="sample",
        total=len(validation_sources),
        total_semantics="validation sources evaluated",
    ) as task:
        val_channel, val_length, val_records = _evaluate_sources(
            model,
            validation_sources,
            embeddings=embeddings,
            partition_policy=partition_policy,
            weighting=weighting,
            device=device,
            precision_policy=precision_policy,
        )
        task.advance(len(validation_sources))
    validation_auxiliary = _evaluate_sequence_auxiliary_metrics(
        model,
        validation_sources,
        embeddings=embeddings,
        partition_policy=partition_policy,
        mask_config=mask_config,
        weighting=weighting,
        device=device,
        precision_policy=precision_policy,
    )
    metrics.extend(
        (
            _metric("train", "channel_weighted_loss", train_channel, config.training.max_epochs - 1, global_step),
            _metric("train", "length_loss", train_length, config.training.max_epochs - 1, global_step),
            _metric("val", config.checkpoints.selection_metric, val_channel, config.training.max_epochs - 1, global_step),
            _metric("val", "validation_length_loss", val_length, config.training.max_epochs - 1, global_step),
        )
    )
    for metric_name, metric_value in validation_auxiliary.items():
        metrics.append(_metric("val", metric_name, metric_value, config.training.max_epochs - 1, global_step))
    model_run_root = Path(output_root)
    checkpoint_root = model_run_root / "checkpoints"
    artifact_root = model_run_root / "intermediates" / "articulator"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ARTICULATOR_CHECKPOINT_SCHEMA_VERSION,
        "model_key": config.identity.model_key,
        "run_name": run_name,
        "model_state_dict": model.state_dict(),
        "length_predictor_state_dict": model.length_predictor.state_dict(),
        "model_config": build_articulator_model_config_payload(
            config,
            partition_policy,
            text_embedding_dim=text_dim,
            text_encoder_metadata_payload=text_encoder_metadata(text_encoder),
        ),
        "length_predictor_config": build_articulator_length_config_payload(
            config,
            text_embedding_dim=text_dim,
        ),
        "provider_config": config.to_dict(),
        "text_embedding_dim": text_dim,
        "text_encoder_config": config.text_encoder.to_dict(),
        "text_encoder_metadata": text_encoder_metadata(text_encoder),
        "run_seed": run_seed,
        "selection_metric": config.checkpoints.selection_metric,
        "selection_metric_value": val_channel,
    }
    best_checkpoint = checkpoint_root / "best.pt"
    last_checkpoint = checkpoint_root / "last.pt"
    torch.save(payload, best_checkpoint)
    torch.save(payload, last_checkpoint)
    model_config_path = artifact_root / "model_config.json"
    length_config_path = artifact_root / "length_predictor_config.json"
    write_json(
        model_config_path,
        build_articulator_model_config_payload(
            config,
            partition_policy,
            text_embedding_dim=text_dim,
            text_encoder_metadata_payload=text_encoder_metadata(text_encoder),
        ),
    )
    write_json(
        length_config_path,
        build_articulator_length_config_payload(config, text_embedding_dim=text_dim),
    )
    metrics_path = artifact_root / "training_metrics.jsonl"
    write_training_metric_records_jsonl(metrics_path, metrics)
    train_loss_path = artifact_root / "channel_loss_records_train.jsonl"
    val_loss_path = artifact_root / "channel_loss_records_val.jsonl"
    write_channel_loss_records_jsonl(train_loss_path, train_records)
    write_channel_loss_records_jsonl(val_loss_path, val_records)
    train_diagnostics, train_summary = _diagnostics(
        train_sources, train_records, split=SampleSplit.TRAIN, partition_policy=partition_policy, mask_config=mask_config
    )
    val_diagnostics, val_summary = _diagnostics(
        validation_sources, val_records, split=SampleSplit.VAL, partition_policy=partition_policy, mask_config=mask_config
    )
    train_diag_path = artifact_root / "channel_diagnostics_train.jsonl"
    val_diag_path = artifact_root / "channel_diagnostics_val.jsonl"
    train_summary_path = artifact_root / "channel_diagnostics_summary_train.json"
    val_summary_path = artifact_root / "channel_diagnostics_summary_val.json"
    write_channel_diagnostic_records_jsonl(train_diag_path, train_diagnostics)
    write_channel_diagnostic_records_jsonl(val_diag_path, val_diagnostics)
    write_channel_diagnostic_summary_json(train_summary_path, train_summary)
    write_channel_diagnostic_summary_json(val_summary_path, val_summary)
    best_metadata = artifact_root / "best_metadata.json"
    last_metadata = artifact_root / "last_metadata.json"
    for role, checkpoint, metadata in (
        (CheckpointRole.BEST, best_checkpoint, best_metadata),
        (CheckpointRole.LAST, last_checkpoint, last_metadata),
    ):
        write_checkpoint_metadata_json(
            metadata,
            ModelCheckpointMetadata(
                schema_version=CHECKPOINT_METADATA_SCHEMA_VERSION,
                model_key=config.identity.model_key,
                run_name=run_name,
                role=role,
                stage_name="train_structure_aware",
                checkpoint_path=checkpoint,
                epoch=config.training.max_epochs - 1,
                global_step=global_step,
                selection_metric=config.checkpoints.selection_metric,
                selection_metric_value=val_channel,
                lower_is_better=config.checkpoints.lower_is_better,
                created_at_utc=datetime.now(UTC).isoformat(),
                extra={
                    "validation_length_loss": val_length,
                    "length_policy": config.length.policy,
                    "model_config_path": str(model_config_path),
                    "length_predictor_config_path": str(length_config_path),
                },
            ),
        )
    return ArticulatorTrainingResult(
        best_checkpoint_path=best_checkpoint,
        last_checkpoint_path=last_checkpoint,
        best_metadata_path=best_metadata,
        last_metadata_path=last_metadata,
        training_metrics_path=metrics_path,
        train_channel_loss_records_path=train_loss_path,
        validation_channel_loss_records_path=val_loss_path,
        train_channel_diagnostics_path=train_diag_path,
        validation_channel_diagnostics_path=val_diag_path,
        train_channel_diagnostics_summary_path=train_summary_path,
        validation_channel_diagnostics_summary_path=val_summary_path,
        model_config_path=model_config_path,
        length_predictor_config_path=length_config_path,
        final_train_loss=final_train_loss,
        validation_channel_weighted_loss=val_channel,
        validation_length_loss=val_length,
        completed_epoch=config.training.max_epochs - 1,
        global_step=global_step,
    )


_train_articulator_model_legacy = train_articulator_model


def train_articulator_model_from_surfaces(
    *,
    config: ArticulatorAwareConfig,
    train_surface: ModelDataSurface,
    validation_surface: ModelDataSurface,
    train_source_surface: ModelDataSurface,
    validation_source_surface: ModelDataSurface,
    partition_policy: ArticulatorChannelPartitionPolicy,
    mask_config: ChannelMaskStrategyConfig,
    weighting: ChannelLossWeightingPolicy,
    output_root: Path,
    run_name: str,
    seed: int | None,
    semantic_config: SemanticConsistencyObjectiveConfig | None = None,
    progress=None,
    precision_policy_name: str = "auto",
) -> ArticulatorTrainingResult:
    """Train from frame tensor surfaces without materializing source payload objects."""

    if not isinstance(train_surface, ModelDataSurface) or not isinstance(
        validation_surface,
        ModelDataSurface,
    ):
        raise ArticulatorAwareError("train_surface and validation_surface must be ModelDataSurface.")
    if not isinstance(config, ArticulatorAwareConfig):
        raise ArticulatorAwareError("config must be an ArticulatorAwareConfig.")
    if semantic_config is not None:
        raise ArticulatorAwareError(
            "semantic_consistency training from articulator surfaces requires a temporal "
            "window surface and is not accepted on frame-unit surfaces."
        )
    train_sources = _surface_source_metadata_by_index(train_source_surface)
    validation_sources = _surface_source_metadata_by_index(validation_source_surface)
    if not train_sources or not validation_sources:
        raise ArticulatorAwareError("surface training requires non-empty train and validation sources.")
    _validate_surface_source_lengths(train_sources, config=config)
    _validate_surface_source_lengths(validation_sources, config=config)
    run_seed = config.training.seed if seed is None else seed
    if run_seed is None:
        run_seed = 0
    _set_seed(run_seed)
    device = _resolve_device(config.training.device)
    precision_policy = resolve_precision_policy(precision_policy_name, device)
    text_encoder = build_text_encoder(config.text_encoder)
    text_dim = text_encoder.output_dim
    train_embeddings = _surface_text_embeddings(
        text_encoder,
        train_sources,
        progress=progress,
        operation="encode_train_text",
        label="articulator encode train text",
    )
    validation_embeddings = _surface_text_embeddings(
        text_encoder,
        validation_sources,
        progress=progress,
        operation="encode_validation_text",
        label="articulator encode validation text",
    )
    if config.structure_variant.architecture == ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL:
        model = ArticulatorFactorizedTemporalModel(
            partition_policy=partition_policy,
            text_embedding_dim=text_dim,
            hidden_dim=config.structure_variant.hidden_dim,
            fusion_hidden_dim=int(config.structure_variant.fusion_hidden_dim),
            temporal_layers=config.structure_variant.temporal_layers,
            dropout=config.structure_variant.dropout,
            max_positions=config.length.max_positions,
        ).to(device)
    else:
        length_predictor = ArticulatorLengthPredictor(
            text_embedding_dim=text_dim,
            hidden_dim=config.structure_variant.hidden_dim,
        )
        model = ChannelFusionPoseModel(
            partition_policy=partition_policy,
            text_embedding_dim=text_dim,
            hidden_dim=config.structure_variant.hidden_dim,
            dropout=config.structure_variant.dropout,
            max_positions=config.length.max_positions,
            length_predictor=length_predictor,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    metrics: list[TrainingMetricRecord] = []
    global_step = 0
    final_train_loss = 0.0
    batch_size = max(1, int(config.training.frame_batch_size or config.training.batch_size))
    train_total = surface_batch_count(
        unit_count=train_surface.metadata.unit_count,
        batch_size=batch_size,
    )
    reader_runtime_trace: dict[str, object] = {}
    for epoch in range(config.training.max_epochs):
        model.train()
        losses: list[float] = []
        with _progress_task(
            progress,
            operation=f"train_epoch_{epoch + 1}",
            label=f"articulator train epoch {epoch + 1}/{config.training.max_epochs}",
            unit="batch",
            total=train_total,
            total_semantics="articulator frame-surface batches in epoch",
            allowed_counters=("loss",),
        ) as task:
            if isinstance(model, ArticulatorFactorizedTemporalModel):
                iterator = _surface_sequence_batches(
                    train_surface,
                    sources=train_sources,
                    embeddings=train_embeddings,
                    partition_policy=partition_policy,
                    batch_size=max(1, int(config.training.batch_size)),
                    device=device,
                )
            else:
                iterator = _surface_frame_batches(
                    train_surface,
                    sources=train_sources,
                    embeddings=train_embeddings,
                    partition_policy=partition_policy,
                    batch_size=batch_size,
                    device=device,
                    num_workers=config.training.num_workers,
                    runtime_trace=reader_runtime_trace,
                )
            for batch in iterator:
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision_policy):
                    if isinstance(model, ArticulatorFactorizedTemporalModel):
                        output = model(batch["text_embeddings"], batch["positions"])
                        reconstruction_loss, _ = compute_channel_weighted_sequence_reconstruction_loss_torch(
                            predicted=output.full_pose_values,
                            target=batch["target_values"],
                            channel_masks=batch["channel_masks"],
                            weighting=weighting,
                        )
                        length_loss = compute_articulator_length_loss(
                            predicted_lengths=output.predicted_lengths,
                            target_lengths=batch["source_lengths"],
                        )
                    else:
                        output = model(batch["text_embeddings"], batch["positions"])
                        reconstruction_loss, _ = compute_channel_weighted_reconstruction_loss_torch(
                            predicted=output.full_pose_values,
                            target=batch["target_values"],
                            channel_masks=batch["channel_masks"],
                            weighting=weighting,
                        )
                        unique_indices = torch.unique(batch["source_index"].long())
                        source_text = torch.stack(
                            [
                                torch.as_tensor(
                                    train_embeddings[int(index.item())],
                                    dtype=torch.float32,
                                    device=device,
                                )
                                for index in unique_indices
                            ]
                        )
                        target_lengths = torch.as_tensor(
                            [
                                float(train_sources[int(index.item())]["frame_count"])
                                for index in unique_indices
                            ],
                            dtype=torch.float32,
                            device=device,
                        )
                        length_loss = compute_articulator_length_loss(
                            predicted_lengths=model.length_predictor(source_text),
                            target_lengths=target_lengths,
                        )
                    total_loss = reconstruction_loss + float(config.length.loss_weight) * length_loss
                if not torch.isfinite(total_loss):
                    raise ArticulatorAwareError("surface training total loss is not finite.")
                total_loss.backward()
                optimizer.step()
                loss_value = float(total_loss.detach().cpu().item())
                losses.append(loss_value)
                metrics.append(_metric("train", "total_loss", loss_value, epoch, global_step))
                task.advance(1, counters={"loss": f"{loss_value:.6g}"})
                global_step += 1
        final_train_loss = _finite_mean(losses, "final_train_loss")
    train_channel, train_length, train_records = _evaluate_surface(
        model,
        train_surface,
        sources=train_sources,
        embeddings=train_embeddings,
        partition_policy=partition_policy,
        weighting=weighting,
        batch_size=batch_size,
        device=device,
        precision_policy=precision_policy,
        split=SampleSplit.TRAIN,
        progress=progress,
        operation="evaluate_train",
        label="articulator evaluate train surface",
        num_workers=config.training.num_workers,
    )
    val_channel, val_length, val_records = _evaluate_surface(
        model,
        validation_surface,
        sources=validation_sources,
        embeddings=validation_embeddings,
        partition_policy=partition_policy,
        weighting=weighting,
        batch_size=batch_size,
        device=device,
        precision_policy=precision_policy,
        split=SampleSplit.VAL,
        progress=progress,
        operation="evaluate_validation",
        label="articulator evaluate validation surface",
        num_workers=config.training.num_workers,
    )
    metrics.extend(
        (
            _metric("train", "channel_weighted_loss", train_channel, config.training.max_epochs - 1, global_step),
            _metric("train", "length_loss", train_length, config.training.max_epochs - 1, global_step),
            _metric("val", config.checkpoints.selection_metric, val_channel, config.training.max_epochs - 1, global_step),
            _metric("val", "validation_length_loss", val_length, config.training.max_epochs - 1, global_step),
        )
    )
    train_diagnostics, train_summary = _surface_diagnostics(
        train_records,
        sources=train_sources,
        split=SampleSplit.TRAIN,
        partition_policy=partition_policy,
    )
    val_diagnostics, val_summary = _surface_diagnostics(
        val_records,
        sources=validation_sources,
        split=SampleSplit.VAL,
        partition_policy=partition_policy,
    )
    return _write_articulator_training_artifacts(
        config=config,
        partition_policy=partition_policy,
        mask_config=mask_config,
        weighting=weighting,
        output_root=output_root,
        run_name=run_name,
        run_seed=run_seed,
        model=model,
        text_dim=text_dim,
        text_encoder=text_encoder,
        metrics=metrics,
        train_sources=(),
        validation_sources=(),
        train_records=train_records,
        val_records=val_records,
        final_train_loss=final_train_loss,
        val_channel=val_channel,
        val_length=val_length,
        global_step=global_step,
        semantic_config=semantic_config,
        validation_auxiliary={},
        train_diagnostics=train_diagnostics,
        train_summary=train_summary,
        val_diagnostics=val_diagnostics,
        val_summary=val_summary,
        runtime_trace=reader_runtime_trace,
    )


def _surface_sources(surface: ModelDataSurface) -> dict[str, ArticulatorSourceSample]:
    path = surface.root / "sources.jsonl"
    if not path.is_file():
        raise ArticulatorAwareError("articulator source surface is missing sources.jsonl.")
    sources: dict[str, ArticulatorSourceSample] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ArticulatorAwareError("articulator source surface rows must be objects.")
            source = _source_from_surface_record(record)
            sources[source.sample_id] = source
    return sources


def _surface_source_metadata_by_index(surface: ModelDataSurface) -> dict[int, dict[str, object]]:
    path = surface.root / "sources.jsonl"
    if not path.is_file():
        raise ArticulatorAwareError("articulator source surface is missing sources.jsonl.")
    sources: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ArticulatorAwareError(
                    f"articulator source surface row {line_number} must be an object."
                )
            index = int(record["source_index"])
            sources[index] = {
                "source_index": index,
                "sample_id": str(record["sample_id"]),
                "source_sentence_name": str(record["source_sentence_name"]),
                "text": str(record["text"]),
                "split": str(record["split"]),
                "frame_count": int(record["frame_count"]),
            }
    return sources


def _validate_surface_source_lengths(
    sources: Mapping[int, Mapping[str, object]],
    *,
    config: ArticulatorAwareConfig,
) -> None:
    for source in sources.values():
        frame_count = int(source["frame_count"])
        if frame_count > config.length.max_positions:
            raise ArticulatorAwareError(
                f"sample_id={source['sample_id']!r} frame_count {frame_count} exceeds "
                f"length.max_positions {config.length.max_positions}."
            )


def _surface_text_embeddings(
    text_encoder,
    sources: Mapping[int, Mapping[str, object]],
    *,
    progress,
    operation: str,
    label: str,
) -> dict[int, np.ndarray]:
    ordered = [sources[index] for index in sorted(sources)]
    embeddings: dict[int, np.ndarray] = {}
    with _progress_task(
        progress,
        operation=operation,
        label=label,
        unit="sample",
        total=len(ordered),
        total_semantics="surface source text embeddings encoded",
    ) as task:
        encoded = text_encoder.encode_batch(
            [str(source["text"]) for source in ordered],
            sentence_names=[str(source["source_sentence_name"]) for source in ordered],
        )
        task.advance(len(ordered))
    for source, output in zip(ordered, encoded, strict=True):
        embeddings[int(source["source_index"])] = np.asarray(
            output.pooled_embedding,
            dtype=np.float32,
        ).copy()
    return embeddings


def _surface_frame_batches(
    surface: ModelDataSurface,
    *,
    sources: Mapping[int, Mapping[str, object]],
    embeddings: Mapping[int, np.ndarray],
    partition_policy: ArticulatorChannelPartitionPolicy,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    runtime_trace: dict[str, object] | None = None,
):
    del sources
    reader = ModelDataSurfaceReader(surface)
    for batch in reader.iter_batches(
        batch_size=batch_size,
        num_workers=num_workers,
        runtime_trace=runtime_trace,
    ):
        source_index = batch["source_index"].long()
        text = torch.stack(
            [
                torch.as_tensor(
                    embeddings[int(index.item())],
                    dtype=torch.float32,
                    device=device,
                )
                for index in source_index
            ]
        )
        yield {
            "text_embeddings": text,
            "positions": batch["frame_index"].long().to(device),
            "target_values": batch["target_values"].float().to(device),
            "channel_masks": _surface_channel_masks(
                batch,
                partition_policy=partition_policy,
                device=device,
            ),
            "source_index": source_index.to(device),
        }


def _surface_sequence_batches(
    surface: ModelDataSurface,
    *,
    sources: Mapping[int, Mapping[str, object]],
    embeddings: Mapping[int, np.ndarray],
    partition_policy: ArticulatorChannelPartitionPolicy,
    batch_size: int,
    device: torch.device,
):
    reader = ModelDataSurfaceReader(surface)
    for shard in reader.iter_shards():
        source_indices = torch.unique(shard["source_index"].long(), sorted=True)
        grouped: list[dict[str, torch.Tensor | int]] = []
        for source_index in source_indices:
            mask = shard["source_index"].long() == source_index
            order = torch.argsort(shard["frame_index"][mask].long())
            grouped.append(
                {
                    "source_index": int(source_index.item()),
                    "target_values": shard["target_values"][mask][order].float(),
                    "positions": shard["frame_index"][mask][order].long(),
                    **{
                        f"{channel.value}_mask": shard[f"{channel.value}_mask"][mask][order].bool()
                        for channel in partition_policy.primary_channels
                        if f"{channel.value}_mask" in shard
                    },
                }
            )
        for start in range(0, len(grouped), batch_size):
            yield _surface_sequence_batch(
                grouped[start : start + batch_size],
                sources=sources,
                embeddings=embeddings,
                partition_policy=partition_policy,
                device=device,
            )


def _surface_sequence_batch(
    grouped,
    *,
    sources: Mapping[int, Mapping[str, object]],
    embeddings: Mapping[int, np.ndarray],
    partition_policy: ArticulatorChannelPartitionPolicy,
    device: torch.device,
) -> dict[str, object]:
    max_frames = max(int(item["target_values"].shape[0]) for item in grouped)
    feature_dim = partition_policy.layout.total_feature_dim
    target = torch.zeros((len(grouped), max_frames, feature_dim), dtype=torch.float32)
    positions = torch.zeros((len(grouped), max_frames), dtype=torch.long)
    frame_mask = torch.zeros((len(grouped), max_frames), dtype=torch.bool)
    channel_masks = {
        channel: torch.zeros((len(grouped), max_frames, feature_dim), dtype=torch.bool)
        for channel in partition_policy.primary_channels
    }
    text_rows = []
    lengths = []
    for row, item in enumerate(grouped):
        source_index = int(item["source_index"])
        values = item["target_values"]
        count = int(values.shape[0])
        target[row, :count, :] = values
        positions[row, :count] = item["positions"]
        frame_mask[row, :count] = True
        text_rows.append(
            torch.as_tensor(embeddings[source_index], dtype=torch.float32)
        )
        lengths.append(float(sources[source_index]["frame_count"]))
        for channel in partition_policy.primary_channels:
            key = f"{channel.value}_mask"
            mask = item.get(key)
            if isinstance(mask, torch.Tensor):
                channel_masks[channel][row, :count, :] = mask
            else:
                channel_masks[channel][row, :count, :] = True
    return {
        "text_embeddings": torch.stack(text_rows).to(device),
        "positions": positions.to(device),
        "target_values": target.to(device),
        "frame_mask": frame_mask.to(device),
        "channel_masks": {
            channel: mask.to(device) for channel, mask in channel_masks.items()
        },
        "source_lengths": torch.as_tensor(lengths, dtype=torch.float32, device=device),
    }


def _surface_channel_masks(
    batch: Mapping[str, object],
    *,
    partition_policy: ArticulatorChannelPartitionPolicy,
    device: torch.device,
) -> dict[PoseChannel, torch.Tensor]:
    fallback = batch["validity_mask"].bool()
    return {
        channel: (
            batch[f"{channel.value}_mask"].bool()
            if f"{channel.value}_mask" in batch
            else fallback
        ).to(device)
        for channel in partition_policy.primary_channels
    }


def _evaluate_surface(
    model,
    surface: ModelDataSurface,
    *,
    sources: Mapping[int, Mapping[str, object]],
    embeddings: Mapping[int, np.ndarray],
    partition_policy: ArticulatorChannelPartitionPolicy,
    weighting: ChannelLossWeightingPolicy,
    batch_size: int,
    device: torch.device,
    precision_policy: ResolvedPrecisionPolicy,
    split: SampleSplit,
    progress,
    operation: str,
    label: str,
    num_workers: int = 0,
) -> tuple[float, float, tuple[ChannelLossRecord, ...]]:
    model.eval()
    channel_sums: dict[tuple[int, PoseChannel], float] = {}
    channel_counts: dict[tuple[int, PoseChannel], int] = {}
    length_losses: list[float] = []
    batch_losses: list[float] = []
    total = surface_batch_count(unit_count=surface.metadata.unit_count, batch_size=batch_size)
    with _progress_task(
        progress,
        operation=operation,
        label=label,
        unit="batch",
        total=total,
        total_semantics="articulator frame-surface batches evaluated",
    ) as task:
        with torch.no_grad():
            for batch in _surface_frame_batches(
                surface,
                sources=sources,
                embeddings=embeddings,
                partition_policy=partition_policy,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
            ):
                with autocast_context(precision_policy):
                    output = model(batch["text_embeddings"], batch["positions"])
                    loss, _ = compute_channel_weighted_reconstruction_loss_torch(
                        predicted=output.full_pose_values,
                        target=batch["target_values"],
                        channel_masks=batch["channel_masks"],
                        weighting=weighting,
                    )
                batch_losses.append(float(loss.cpu().item()))
                source_index = batch["source_index"].detach().cpu()
                predicted = output.full_pose_values.detach().cpu()
                target = batch["target_values"].detach().cpu()
                for channel in partition_policy.primary_channels:
                    mask = batch["channel_masks"][channel].detach().cpu().bool()
                    for row, idx in enumerate(source_index.tolist()):
                        valid = mask[row]
                        count = int(torch.count_nonzero(valid).item())
                        key = (int(idx), channel)
                        if count == 0:
                            channel_counts.setdefault(key, 0)
                            channel_sums.setdefault(key, 0.0)
                            continue
                        value = float(torch.sum(torch.square(predicted[row][valid] - target[row][valid])).item())
                        channel_sums[key] = channel_sums.get(key, 0.0) + value
                        channel_counts[key] = channel_counts.get(key, 0) + count
                task.advance()
    for source_index, source in sources.items():
        text = torch.as_tensor(
            embeddings[source_index],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        target_length = torch.as_tensor(
            [float(source["frame_count"])],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad(), autocast_context(precision_policy):
            length_losses.append(
                float(
                    compute_articulator_length_loss(
                        predicted_lengths=model.length_predictor(text),
                        target_lengths=target_length,
                    ).cpu().item()
                )
            )
    records: list[ChannelLossRecord] = []
    for source_index, source in sources.items():
        for channel in partition_policy.primary_channels:
            key = (source_index, channel)
            count = channel_counts.get(key, 0)
            if count <= 0:
                records.append(
                    ChannelLossRecord(
                        schema_version="t2sp-channel-loss-record-v1",
                        sample_id=str(source["sample_id"]),
                        split=split,
                        channel=channel,
                        loss_name="masked_mse",
                        value=None,
                        valid_observation_count=0,
                        skipped=True,
                        reason=f"{channel.value}: no valid observations; channel loss skipped.",
                    )
                )
                continue
            records.append(
                ChannelLossRecord(
                    schema_version="t2sp-channel-loss-record-v1",
                    sample_id=str(source["sample_id"]),
                    split=split,
                    channel=channel,
                    loss_name="masked_mse",
                    value=channel_sums[key] / count,
                    valid_observation_count=count,
                    skipped=False,
                    reason=None,
                )
            )
    return (
        _finite_mean(batch_losses, "channel_weighted_loss"),
        _finite_mean(length_losses, "length_loss"),
        tuple(records),
    )


def _surface_diagnostics(
    records: Sequence[ChannelLossRecord],
    *,
    sources: Mapping[int, Mapping[str, object]],
    split: SampleSplit,
    partition_policy: ArticulatorChannelPartitionPolicy,
):
    loss_by_key = {
        (record.sample_id, record.channel): record
        for record in records
    }
    diagnostics = []
    for source in sources.values():
        sample_id = str(source["sample_id"])
        for channel in partition_policy.primary_channels:
            loss = loss_by_key.get((sample_id, channel))
            diagnostics.append(
                ChannelDiagnosticRecord(
                    schema_version=CHANNEL_DIAGNOSTIC_RECORD_SCHEMA_VERSION,
                    sample_id=sample_id,
                    split=split,
                    channel=channel,
                    valid_fraction=0.0 if loss is not None and loss.skipped else 1.0,
                    weighted_loss=None if loss is None or loss.skipped else loss.value,
                    masked_l1_hint=None,
                    skipped=loss is None or loss.skipped,
                    issues=() if loss is not None and not loss.skipped else (
                        f"{channel.value}: no valid observations; channel loss skipped.",
                    ),
                )
            )
    materialized = tuple(diagnostics)
    return materialized, aggregate_channel_diagnostics(materialized, split=split)


def _source_from_surface_record(record: Mapping[str, object]) -> ArticulatorSourceSample:
    from text_to_sign_production.modeling.backbones.bfh_vectorization import (
        default_bfh_tensor_layout,
    )

    frame_count = int(record["frame_count"])
    sample_id = str(record["sample_id"])
    layout = default_bfh_tensor_layout()
    vectorized = BfhVectorizedPose(
        layout=layout,
        values=np.zeros(
            (frame_count, layout.total_joint_count, layout.coordinate_dimensions),
            dtype=np.float32,
        ),
        validity_mask=np.ones((frame_count, layout.total_joint_count), dtype=np.bool_),
        frame_validity_mask=np.ones((frame_count,), dtype=np.bool_),
        confidence_values=np.ones((frame_count, layout.total_joint_count), dtype=np.float32),
        frame_count=frame_count,
        source_sample_id=sample_id,
    )
    return ArticulatorSourceSample(
        sample_id=sample_id,
        source_sentence_name=str(record["source_sentence_name"]),
        text=str(record["text"]),
        source_video_id=str(record["source_video_id"]),
        source_sentence_id=str(record["source_sentence_id"]),
        reference_payload_ref=str(record["reference_payload_ref"]),
        split=SampleSplit(str(record["split"])),
        frame_count=frame_count,
        vectorized_pose=vectorized,
    )


def _train_factorized_temporal_model(
    *,
    config: ArticulatorAwareConfig,
    train_samples: Sequence[ArticulatorSourceSample],
    validation_samples: Sequence[ArticulatorSourceSample],
    partition_policy: ArticulatorChannelPartitionPolicy,
    mask_config: ChannelMaskStrategyConfig,
    weighting: ChannelLossWeightingPolicy,
    output_root: Path,
    run_name: str,
    seed: int | None,
    semantic_config: SemanticConsistencyObjectiveConfig | None,
    progress=None,
    precision_policy_name: str = "auto",
) -> ArticulatorTrainingResult:
    train_sources = tuple(train_samples)
    validation_sources = tuple(validation_samples)
    if not train_sources or not validation_sources:
        raise ArticulatorAwareError("training requires non-empty train and validation samples.")
    _validate_source_lengths((*train_sources, *validation_sources), config=config)
    run_seed = config.training.seed if seed is None else seed
    if run_seed is None:
        run_seed = 0
    _set_seed(run_seed)
    device = _resolve_device(config.training.device)
    precision_policy = resolve_precision_policy(precision_policy_name, device)
    text_encoder = build_text_encoder(config.text_encoder)
    text_dim = text_encoder.output_dim
    model = ArticulatorFactorizedTemporalModel(
        partition_policy=partition_policy,
        text_embedding_dim=text_dim,
        hidden_dim=config.structure_variant.hidden_dim,
        fusion_hidden_dim=int(config.structure_variant.fusion_hidden_dim),
        temporal_layers=config.structure_variant.temporal_layers,
        dropout=config.structure_variant.dropout,
        max_positions=config.length.max_positions,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    embedding_sources = (*train_sources, *validation_sources)
    with _progress_task(
        progress,
        operation="encode_text",
        label="articulator encode train and validation text",
        unit="sample",
        total=len(embedding_sources),
        total_semantics="train and validation text embeddings encoded",
    ) as task:
        encoded_text = text_encoder.encode_batch(
            [source.text for source in embedding_sources],
            sentence_names=[source.source_sentence_name for source in embedding_sources],
        )
        task.advance(len(embedding_sources))
    embeddings = {
        source.sample_id: np.asarray(output.pooled_embedding, dtype=np.float32)
        for source, output in zip(embedding_sources, encoded_text, strict=True)
    }
    metrics: list[TrainingMetricRecord] = []
    global_step = 0
    final_train_loss = 0.0
    for epoch in range(config.training.max_epochs):
        model.train()
        losses: list[float] = []
        source_batches = _batches(train_sources, config.training.batch_size)
        with _progress_task(
            progress,
            operation=f"train_epoch_{epoch + 1}",
            label=f"articulator train epoch {epoch + 1}/{config.training.max_epochs}",
            unit="batch",
            total=len(source_batches),
            total_semantics="articulator source batches in epoch",
            allowed_counters=("loss",),
        ) as task:
            for source_batch in source_batches:
                batch = _sequence_batch(
                    source_batch,
                    embeddings=embeddings,
                    partition_policy=partition_policy,
                    mask_config=mask_config,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision_policy):
                    output = model(batch.text_embeddings, batch.positions)
                    reconstruction_loss, _ = compute_channel_weighted_sequence_reconstruction_loss_torch(
                        predicted=output.full_pose_values,
                        target=batch.target_values,
                        channel_masks=batch.channel_masks,
                        weighting=weighting,
                    )
                    velocity_loss = compute_channel_velocity_loss_torch(
                        predicted=output.full_pose_values,
                        target=batch.target_values,
                        channel_masks=batch.channel_masks,
                        weighting=weighting,
                    )
                    hand_coordination_loss = compute_hand_motion_coordination_loss_torch(
                        predicted=output.full_pose_values,
                        target=batch.target_values,
                        channel_masks=batch.channel_masks,
                    )
                    cross_channel_loss = compute_cross_channel_temporal_consistency_loss_torch(
                        predicted=output.full_pose_values,
                        target=batch.target_values,
                        channel_masks=batch.channel_masks,
                        channels=partition_policy.primary_channels,
                    )
                    length_loss = compute_articulator_length_loss(
                        predicted_lengths=output.predicted_lengths,
                        target_lengths=batch.source_lengths,
                    )
                    total_loss = (
                        reconstruction_loss
                        + weighting.velocity_weight * velocity_loss
                        + weighting.symmetry_weight * hand_coordination_loss
                        + weighting.cross_channel_consistency_weight * cross_channel_loss
                        + float(config.length.loss_weight) * length_loss
                    )
                    semantic_result = None
                    if semantic_config is not None:
                        pose_validity = _combined_pose_validity_mask(
                            batch.channel_masks,
                            output.full_pose_values.shape,
                        )
                        semantic_result = compute_semantic_training_loss(
                            text_embeddings=batch.text_embeddings,
                            pose_values=output.full_pose_values,
                            pose_validity_mask=pose_validity,
                            frame_mask=batch.frame_mask,
                            layout=partition_policy.layout,
                            config=semantic_config,
                        )
                        total_loss = total_loss + semantic_result.weighted_semantic_loss
                if not torch.isfinite(total_loss):
                    raise ArticulatorAwareError("factorized temporal training total loss is not finite.")
                total_loss.backward()
                optimizer.step()
                values = {
                    "total_loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "velocity_loss": velocity_loss,
                    "hand_coordination_loss": hand_coordination_loss,
                    "cross_channel_consistency_loss": cross_channel_loss,
                    "length_loss": length_loss,
                }
                if semantic_result is not None:
                    values.update(
                        {
                            "semantic_loss": semantic_result.semantic_loss,
                            "semantic_weighted_loss": semantic_result.weighted_semantic_loss,
                            "semantic_cosine_similarity": semantic_result.cosine_similarity,
                            "semantic_cosine_distance": semantic_result.cosine_distance,
                        }
                    )
                for name, tensor in values.items():
                    metrics.append(
                        _metric(
                            "train",
                            name,
                            float(tensor.detach().cpu().item()),
                            epoch,
                            global_step,
                        )
                    )
                loss_value = float(total_loss.detach().cpu().item())
                losses.append(loss_value)
                task.advance(1, counters={"loss": f"{loss_value:.6g}"})
                global_step += 1
        final_train_loss = _finite_mean(losses, "final_train_loss")
    train_channel, train_length, train_records = _evaluate_sources(
        model,
        train_sources,
        embeddings=embeddings,
        partition_policy=partition_policy,
        weighting=weighting,
        device=device,
        precision_policy=precision_policy,
    )
    val_channel, val_length, val_records = _evaluate_sources(
        model,
        validation_sources,
        embeddings=embeddings,
        partition_policy=partition_policy,
        weighting=weighting,
        device=device,
        precision_policy=precision_policy,
    )
    metrics.extend(
        (
            _metric("train", "channel_weighted_loss", train_channel, config.training.max_epochs - 1, global_step),
            _metric("train", "length_loss", train_length, config.training.max_epochs - 1, global_step),
            _metric("val", config.checkpoints.selection_metric, val_channel, config.training.max_epochs - 1, global_step),
            _metric("val", "validation_length_loss", val_length, config.training.max_epochs - 1, global_step),
        )
    )
    return _write_articulator_training_artifacts(
        config=config,
        partition_policy=partition_policy,
        mask_config=mask_config,
        weighting=weighting,
        output_root=output_root,
        run_name=run_name,
        run_seed=run_seed,
        model=model,
        text_dim=text_dim,
        text_encoder=text_encoder,
        metrics=metrics,
        train_sources=train_sources,
        validation_sources=validation_sources,
        train_records=train_records,
        val_records=val_records,
        final_train_loss=final_train_loss,
        val_channel=val_channel,
        val_length=val_length,
        global_step=global_step,
        semantic_config=semantic_config,
        validation_auxiliary=validation_auxiliary,
    )


def _write_articulator_training_artifacts(
    *,
    config,
    partition_policy,
    mask_config,
    weighting,
    output_root,
    run_name,
    run_seed,
    model,
    text_dim,
    text_encoder,
    metrics,
    train_sources,
    validation_sources,
    train_records,
    val_records,
    final_train_loss,
    val_channel,
    val_length,
    global_step,
    semantic_config=None,
    validation_auxiliary=None,
    train_diagnostics=None,
    train_summary=None,
    val_diagnostics=None,
    val_summary=None,
    runtime_trace=None,
) -> ArticulatorTrainingResult:
    model_run_root = Path(output_root)
    checkpoint_root = model_run_root / "checkpoints"
    artifact_root = model_run_root / "intermediates" / "articulator"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ARTICULATOR_CHECKPOINT_SCHEMA_VERSION,
        "model_key": config.identity.model_key,
        "run_name": run_name,
        "model_state_dict": model.state_dict(),
        "length_predictor_state_dict": model.length_predictor.state_dict(),
        "model_config": build_articulator_model_config_payload(
            config,
            partition_policy,
            text_embedding_dim=text_dim,
            text_encoder_metadata_payload=text_encoder_metadata(text_encoder),
        ),
        "length_predictor_config": build_articulator_length_config_payload(
            config,
            text_embedding_dim=text_dim,
        ),
        "provider_config": config.to_dict(),
        "text_embedding_dim": text_dim,
        "text_encoder_config": config.text_encoder.to_dict(),
        "text_encoder_metadata": text_encoder_metadata(text_encoder),
        "run_seed": run_seed,
        "selection_metric": config.checkpoints.selection_metric,
        "selection_metric_value": val_channel,
    }
    semantic_extra = _semantic_checkpoint_extra(semantic_config)
    auxiliary_extra = {
        "validation_auxiliary_metrics": dict(validation_auxiliary or {}),
        "validation_auxiliary_metrics_unavailable": (
            [] if validation_auxiliary else [
                "no factorized validation auxiliary observations were available"
            ]
        ),
    }
    payload.update(semantic_extra)
    payload.update(auxiliary_extra)
    best_checkpoint = checkpoint_root / "best.pt"
    last_checkpoint = checkpoint_root / "last.pt"
    torch.save(payload, best_checkpoint)
    torch.save(payload, last_checkpoint)
    model_config_path = artifact_root / "model_config.json"
    length_config_path = artifact_root / "length_predictor_config.json"
    write_json(model_config_path, payload["model_config"])
    write_json(length_config_path, payload["length_predictor_config"])
    metrics_path = artifact_root / "training_metrics.jsonl"
    write_training_metric_records_jsonl(metrics_path, metrics)
    train_loss_path = artifact_root / "channel_loss_records_train.jsonl"
    val_loss_path = artifact_root / "channel_loss_records_val.jsonl"
    write_channel_loss_records_jsonl(train_loss_path, train_records)
    write_channel_loss_records_jsonl(val_loss_path, val_records)
    if train_diagnostics is None or train_summary is None:
        train_diagnostics, train_summary = _diagnostics(
            train_sources, train_records, split=SampleSplit.TRAIN, partition_policy=partition_policy, mask_config=mask_config
        )
    if val_diagnostics is None or val_summary is None:
        val_diagnostics, val_summary = _diagnostics(
            validation_sources, val_records, split=SampleSplit.VAL, partition_policy=partition_policy, mask_config=mask_config
        )
    train_diag_path = artifact_root / "channel_diagnostics_train.jsonl"
    val_diag_path = artifact_root / "channel_diagnostics_val.jsonl"
    train_summary_path = artifact_root / "channel_diagnostics_summary_train.json"
    val_summary_path = artifact_root / "channel_diagnostics_summary_val.json"
    write_channel_diagnostic_records_jsonl(train_diag_path, train_diagnostics)
    write_channel_diagnostic_records_jsonl(val_diag_path, val_diagnostics)
    write_channel_diagnostic_summary_json(train_summary_path, train_summary)
    write_channel_diagnostic_summary_json(val_summary_path, val_summary)
    best_metadata = artifact_root / "best_metadata.json"
    last_metadata = artifact_root / "last_metadata.json"
    for role, checkpoint, metadata in (
        (CheckpointRole.BEST, best_checkpoint, best_metadata),
        (CheckpointRole.LAST, last_checkpoint, last_metadata),
    ):
        write_checkpoint_metadata_json(
            metadata,
            ModelCheckpointMetadata(
                schema_version=CHECKPOINT_METADATA_SCHEMA_VERSION,
                model_key=config.identity.model_key,
                run_name=run_name,
                role=role,
                stage_name="train_structure_aware",
                checkpoint_path=checkpoint,
                epoch=config.training.max_epochs - 1,
                global_step=global_step,
                selection_metric=config.checkpoints.selection_metric,
                selection_metric_value=val_channel,
                lower_is_better=config.checkpoints.lower_is_better,
                created_at_utc=datetime.now(UTC).isoformat(),
                extra={
                    "validation_length_loss": val_length,
                    "length_policy": config.length.policy,
                    "model_config_path": str(model_config_path),
                    "length_predictor_config_path": str(length_config_path),
                    **semantic_extra,
                    **auxiliary_extra,
                },
            ),
        )
    return ArticulatorTrainingResult(
        best_checkpoint_path=best_checkpoint,
        last_checkpoint_path=last_checkpoint,
        best_metadata_path=best_metadata,
        last_metadata_path=last_metadata,
        training_metrics_path=metrics_path,
        train_channel_loss_records_path=train_loss_path,
        validation_channel_loss_records_path=val_loss_path,
        train_channel_diagnostics_path=train_diag_path,
        validation_channel_diagnostics_path=val_diag_path,
        train_channel_diagnostics_summary_path=train_summary_path,
        validation_channel_diagnostics_summary_path=val_summary_path,
        model_config_path=model_config_path,
        length_predictor_config_path=length_config_path,
        final_train_loss=final_train_loss,
        validation_channel_weighted_loss=val_channel,
        validation_length_loss=val_length,
        completed_epoch=config.training.max_epochs - 1,
        global_step=global_step,
        runtime_trace=dict(runtime_trace or {}),
    )


def _sequence_batch(
    sources,
    *,
    embeddings,
    partition_policy,
    mask_config,
    device,
) -> ArticulatorSequenceBatch:
    max_frames = max(source.frame_count for source in sources)
    feature_dim = partition_policy.layout.total_feature_dim
    target = np.zeros((len(sources), max_frames, feature_dim), dtype=np.float32)
    frame_mask = np.zeros((len(sources), max_frames), dtype=np.bool_)
    channel_masks = {
        channel: np.zeros((len(sources), max_frames, feature_dim), dtype=np.bool_)
        for channel in partition_policy.primary_channels
    }
    positions = np.zeros((len(sources), max_frames), dtype=np.int64)
    for row, source in enumerate(sources):
        length = source.frame_count
        positions[row, :length] = np.arange(length, dtype=np.int64)
        frame_mask[row, :length] = True
        target[row, :length, :] = flatten_bfh_vectorized_pose(source.vectorized_pose)
        masks = build_channel_loss_masks(
            vectorized=source.vectorized_pose,
            policy=partition_policy,
            config=mask_config,
        )
        for channel in partition_policy.primary_channels:
            channel_masks[channel][row, :length, :] = masks[channel]
    return ArticulatorSequenceBatch(
        text_embeddings=torch.as_tensor(
            np.stack([embeddings[source.sample_id] for source in sources]),
            dtype=torch.float32,
            device=device,
        ),
        positions=torch.as_tensor(positions, dtype=torch.long, device=device),
        target_values=torch.as_tensor(target, dtype=torch.float32, device=device),
        frame_mask=torch.as_tensor(frame_mask, dtype=torch.bool, device=device),
        channel_masks={
            channel: torch.as_tensor(mask, dtype=torch.bool, device=device)
            for channel, mask in channel_masks.items()
        },
        source_lengths=torch.as_tensor(
            [float(source.frame_count) for source in sources],
            dtype=torch.float32,
            device=device,
        ),
        sample_ids=tuple(source.sample_id for source in sources),
    )


def _combined_pose_validity_mask(
    channel_masks: Mapping[PoseChannel, torch.Tensor],
    shape: torch.Size,
) -> torch.Tensor:
    masks = tuple(mask for mask in channel_masks.values())
    if not masks or any(mask.shape != shape for mask in masks):
        raise ArticulatorAwareError("semantic channel masks must align with predicted sequence.")
    combined = masks[0].clone()
    for mask in masks[1:]:
        combined = combined | mask
    return combined


def _evaluate_sequence_auxiliary_metrics(
    model,
    sources,
    *,
    embeddings,
    partition_policy,
    mask_config,
    weighting,
    device,
    precision_policy: ResolvedPrecisionPolicy | None = None,
) -> dict[str, float]:
    if not isinstance(model, ArticulatorFactorizedTemporalModel):
        return {}
    values: dict[str, list[float]] = {
        "validation_velocity_loss": [],
        "validation_hand_coordination_loss": [],
        "validation_cross_channel_consistency_loss": [],
    }
    model.eval()
    resolved_precision = precision_policy or resolve_precision_policy("auto", torch.device(device))
    with torch.no_grad():
        for source_batch in _batches(tuple(sources), max(1, min(len(sources), 8))):
            batch = _sequence_batch(
                source_batch,
                embeddings=embeddings,
                partition_policy=partition_policy,
                mask_config=mask_config,
                device=device,
            )
            with autocast_context(resolved_precision):
                output = model(batch.text_embeddings, batch.positions)
            values["validation_velocity_loss"].append(
                float(
                    compute_channel_velocity_loss_torch(
                        predicted=output.full_pose_values,
                        target=batch.target_values,
                        channel_masks=batch.channel_masks,
                        weighting=weighting,
                    ).cpu().item()
                )
            )
            values["validation_hand_coordination_loss"].append(
                float(
                    compute_hand_motion_coordination_loss_torch(
                        predicted=output.full_pose_values,
                        target=batch.target_values,
                        channel_masks=batch.channel_masks,
                    ).cpu().item()
                )
            )
            values["validation_cross_channel_consistency_loss"].append(
                float(
                    compute_cross_channel_temporal_consistency_loss_torch(
                        predicted=output.full_pose_values,
                        target=batch.target_values,
                        channel_masks=batch.channel_masks,
                        channels=partition_policy.primary_channels,
                    ).cpu().item()
                )
            )
    return {
        name: _finite_mean(metric_values, name)
        for name, metric_values in values.items()
        if metric_values
    }


def _semantic_checkpoint_extra(
    semantic_config: SemanticConsistencyObjectiveConfig | None,
) -> dict[str, object]:
    if semantic_config is None:
        return {"semantic_consistency_enabled": False}
    return {
        "semantic_consistency_enabled": True,
        "semantic_training_objective_schema_version": "t2sp-semantic-objective-config-v2",
        "semantic_config_hash": hashlib.sha256(
            json.dumps(semantic_config.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "semantic_loss_weight": semantic_config.training_objective.alignment.loss_weight,
        "semantic_pose_projection_backend": (
            semantic_config.training_objective.pose_projection.backend
        ),
        "semantic_text_source": semantic_config.training_objective.text_source,
        "semantic_requires_ablation": semantic_config.training_objective.requires_ablation,
    }


def _batch_tensors(batch, *, embeddings, device):
    text = torch.as_tensor(
        np.stack([embeddings[item.source.sample_id] for item in batch]),
        dtype=torch.float32,
        device=device,
    )
    positions = torch.as_tensor([item.frame_index for item in batch], dtype=torch.long, device=device)
    target = torch.as_tensor(np.stack([item.target_values for item in batch]), dtype=torch.float32, device=device)
    masks = {
        channel: torch.as_tensor(
            np.stack([item.channel_masks[channel] for item in batch]),
            dtype=torch.bool,
            device=device,
        )
        for channel in batch[0].channel_masks
    }
    return text, positions, target, masks


def _backprop_frame_chunk_reconstruction(
    *,
    model,
    frames,
    frame_batch_size,
    embeddings,
    device,
    weighting,
    precision_policy: ResolvedPrecisionPolicy | None = None,
) -> float:
    """Backpropagate one source-batch reconstruction loss in capped frame chunks."""

    valid_counts = {
        channel: sum(
            int(np.count_nonzero(frame.channel_masks[channel]))
            for frame in frames
        )
        for channel in weighting.normalized_channel_weights
    }
    active = tuple(channel for channel, count in valid_counts.items() if count > 0)
    if not active:
        raise ArticulatorAwareError("no valid channels remain for torch reconstruction loss.")
    weight_sum = sum(weighting.normalized_channel_weights[channel] for channel in active)
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        raise ArticulatorAwareError("valid channel weights must have a positive finite sum.")
    total_value = 0.0
    resolved_precision = precision_policy or resolve_precision_policy("auto", torch.device(device))
    for frame_batch in _batches(frames, frame_batch_size):
        text, positions, target, masks = _batch_tensors(
            frame_batch,
            embeddings=embeddings,
            device=device,
        )
        with autocast_context(resolved_precision):
            predicted = model(text, positions).full_pose_values
            terms: list[torch.Tensor] = []
            for channel in active:
                valid = masks[channel]
                if torch.any(valid):
                    terms.append(
                        torch.sum(torch.square(predicted[valid] - target[valid]))
                        * (weighting.normalized_channel_weights[channel] / weight_sum)
                        / valid_counts[channel]
                    )
            if not terms:
                continue
            chunk_loss = sum(terms)
        if not torch.isfinite(chunk_loss):
            raise ArticulatorAwareError("articulator training reconstruction loss is not finite.")
        chunk_loss.backward()
        total_value += float(chunk_loss.detach().cpu().item())
    if not math.isfinite(total_value):
        raise ArticulatorAwareError("articulator training reconstruction loss is not finite.")
    return total_value


def _evaluate_sources(
    model,
    sources,
    *,
    embeddings,
    partition_policy,
    weighting,
    device,
    precision_policy: ResolvedPrecisionPolicy | None = None,
):
    model.eval()
    channel_losses: list[float] = []
    length_losses: list[float] = []
    records: list[ChannelLossRecord] = []
    resolved_precision = precision_policy or resolve_precision_policy("auto", torch.device(device))
    with torch.no_grad():
        for source in sources:
            text = torch.tensor(embeddings[source.sample_id], dtype=torch.float32, device=device).unsqueeze(0)
            positions = torch.arange(source.frame_count, dtype=torch.long, device=device)
            with autocast_context(resolved_precision):
                if isinstance(model, ArticulatorFactorizedTemporalModel):
                    output = model(text, positions.unsqueeze(0))
                    predicted_flat = output.full_pose_values.squeeze(0).float().detach().cpu().numpy()
                else:
                    output = model(text.repeat(source.frame_count, 1), positions)
                    predicted_flat = output.full_pose_values.float().detach().cpu().numpy()
            target = source.vectorized_pose
            predicted = BfhVectorizedPose(
                layout=target.layout,
                values=unflatten_bfh_pose_values(predicted_flat, layout=target.layout, frame_count=source.frame_count),
                validity_mask=target.validity_mask,
                frame_validity_mask=target.frame_validity_mask,
                confidence_values=target.confidence_values,
                frame_count=source.frame_count,
                source_sample_id=source.sample_id,
            )
            total, source_records = compute_channel_weighted_reconstruction_loss_numpy(
                predicted=predicted,
                target=target,
                partition_policy=partition_policy,
                weighting=weighting,
                split=source.split,
                sample_id=source.sample_id,
            )
            channel_losses.append(total)
            records.extend(source_records)
            target_length = torch.tensor([float(source.frame_count)], dtype=torch.float32, device=device)
            with autocast_context(resolved_precision):
                predicted_length = model.length_predictor(text)
                length_losses.append(
                    float(
                        compute_articulator_length_loss(
                            predicted_lengths=predicted_length,
                            target_lengths=target_length,
                        ).cpu().item()
                    )
                )
    return (
        _finite_mean(channel_losses, "channel_weighted_loss"),
        _finite_mean(length_losses, "length_loss"),
        tuple(records),
    )


def _diagnostics(sources, records, *, split, partition_policy, mask_config):
    summaries = tuple(
        summarize_channel_masks(
            vectorized=source.vectorized_pose,
            split=split,
            sample_id=source.sample_id,
            policy=partition_policy,
            config=mask_config,
        )
        for source in sources
    )
    diagnostics = build_channel_diagnostic_records(mask_summaries=summaries, loss_records=records)
    return diagnostics, aggregate_channel_diagnostics(diagnostics, split=split)


def build_articulator_model_config_payload(
    config,
    policy,
    *,
    text_embedding_dim: int | None = None,
    text_encoder_metadata_payload: Mapping[str, object] | None = None,
):
    """Build the model support config embedded in and restored beside checkpoints."""

    resolved_text_dim = _resolve_articulator_text_embedding_dim(
        config,
        text_embedding_dim=text_embedding_dim,
    )
    return {
        "schema_version": ARTICULATOR_MODEL_CONFIG_SCHEMA_VERSION,
        "architecture": config.structure_variant.architecture,
        "fusion": config.structure_variant.fusion,
        "temporal_layers": config.structure_variant.temporal_layers,
        "fusion_hidden_dim": config.structure_variant.fusion_hidden_dim,
        "text_encoder": config.text_encoder.to_dict(),
        "text_encoder_metadata": dict(text_encoder_metadata_payload or {}),
        "text_embedding_dim": resolved_text_dim,
        "hidden_dim": config.structure_variant.hidden_dim,
        "dropout": float(config.structure_variant.dropout),
        "max_positions": config.length.max_positions,
        "total_feature_dim": policy.layout.total_feature_dim,
        "primary_channels": [channel.value for channel in policy.primary_channels],
        "loss_weighting": {
            "velocity_weight": float(config.loss_weighting.velocity_weight),
            "symmetry_weight": float(config.loss_weighting.symmetry_weight),
            "cross_channel_consistency_weight": float(
                config.loss_weighting.cross_channel_consistency_weight
            ),
        },
    }


def build_articulator_length_config_payload(config, *, text_embedding_dim: int | None = None):
    """Build the length support config embedded in and restored beside checkpoints."""

    resolved_text_dim = _resolve_articulator_text_embedding_dim(
        config,
        text_embedding_dim=text_embedding_dim,
    )
    return {
        "schema_version": ARTICULATOR_LENGTH_CONFIG_SCHEMA_VERSION,
        **config.length.to_dict(),
        "text_embedding_dim": resolved_text_dim,
        "hidden_dim": config.structure_variant.hidden_dim,
    }


def _resolve_articulator_text_embedding_dim(config, *, text_embedding_dim: int | None) -> int:
    if text_embedding_dim is not None:
        resolved = int(text_embedding_dim)
        if resolved <= 0:
            raise ArticulatorAwareError("text_embedding_dim must be positive.")
        return resolved
    if (
        config.text_encoder.backend == "deterministic_hash"
        and config.text_encoder.embedding_dim is not None
    ):
        return int(config.text_encoder.embedding_dim)
    raise ArticulatorAwareError(
        "text_embedding_dim must be provided for non-legacy or dynamic text encoders."
    )


def _validate_source_lengths(sources, *, config):
    for source in sources:
        if source.frame_count > config.length.max_positions:
            raise ArticulatorAwareError(
                f"sample_id={source.sample_id!r} frame_count {source.frame_count} exceeds "
                f"length.max_positions {config.length.max_positions}."
            )


def _batches(values, batch_size):
    return tuple(tuple(values[start : start + batch_size]) for start in range(0, len(values), batch_size))


class _NoOpTask:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exception, traceback) -> None:
        return

    def advance(self, count: int = 1, *, counters=None) -> None:
        del count, counters


def _progress_task(progress, **kwargs):
    if progress is None:
        return _NoOpTask()
    return progress.task(**kwargs)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _finite_mean(values: Sequence[float], name: str) -> float:
    if not values:
        raise ArticulatorAwareError(f"{name} requires at least one value.")
    value = float(np.mean(values))
    if not math.isfinite(value):
        raise ArticulatorAwareError(f"{name} is not finite.")
    return value


def _metric(split: str, name: str, value: float, epoch: int, step: int) -> TrainingMetricRecord:
    return TrainingMetricRecord(
        schema_version=TRAINING_METRIC_SCHEMA_VERSION,
        split=split,
        metric_name=name,
        value=value,
        epoch=epoch,
        global_step=step,
        stage_name="train_structure_aware",
        created_at_utc=datetime.now(UTC).isoformat(),
        extra={},
    )


__all__ = [
    "ARTICULATOR_CHECKPOINT_SCHEMA_VERSION",
    "ARTICULATOR_LENGTH_CONFIG_SCHEMA_VERSION",
    "ARTICULATOR_MODEL_CONFIG_SCHEMA_VERSION",
    "ArticulatorSequenceBatch",
    "ArticulatorTrainingResult",
    "build_articulator_length_config_payload",
    "build_articulator_model_config_payload",
    "compute_articulator_length_loss",
    "compute_channel_weighted_reconstruction_loss_torch",
    "train_articulator_model",
    "train_articulator_model_from_surfaces",
]
