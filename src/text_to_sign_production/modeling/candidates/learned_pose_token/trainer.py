"""Training scaffold for the learned pose-token representation foundation."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates.learned_pose_token.codebook import (
    CODEBOOK_METADATA_SCHEMA_VERSION,
    PoseCodebookMetadata,
    compute_codebook_stability,
    write_codebook_json,
)
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    BfhStandardizationStats,
    compute_bfh_standardization_stats,
    read_bfh_standardization_stats_json,
    write_bfh_standardization_stats_json,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    LearnedPoseTokenConfig,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.dataset import (
    PoseTokenTrainingSample,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.io import (
    write_pose_token_manifest_jsonl,
    write_pose_token_sequence_npz,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.model import (
    build_learned_pose_tokenizer,
    compute_tokenizer_losses,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.reports import (
    build_reconstruction_report_markdown,
    build_token_manifest_summary,
    build_tokenizer_design_report,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.tokens import (
    POSE_TOKEN_MANIFEST_SCHEMA_VERSION,
    POSE_TOKEN_SCHEMA_VERSION,
    PoseTokenManifestEntry,
    PoseTokenSequence,
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
    autocast_context,
    resolve_precision_policy,
)
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurface,
    ModelDataSurfaceReader,
    surface_batch_count,
)


@dataclass(frozen=True, slots=True)
class PoseTokenizerTrainingResult:
    tokenizer_checkpoint_path: Path
    tokenizer_checkpoint_metadata_path: Path
    codebook_metadata_path: Path
    standardization_stats_path: Path
    train_token_manifest_path: Path
    validation_token_manifest_path: Path
    validation_reconstruction_summary_manifest_path: Path
    train_codebook_stability_path: Path
    validation_codebook_stability_path: Path
    training_metrics_path: Path
    reconstruction_summary_path: Path
    tokenizer_design_report_path: Path
    token_manifest_summary_path: Path
    runtime_trace: dict[str, object] | None = None

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name == "runtime_trace":
                continue
            path = Path(getattr(self, field_name))
            if not path.is_file():
                raise LearnedPoseTokenError(f"{field_name} does not exist: {path}")
            object.__setattr__(self, field_name, path)

def _require_surface_reader_runtime_trace(
    runtime_trace: Mapping[str, object] | None,
    *,
    expected_num_workers: int,
    context: str,
) -> None:
    if expected_num_workers <= 0:
        return

    if not isinstance(runtime_trace, Mapping):
        raise LearnedPoseTokenError(
            f"{context} expected surface reader runtime trace because "
            f"num_workers={expected_num_workers}, but runtime_trace is missing."
        )

    missing = [
        key
        for key in (
            "surface_reader_num_workers_used",
            "surface_reader_worker_mode",
        )
        if runtime_trace.get(key) is None
    ]
    if missing:
        raise LearnedPoseTokenError(
            f"{context} expected surface reader runtime trace keys {missing} "
            f"because num_workers={expected_num_workers}. This means the stage did "
            "not execute through the trace-aware ModelDataSurfaceReader.iter_batches path."
        )

    observed_num_workers = runtime_trace.get("surface_reader_num_workers_used")
    if observed_num_workers != expected_num_workers:
        raise LearnedPoseTokenError(
            f"{context} surface reader num_workers mismatch: "
            f"expected {expected_num_workers}, observed {observed_num_workers}."
        )

    worker_mode = runtime_trace.get("surface_reader_worker_mode")
    expected_mode = "multiprocess" if expected_num_workers > 0 else "single_process"
    if worker_mode != expected_mode:
        raise LearnedPoseTokenError(
            f"{context} surface reader worker mode mismatch: "
            f"expected {expected_mode!r}, observed {worker_mode!r}."
        )


def train_pose_tokenizer(
    *,
    config: LearnedPoseTokenConfig,
    train_samples: Sequence[PoseTokenTrainingSample] | None = None,
    validation_samples: Sequence[PoseTokenTrainingSample] | None = None,
    train_surface: ModelDataSurface | ModelDataSurfaceReader | None = None,
    validation_surface: ModelDataSurface | ModelDataSurfaceReader | None = None,
    output_root: Path,
    run_name: str,
    seed: int | None,
    progress=None,
    precision_policy_name: str = "auto",
) -> PoseTokenizerTrainingResult:
    """Train a minimal token-unit VQ tokenizer and write real foundation artifacts."""

    if not isinstance(config, LearnedPoseTokenConfig):
        raise LearnedPoseTokenError("config must be a LearnedPoseTokenConfig.")
    _require_text(run_name, "run_name")
    if train_surface is not None or validation_surface is not None:
        if train_surface is None or validation_surface is None:
            raise LearnedPoseTokenError(
                "train_surface and validation_surface must be provided together."
            )

        result = _train_pose_tokenizer_from_surfaces(
            config=config,
            train_surface=train_surface,
            validation_surface=validation_surface,
            output_root=output_root,
            run_name=run_name,
            seed=seed,
            progress=progress,
            precision_policy_name=precision_policy_name,
        )

        _require_surface_reader_runtime_trace(
            result.runtime_trace,
            expected_num_workers=config.training.num_workers,
            context="train_pose_tokenizer(surface)",
        )

        return result
    if train_samples is None or validation_samples is None:
        raise LearnedPoseTokenError("train_samples and validation_samples are required without surfaces.")
    train = tuple(train_samples)
    validation = tuple(validation_samples)
    if not train:
        raise LearnedPoseTokenError("train_samples must be non-empty.")
    if not validation:
        raise LearnedPoseTokenError("validation_samples must be non-empty.")
    _validate_samples(train, "train_samples")
    _validate_samples(validation, "validation_samples")
    effective_seed = config.training.seed if seed is None else seed
    _set_seed(effective_seed)

    root = Path(output_root)
    checkpoint_dir = root / "checkpoints"
    token_root = root / "tokens"
    report_root = root / "reports"
    for directory in (checkpoint_dir, token_root, report_root):
        directory.mkdir(parents=True, exist_ok=True)

    stats = _standardization_stats(
        train,
        missing_observation_policy=(
            config.representation.standardization_missing_observation_policy
        ),
    )
    stats_path = root / "standardization_stats.json"
    write_bfh_standardization_stats_json(stats_path, stats)

    input_dim = int(train[0].values.shape[0])
    model = build_learned_pose_tokenizer(input_dim=input_dim, config=config)
    device = _resolve_device(config.training.device)
    precision_policy = resolve_precision_policy(precision_policy_name, device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    metrics: list[TrainingMetricRecord] = []
    global_step = 0
    trainable = tuple(sample for sample in train if bool(sample.validity_mask.any()))
    if not trainable:
        raise LearnedPoseTokenError("train_samples contain no supervised valid pose features.")
    tokenizer_batch_size = config.training.effective_tokenizer_batch_size
    train_batches = _batches(trainable, tokenizer_batch_size)
    for epoch in range(config.training.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        with _task_or_noop(
            progress,
            operation=f"tokenizer_train_epoch_{epoch + 1}",
            label=(
                "learned_pose_token tokenizer train epoch "
                f"{epoch + 1}/{config.training.max_epochs}"
            ),
            unit="batch",
            total=len(train_batches),
            total_semantics="tokenizer training batches in epoch",
            allowed_counters=("loss",),
        ) as task:
            for batch in train_batches:
                values, mask, real_frame_mask = _batch_tensors(batch, stats=stats, device=device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision_policy):
                    output = model(values)
                    losses = compute_tokenizer_losses(
                        output=output,
                        target=values,
                        validity_mask=mask,
                        commitment_weight=config.tokenizer.commitment_weight,
                        temporal_granularity=config.tokenizer.temporal_granularity,
                        window_size=config.tokenizer.window_size,
                        feature_dim=stats.layout.total_feature_dim,
                        real_frame_mask=real_frame_mask,
                        velocity_loss_weight=config.tokenizer.velocity_loss_weight,
                    )
                losses["total_loss"].float().backward()
                optimizer.step()
                loss_value = float(losses["total_loss"].detach().cpu().item())
                epoch_losses.append(loss_value)
                metrics.append(
                    _metric(
                        split="train",
                        name="total_loss",
                        value=loss_value,
                        epoch=epoch,
                        global_step=global_step,
                    )
                )
                task.advance(1, counters={"loss": f"{loss_value:.6g}"})
                global_step += 1
        metrics.append(
            _metric(
                split="train",
                name="epoch_mean_total_loss",
                value=float(np.mean(epoch_losses)),
                epoch=epoch,
                global_step=global_step,
            )
        )

    validation_loss = _validation_loss(
        model,
        validation,
        stats=stats,
        device=device,
        config=config,
        precision_policy=precision_policy,
        progress=progress,
    )
    metrics.append(
        _metric(
            split="val",
            name="mean_total_loss",
            value=validation_loss,
            epoch=config.training.max_epochs - 1,
            global_step=global_step,
        )
    )

    checkpoint_path = checkpoint_dir / "tokenizer.pt"
    torch.save(
        {
            "schema_version": "t2sp-learned-pose-tokenizer-checkpoint-v1",
            "model_key": config.identity.model_key,
            "run_name": run_name,
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "standardization_stats_path": str(stats_path),
        },
        checkpoint_path,
    )
    tokenizer_checkpoint_metadata_path = checkpoint_dir / "tokenizer.metadata.json"
    write_checkpoint_metadata_json(
        tokenizer_checkpoint_metadata_path,
        ModelCheckpointMetadata(
            schema_version=CHECKPOINT_METADATA_SCHEMA_VERSION,
            model_key=config.identity.model_key,
            run_name=run_name,
            role=CheckpointRole.LAST,
            stage_name="train_pose_tokenizer",
            checkpoint_path=checkpoint_path,
            epoch=config.training.max_epochs - 1,
            global_step=global_step,
            selection_metric="val.mean_total_loss",
            selection_metric_value=validation_loss,
            lower_is_better=True,
            created_at_utc=_now(),
            extra={"foundation_only": True},
        ),
    )

    train_sequences = _token_sequences(
        model,
        train,
        stats=stats,
        device=device,
        config=config,
        split_name="train",
        token_root=token_root,
        progress=progress,
    )
    validation_sequences = _token_sequences(
        model,
        validation,
        stats=stats,
        device=device,
        config=config,
        split_name="val",
        token_root=token_root,
        progress=progress,
    )
    train_manifest_path = token_root / "train_manifest.jsonl"
    validation_manifest_path = token_root / "validation_manifest.jsonl"
    train_entries = _manifest_entries(train_sequences, token_root=token_root)
    validation_entries = _manifest_entries(validation_sequences, token_root=token_root)
    write_pose_token_manifest_jsonl(train_manifest_path, train_entries)
    write_pose_token_manifest_jsonl(validation_manifest_path, validation_entries)

    codebook_metadata_path = root / "codebook_metadata.json"
    write_codebook_json(
        codebook_metadata_path,
        PoseCodebookMetadata(
            schema_version=CODEBOOK_METADATA_SCHEMA_VERSION,
            codebook_size=config.codebook.size,
            embedding_dim=config.codebook.embedding_dim,
            initialization=config.codebook.initialization,
            checkpoint_path=checkpoint_path,
            created_at_utc=_now(),
            training_sample_count=len(train),
            training_frame_count=sum(
                source.frame_count for source in _unique_training_sources(train)
            ),
            training_token_unit_count=len(train),
            temporal_granularity=config.tokenizer.temporal_granularity,
            window_size=config.tokenizer.window_size,
            stride=config.tokenizer.stride,
            notes=(
                "Foundation tokenizer codebook metadata only; no provider registration is implied.",
                "training_frame_count is the source BFH frame count; training_token_unit_count is the tokenizer unit count.",
            ),
        ),
    )
    train_stability_path = report_root / "train_codebook_stability.json"
    validation_stability_path = report_root / "validation_codebook_stability.json"
    write_codebook_json(
        train_stability_path,
        compute_codebook_stability(
            train_sequences,
            codebook_size=config.codebook.size,
            dead_code_threshold=config.codebook.dead_code_threshold,
            collapse_perplexity_threshold=config.codebook.collapse_perplexity_threshold,
            split=train[0].split,
        ),
    )
    write_codebook_json(
        validation_stability_path,
        compute_codebook_stability(
            validation_sequences,
            codebook_size=config.codebook.size,
            dead_code_threshold=config.codebook.dead_code_threshold,
            collapse_perplexity_threshold=config.codebook.collapse_perplexity_threshold,
            split=validation[0].split,
        ),
    )

    metrics_path = root / "training_metrics.jsonl"
    write_training_metric_records_jsonl(metrics_path, metrics)
    reconstruction_summary_manifest_path = report_root / "validation_reconstruction_summary_manifest.jsonl"
    _write_reconstruction_summary_manifest(
        reconstruction_summary_manifest_path,
        validation_entries,
        mean_loss=validation_loss,
    )
    reconstruction_summary_path = report_root / "reconstruction_summary.md"
    reconstruction_summary_path.write_text(
        build_reconstruction_report_markdown(
            run_name=run_name,
            split="val",
            sample_count=len(validation_entries),
            mean_reconstruction_loss=validation_loss,
            temporal_granularity=config.tokenizer.temporal_granularity,
            window_size=config.tokenizer.window_size,
            stride=config.tokenizer.stride,
        ),
        encoding="utf-8",
    )
    tokenizer_design_report_path = report_root / "tokenizer_design_report.json"
    token_manifest_summary_path = report_root / "token_manifest_summary.json"
    _write_json(tokenizer_design_report_path, build_tokenizer_design_report(config))
    _write_json(token_manifest_summary_path, build_token_manifest_summary(validation_entries))

    return PoseTokenizerTrainingResult(
        tokenizer_checkpoint_path=checkpoint_path,
        tokenizer_checkpoint_metadata_path=tokenizer_checkpoint_metadata_path,
        codebook_metadata_path=codebook_metadata_path,
        standardization_stats_path=stats_path,
        train_token_manifest_path=train_manifest_path,
        validation_token_manifest_path=validation_manifest_path,
        validation_reconstruction_summary_manifest_path=reconstruction_summary_manifest_path,
        train_codebook_stability_path=train_stability_path,
        validation_codebook_stability_path=validation_stability_path,
        training_metrics_path=metrics_path,
        reconstruction_summary_path=reconstruction_summary_path,
        tokenizer_design_report_path=tokenizer_design_report_path,
        token_manifest_summary_path=token_manifest_summary_path,
    )


def _train_pose_tokenizer_from_surfaces(
    *,
    config: LearnedPoseTokenConfig,
    train_surface: ModelDataSurface | ModelDataSurfaceReader,
    validation_surface: ModelDataSurface | ModelDataSurfaceReader,
    output_root: Path,
    run_name: str,
    seed: int | None,
    progress=None,
    precision_policy_name: str = "auto",
) -> PoseTokenizerTrainingResult:
    train_reader = _surface_reader(train_surface)
    validation_reader = _surface_reader(validation_surface)
    if train_reader.metadata.unit_count <= 0 or validation_reader.metadata.unit_count <= 0:
        raise LearnedPoseTokenError("pose-token surfaces must be non-empty.")
    effective_seed = config.training.seed if seed is None else seed
    _set_seed(effective_seed)

    root = Path(output_root)
    checkpoint_dir = root / "checkpoints"
    token_root = root / "tokens"
    report_root = root / "reports"
    for directory in (checkpoint_dir, token_root, report_root):
        directory.mkdir(parents=True, exist_ok=True)

    source_stats_path = train_reader.surface.root / "standardization_stats.json"
    if not source_stats_path.is_file():
        raise LearnedPoseTokenError(
            "train_surface is missing standardization_stats.json; rebuild the tensor surface."
        )
    stats = read_bfh_standardization_stats_json(source_stats_path)
    stats_path = root / "standardization_stats.json"
    write_bfh_standardization_stats_json(stats_path, stats)

    input_dim = _surface_feature_dim(train_reader)
    model = build_learned_pose_tokenizer(input_dim=input_dim, config=config)
    device = _resolve_device(config.training.device)
    precision_policy = resolve_precision_policy(precision_policy_name, device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    metrics: list[TrainingMetricRecord] = []
    global_step = 0
    tokenizer_batch_size = config.training.effective_tokenizer_batch_size
    train_batch_total = surface_batch_count(
        unit_count=train_reader.metadata.unit_count,
        batch_size=tokenizer_batch_size,
    )
    reader_runtime_trace: dict[str, object] = {}
    for epoch in range(config.training.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        with _task_or_noop(
            progress,
            operation=f"tokenizer_train_epoch_{epoch + 1}",
            label=(
                "learned_pose_token tokenizer train epoch "
                f"{epoch + 1}/{config.training.max_epochs}"
            ),
            unit="batch",
            total=train_batch_total,
            total_semantics="tokenizer training batches in epoch",
            allowed_counters=("loss",),
        ) as task:
            for batch in train_reader.iter_batches(
                batch_size=tokenizer_batch_size,
                shuffle_shards=True,
                shuffle_units=True,
                num_workers=config.training.num_workers,
                runtime_trace=reader_runtime_trace,
            ):
                values, mask, real_frame_mask = _surface_batch_tensors(batch, stats=stats, device=device)
                if not bool(mask.any().detach().cpu().item()):
                    continue
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision_policy):
                    output = model(values)
                    losses = compute_tokenizer_losses(
                        output=output,
                        target=values,
                        validity_mask=mask,
                        commitment_weight=config.tokenizer.commitment_weight,
                        temporal_granularity=config.tokenizer.temporal_granularity,
                        window_size=config.tokenizer.window_size,
                        feature_dim=stats.layout.total_feature_dim,
                        real_frame_mask=real_frame_mask,
                        velocity_loss_weight=config.tokenizer.velocity_loss_weight,
                    )
                losses["total_loss"].float().backward()
                optimizer.step()
                loss_value = float(losses["total_loss"].detach().cpu().item())
                epoch_losses.append(loss_value)
                metrics.append(
                    _metric(
                        split="train",
                        name="total_loss",
                        value=loss_value,
                        epoch=epoch,
                        global_step=global_step,
                    )
                )
                task.advance(1, counters={"loss": f"{loss_value:.6g}"})
                global_step += 1
        if not epoch_losses:
            raise LearnedPoseTokenError("train_surface contains no supervised valid pose features.")
        metrics.append(
            _metric(
                split="train",
                name="epoch_mean_total_loss",
                value=float(np.mean(epoch_losses)),
                epoch=epoch,
                global_step=global_step,
            )
        )

    validation_loss = _surface_validation_loss(
        model,
        validation_reader,
        stats=stats,
        device=device,
        config=config,
        precision_policy=precision_policy,
        progress=progress,
    )
    metrics.append(
        _metric(
            split="val",
            name="mean_total_loss",
            value=validation_loss,
            epoch=config.training.max_epochs - 1,
            global_step=global_step,
        )
    )

    checkpoint_path = checkpoint_dir / "tokenizer.pt"
    torch.save(
        {
            "schema_version": "t2sp-learned-pose-tokenizer-checkpoint-v1",
            "model_key": config.identity.model_key,
            "run_name": run_name,
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "standardization_stats_path": str(stats_path),
        },
        checkpoint_path,
    )
    tokenizer_checkpoint_metadata_path = checkpoint_dir / "tokenizer.metadata.json"
    write_checkpoint_metadata_json(
        tokenizer_checkpoint_metadata_path,
        ModelCheckpointMetadata(
            schema_version=CHECKPOINT_METADATA_SCHEMA_VERSION,
            model_key=config.identity.model_key,
            run_name=run_name,
            role=CheckpointRole.LAST,
            stage_name="train_pose_tokenizer",
            checkpoint_path=checkpoint_path,
            epoch=config.training.max_epochs - 1,
            global_step=global_step,
            selection_metric="val.mean_total_loss",
            selection_metric_value=validation_loss,
            lower_is_better=True,
            created_at_utc=_now(),
            extra={"foundation_only": True},
        ),
    )

    train_sequences = _surface_token_sequences(
        model,
        train_reader,
        stats=stats,
        device=device,
        config=config,
        split_name="train",
        token_root=token_root,
        progress=progress,
    )
    validation_sequences = _surface_token_sequences(
        model,
        validation_reader,
        stats=stats,
        device=device,
        config=config,
        split_name="val",
        token_root=token_root,
        progress=progress,
    )
    train_manifest_path = token_root / "train_manifest.jsonl"
    validation_manifest_path = token_root / "validation_manifest.jsonl"
    train_entries = _manifest_entries(train_sequences, token_root=token_root)
    validation_entries = _manifest_entries(validation_sequences, token_root=token_root)
    write_pose_token_manifest_jsonl(train_manifest_path, train_entries)
    write_pose_token_manifest_jsonl(validation_manifest_path, validation_entries)

    codebook_metadata_path = root / "codebook_metadata.json"
    train_sources = _surface_sources(train_reader)
    validation_sources = _surface_sources(validation_reader)
    write_codebook_json(
        codebook_metadata_path,
        PoseCodebookMetadata(
            schema_version=CODEBOOK_METADATA_SCHEMA_VERSION,
            codebook_size=config.codebook.size,
            embedding_dim=config.codebook.embedding_dim,
            initialization=config.codebook.initialization,
            checkpoint_path=checkpoint_path,
            created_at_utc=_now(),
            training_sample_count=train_reader.metadata.unit_count,
            training_frame_count=sum(int(source["frame_count"]) for source in train_sources),
            training_token_unit_count=train_reader.metadata.unit_count,
            temporal_granularity=config.tokenizer.temporal_granularity,
            window_size=config.tokenizer.window_size,
            stride=config.tokenizer.stride,
            notes=(
                "Foundation tokenizer codebook metadata only; no provider registration is implied.",
                "training_frame_count is the source BFH frame count; training_token_unit_count is the tokenizer unit count.",
            ),
        ),
    )
    train_stability_path = report_root / "train_codebook_stability.json"
    validation_stability_path = report_root / "validation_codebook_stability.json"
    write_codebook_json(
        train_stability_path,
        compute_codebook_stability(
            train_sequences,
            codebook_size=config.codebook.size,
            dead_code_threshold=config.codebook.dead_code_threshold,
            collapse_perplexity_threshold=config.codebook.collapse_perplexity_threshold,
            split=SampleSplit.TRAIN,
        ),
    )
    write_codebook_json(
        validation_stability_path,
        compute_codebook_stability(
            validation_sequences,
            codebook_size=config.codebook.size,
            dead_code_threshold=config.codebook.dead_code_threshold,
            collapse_perplexity_threshold=config.codebook.collapse_perplexity_threshold,
            split=SampleSplit.VAL,
        ),
    )

    metrics_path = root / "training_metrics.jsonl"
    write_training_metric_records_jsonl(metrics_path, metrics)
    reconstruction_summary_manifest_path = report_root / "validation_reconstruction_summary_manifest.jsonl"
    _write_reconstruction_summary_manifest(
        reconstruction_summary_manifest_path,
        validation_entries,
        mean_loss=validation_loss,
    )
    reconstruction_summary_path = report_root / "reconstruction_summary.md"
    reconstruction_summary_path.write_text(
        build_reconstruction_report_markdown(
            run_name=run_name,
            split="val",
            sample_count=len(validation_entries),
            mean_reconstruction_loss=validation_loss,
            temporal_granularity=config.tokenizer.temporal_granularity,
            window_size=config.tokenizer.window_size,
            stride=config.tokenizer.stride,
        ),
        encoding="utf-8",
    )
    tokenizer_design_report_path = report_root / "tokenizer_design_report.json"
    token_manifest_summary_path = report_root / "token_manifest_summary.json"
    _write_json(tokenizer_design_report_path, build_tokenizer_design_report(config))
    _write_json(token_manifest_summary_path, build_token_manifest_summary(validation_entries))
    _require_surface_reader_runtime_trace(
        reader_runtime_trace,
        expected_num_workers=config.training.num_workers,
        context="_train_pose_tokenizer_from_surfaces",
    )
    return PoseTokenizerTrainingResult(
        tokenizer_checkpoint_path=checkpoint_path,
        tokenizer_checkpoint_metadata_path=tokenizer_checkpoint_metadata_path,
        codebook_metadata_path=codebook_metadata_path,
        standardization_stats_path=stats_path,
        train_token_manifest_path=train_manifest_path,
        validation_token_manifest_path=validation_manifest_path,
        validation_reconstruction_summary_manifest_path=reconstruction_summary_manifest_path,
        train_codebook_stability_path=train_stability_path,
        validation_codebook_stability_path=validation_stability_path,
        training_metrics_path=metrics_path,
        reconstruction_summary_path=reconstruction_summary_path,
        tokenizer_design_report_path=tokenizer_design_report_path,
        token_manifest_summary_path=token_manifest_summary_path,
        runtime_trace=dict(reader_runtime_trace),
    )


def _token_sequences(
    model: torch.nn.Module,
    samples: Sequence[PoseTokenTrainingSample],
    *,
    stats: BfhStandardizationStats,
    device: torch.device,
    config: LearnedPoseTokenConfig,
    split_name: str,
    token_root: Path,
    progress=None,
) -> tuple[PoseTokenSequence, ...]:
    groups: OrderedDict[str, list[PoseTokenTrainingSample]] = OrderedDict()
    for sample in samples:
        groups.setdefault(sample.sample_id, []).append(sample)
    sequences: list[PoseTokenSequence] = []
    model.eval()
    with torch.no_grad():
        with _task_or_noop(
            progress,
            operation=f"token_sequence_write_{split_name}",
            label=f"learned_pose_token token sequence write {split_name}",
            unit="sample",
            total=len(groups),
            total_semantics=f"{split_name} token sequences written",
            allowed_counters=("token_units",),
        ) as task:
            for sample_id, grouped in groups.items():
                ordered = sorted(grouped, key=lambda item: item.token_index)
                values, _mask, _real_frame_mask = _batch_tensors(
                    ordered,
                    stats=stats,
                    device=device,
                )
                output = model(values)
                sequence = PoseTokenSequence(
                    schema_version=POSE_TOKEN_SCHEMA_VERSION,
                    sample_id=sample_id,
                    source_sentence_name=ordered[0].source_sentence_name,
                    split=ordered[0].split,
                    token_ids=output.token_ids.detach().cpu().numpy().astype(np.int64),
                    frame_count=ordered[0].source.frame_count,
                    token_count=len(ordered),
                    codebook_size=config.codebook.size,
                    temporal_granularity=config.tokenizer.temporal_granularity,
                    window_size=config.tokenizer.window_size,
                    stride=config.tokenizer.stride,
                )
                path = token_root / split_name / f"{sample_id}.npz"
                write_pose_token_sequence_npz(path, sequence)
                sequences.append(sequence)
                task.advance(1, counters={"token_units": sequence.token_count})
    return tuple(sequences)


def _surface_token_sequences(
    model: torch.nn.Module,
    reader: ModelDataSurfaceReader,
    *,
    stats: BfhStandardizationStats,
    device: torch.device,
    config: LearnedPoseTokenConfig,
    split_name: str,
    token_root: Path,
    progress=None,
) -> tuple[PoseTokenSequence, ...]:
    sources = _surface_sources_by_index(reader)
    sequences: list[PoseTokenSequence] = []
    model.eval()
    with torch.no_grad():
        with _task_or_noop(
            progress,
            operation=f"token_sequence_write_{split_name}",
            label=f"learned_pose_token token sequence write {split_name}",
            unit="sample",
            total=reader.metadata.loaded_sample_count,
            total_semantics=f"{split_name} token sequences written",
            allowed_counters=("token_units",),
        ) as task:
            for shard in reader.iter_shards():
                source_index = shard.get("source_index")
                token_index = shard.get("token_index")
                if not isinstance(source_index, torch.Tensor) or not isinstance(token_index, torch.Tensor):
                    raise LearnedPoseTokenError("pose-token surface shard is missing source/token indices.")
                for raw_source_index in torch.unique(source_index, sorted=True).tolist():
                    mask = source_index == int(raw_source_index)
                    order = torch.argsort(token_index[mask])
                    subset = {
                        key: (value[mask][order] if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == source_index.shape[0] else value)
                        for key, value in shard.items()
                    }
                    values, _mask, _real_frame_mask = _surface_batch_tensors(
                        subset,
                        stats=stats,
                        device=device,
                    )
                    output = model(values)
                    source = sources[int(raw_source_index)]
                    sequence = PoseTokenSequence(
                        schema_version=POSE_TOKEN_SCHEMA_VERSION,
                        sample_id=str(source["sample_id"]),
                        source_sentence_name=str(source["source_sentence_name"]),
                        split=SampleSplit(str(source["split"])),
                        token_ids=output.token_ids.detach().cpu().numpy().astype(np.int64),
                        frame_count=int(source["frame_count"]),
                        token_count=int(values.shape[0]),
                        codebook_size=config.codebook.size,
                        temporal_granularity=config.tokenizer.temporal_granularity,
                        window_size=config.tokenizer.window_size,
                        stride=config.tokenizer.stride,
                    )
                    path = token_root / split_name / f"{sequence.sample_id}.npz"
                    write_pose_token_sequence_npz(path, sequence)
                    sequences.append(sequence)
                    task.advance(1, counters={"token_units": sequence.token_count})
    return tuple(sequences)


def _surface_validation_loss(
    model: torch.nn.Module,
    reader: ModelDataSurfaceReader,
    *,
    stats: BfhStandardizationStats,
    device: torch.device,
    config: LearnedPoseTokenConfig,
    precision_policy=None,
    progress=None,
) -> float:
    losses = []
    model.eval()
    batch_size = config.training.effective_tokenizer_batch_size
    with torch.no_grad():
        with _task_or_noop(
            progress,
            operation="tokenizer_validation",
            label="learned_pose_token tokenizer validation",
            unit="batch",
            total=surface_batch_count(unit_count=reader.metadata.unit_count, batch_size=batch_size),
            total_semantics="tokenizer validation batches",
            allowed_counters=("loss",),
        ) as task:
            for batch in reader.iter_batches(
                batch_size=batch_size,
                num_workers=config.training.num_workers,
            ):
                values, mask, real_frame_mask = _surface_batch_tensors(batch, stats=stats, device=device)
                if not bool(mask.any().detach().cpu().item()):
                    continue
                with autocast_context(
                    precision_policy or resolve_precision_policy("auto", device)
                ):
                    output = model(values)
                    result = compute_tokenizer_losses(
                        output=output,
                        target=values,
                        validity_mask=mask,
                        commitment_weight=config.tokenizer.commitment_weight,
                        temporal_granularity=config.tokenizer.temporal_granularity,
                        window_size=config.tokenizer.window_size,
                        feature_dim=stats.layout.total_feature_dim,
                        real_frame_mask=real_frame_mask,
                        velocity_loss_weight=config.tokenizer.velocity_loss_weight,
                    )
                loss = float(result["total_loss"].detach().cpu().item())
                losses.append(loss)
                task.advance(1, counters={"loss": f"{loss:.6g}"})
    if not losses:
        raise LearnedPoseTokenError("validation_surface contains no supervised valid pose features.")
    return float(np.mean(losses))


def _surface_batch_tensors(
    batch: dict[str, object],
    *,
    stats: BfhStandardizationStats,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = _required_tensor(batch, "values").float()
    masks = _required_tensor(batch, "validity_mask").bool()
    real_frame_mask = _required_tensor(batch, "real_frame_mask").bool()
    mean = torch.as_tensor(np.asarray(stats.mean, dtype=np.float32).reshape(-1))
    std = torch.as_tensor(np.asarray(stats.std, dtype=np.float32).reshape(-1))
    if values.shape[1] % mean.shape[0] != 0:
        raise LearnedPoseTokenError(
            "token-unit feature dimension must be a multiple of frame feature dimension."
        )
    repeat = values.shape[1] // mean.shape[0]
    mean_unit = mean.repeat(repeat)
    std_unit = std.repeat(repeat)
    normalized = torch.where(masks, (values - mean_unit.unsqueeze(0)) / std_unit.unsqueeze(0), torch.zeros_like(values))
    return normalized.to(device), masks.to(device), real_frame_mask.to(device)


def _surface_reader(surface: ModelDataSurface | ModelDataSurfaceReader) -> ModelDataSurfaceReader:
    if isinstance(surface, ModelDataSurfaceReader):
        return surface
    if isinstance(surface, ModelDataSurface):
        return ModelDataSurfaceReader(surface)
    raise LearnedPoseTokenError("surface must be ModelDataSurface or ModelDataSurfaceReader.")


def _surface_feature_dim(reader: ModelDataSurfaceReader) -> int:
    if reader.metadata.feature_dim is not None:
        return reader.metadata.feature_dim
    for shard in reader.iter_shards():
        values = shard.get("values")
        if isinstance(values, torch.Tensor) and values.ndim == 2:
            return int(values.shape[1])
    raise LearnedPoseTokenError("pose-token surface is missing values feature_dim.")


def _surface_sources(reader: ModelDataSurfaceReader) -> tuple[dict[str, object], ...]:
    path = reader.surface.root / "sources.jsonl"
    if not path.is_file():
        raise LearnedPoseTokenError("pose-token surface is missing sources.jsonl.")
    sources = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise LearnedPoseTokenError("sources.jsonl rows must be objects.")
                sources.append(record)
    return tuple(sources)


def _surface_sources_by_index(reader: ModelDataSurfaceReader) -> dict[int, dict[str, object]]:
    result = {}
    for record in _surface_sources(reader):
        source_index = record.get("source_index")
        if not isinstance(source_index, int) or isinstance(source_index, bool):
            raise LearnedPoseTokenError("sources.jsonl source_index must be an integer.")
        result[source_index] = record
    return result


def _required_tensor(batch: dict[str, object], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise LearnedPoseTokenError(f"surface batch is missing tensor field {name!r}.")
    return value


def _manifest_entries(
    sequences: Sequence[PoseTokenSequence],
    *,
    token_root: Path,
) -> tuple[PoseTokenManifestEntry, ...]:
    entries = []
    for sequence in sequences:
        token_path = token_root / sequence.split.value / f"{sequence.sample_id}.npz"
        entries.append(
            PoseTokenManifestEntry(
                schema_version=POSE_TOKEN_MANIFEST_SCHEMA_VERSION,
                sample_id=sequence.sample_id,
                source_sentence_name=sequence.source_sentence_name,
                split=sequence.split,
                token_path=token_path,
                token_count=sequence.token_count,
                frame_count=sequence.frame_count,
                codebook_size=sequence.codebook_size,
                temporal_granularity=sequence.temporal_granularity,
                window_size=sequence.window_size,
                stride=sequence.stride,
                issues=(),
            )
        )
    return tuple(entries)


def _standardization_stats(
    samples: Sequence[PoseTokenTrainingSample],
    *,
    missing_observation_policy: str = "raise",
) -> BfhStandardizationStats:
    sources = OrderedDict((sample.source.sample_id, sample.source.vectorized_pose) for sample in samples)
    try:
        return compute_bfh_standardization_stats(
            sources.values(),
            epsilon=1e-6,
            missing_observation_policy=missing_observation_policy,
        )
    except Exception as exc:
        raise LearnedPoseTokenError(
            "could not compute shared BFH standardization stats for pose-token training"
        ) from exc


def _unique_training_sources(
    samples: Sequence[PoseTokenTrainingSample],
):
    sources = OrderedDict()
    for sample in samples:
        sources.setdefault(sample.source.sample_id, sample.source)
    return tuple(sources.values())


def _batch_tensors(
    samples: Sequence[PoseTokenTrainingSample],
    *,
    stats: BfhStandardizationStats,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(-1)
    values = np.stack([sample.values for sample in samples]).astype(np.float32)
    masks = np.stack([sample.validity_mask for sample in samples]).astype(np.bool_)
    if values.shape[1] % mean.shape[0] != 0:
        raise LearnedPoseTokenError(
            "token-unit feature dimension must be a multiple of frame feature dimension."
        )
    repeat = values.shape[1] // mean.shape[0]
    mean_unit = np.tile(mean, repeat)
    std_unit = np.tile(std, repeat)
    normalized = np.where(masks, (values - mean_unit[None, :]) / std_unit[None, :], 0.0)
    real_frame_mask = np.stack(
        [np.asarray(sample.real_frame_mask, dtype=np.bool_) for sample in samples]
    )
    return (
        torch.as_tensor(normalized, dtype=torch.float32, device=device),
        torch.as_tensor(masks, dtype=torch.bool, device=device),
        torch.as_tensor(real_frame_mask, dtype=torch.bool, device=device),
    )


def _validation_loss(
    model: torch.nn.Module,
    samples: Sequence[PoseTokenTrainingSample],
    *,
    stats: BfhStandardizationStats,
    device: torch.device,
    config: LearnedPoseTokenConfig,
    precision_policy=None,
    progress=None,
) -> float:
    supervised = tuple(sample for sample in samples if bool(sample.validity_mask.any()))
    if not supervised:
        raise LearnedPoseTokenError("validation_samples contain no supervised valid pose features.")
    losses = []
    model.eval()
    with torch.no_grad():
        batches = _batches(supervised, config.training.effective_tokenizer_batch_size)
        with _task_or_noop(
            progress,
            operation="tokenizer_validation",
            label="learned_pose_token tokenizer validation",
            unit="batch",
            total=len(batches),
            total_semantics="tokenizer validation batches",
            allowed_counters=("loss",),
        ) as task:
            for batch in batches:
                values, mask, real_frame_mask = _batch_tensors(batch, stats=stats, device=device)
                with autocast_context(
                    precision_policy or resolve_precision_policy("auto", device)
                ):
                    output = model(values)
                    result = compute_tokenizer_losses(
                        output=output,
                        target=values,
                        validity_mask=mask,
                        commitment_weight=config.tokenizer.commitment_weight,
                        temporal_granularity=config.tokenizer.temporal_granularity,
                        window_size=config.tokenizer.window_size,
                        feature_dim=stats.layout.total_feature_dim,
                        real_frame_mask=real_frame_mask,
                        velocity_loss_weight=config.tokenizer.velocity_loss_weight,
                    )
                loss = float(result["total_loss"].detach().cpu().item())
                losses.append(loss)
                task.advance(1, counters={"loss": f"{loss:.6g}"})
    return float(np.mean(losses))


def _batches(
    samples: Sequence[PoseTokenTrainingSample],
    batch_size: int,
) -> tuple[tuple[PoseTokenTrainingSample, ...], ...]:
    return tuple(
        tuple(samples[index : index + batch_size])
        for index in range(0, len(samples), batch_size)
    )


class _NoOpTask:
    def advance(self, count: int = 1, *, counters=None) -> None:
        del count, counters


def _task_or_noop(progress, **kwargs):
    if progress is None:
        return nullcontext(_NoOpTask())
    return progress.task(**kwargs)


def _metric(
    *,
    split: str,
    name: str,
    value: float,
    epoch: int,
    global_step: int,
) -> TrainingMetricRecord:
    return TrainingMetricRecord(
        schema_version=TRAINING_METRIC_SCHEMA_VERSION,
        split=split,
        metric_name=name,
        value=value,
        epoch=epoch,
        global_step=global_step,
        stage_name="train_pose_tokenizer",
        created_at_utc=_now(),
        extra={"foundation_only": True},
    )


def _write_reconstruction_summary_manifest(
    path: Path,
    entries: Sequence[PoseTokenManifestEntry],
    *,
    mean_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    {
                        "schema_version": "t2sp-learned-pose-reconstruction-manifest-v1",
                        "sample_id": entry.sample_id,
                        "source_sentence_name": entry.source_sentence_name,
                        "split": entry.split.value,
                        "token_path": str(entry.token_path),
                        "frame_count": entry.frame_count,
                        "mean_validation_loss": mean_loss,
                        "limitation": "Reconstruction quality is not semantic adequacy.",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _validate_samples(samples: Sequence[PoseTokenTrainingSample], name: str) -> None:
    if any(not isinstance(sample, PoseTokenTrainingSample) for sample in samples):
        raise LearnedPoseTokenError(f"{name} must contain PoseTokenTrainingSample values.")
    dims = {sample.values.shape for sample in samples}
    if len(dims) != 1:
        raise LearnedPoseTokenError(f"{name} must have one consistent feature dimension.")


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LearnedPoseTokenError(f"{name} must be non-empty.")


__all__ = ["PoseTokenizerTrainingResult", "train_pose_tokenizer"]
