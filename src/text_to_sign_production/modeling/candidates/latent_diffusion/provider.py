"""Phase 7 provider implementation for gloss-free latent_diffusion."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.io import write_json
from text_to_sign_production.modeling.artifacts import (
    GeneratedPoseSample,
    GeneratedPoseSplitWriteResult,
    GeneratedPoseStreamWriteResult,
    diagnostic_generated_pose_paths,
    generated_manifest_entry_from_record,
    generated_manifest_entry_from_sample,
    generated_manifest_entry_to_record,
    load_generated_pose_payload,
    validate_diagnostic_generated_pose_paths,
    validate_generated_pose_manifest_entries,
    write_generated_pose_payload,
    write_generated_pose_split,
    write_generated_pose_split_to_explicit_root,
)
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    BfhStandardizationStats,
    invert_bfh_standardization,
    read_bfh_standardization_stats_json,
    write_bfh_standardization_stats_json,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    bfh_pose_arrays_from_vectorized,
    default_bfh_tensor_layout,
    unflatten_bfh_pose_values,
)
from text_to_sign_production.modeling.backbones.text_encoder import (
    build_text_encoder,
    text_encoder_metadata,
    resolve_text_encoder_output_dim,
)
from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelProviderLoadedConfig,
    ProviderCalibrationPolicy,
    ProviderFullDataPipelineDeclaration,
    ProviderFullDataPipelineCapability,
    ModelRunRequest,
    ModelRuntimeSupportArtifact,
    ModelSingleSampleInferenceContext,
    ModelSingleSampleInferenceResult,
    ModelStageArtifactRef,
    ModelStageExecutionContext,
    ModelStageExecutionError,
    ModelStageKind,
    ModelStagePlan,
    ModelStageResult,
    ModelStageStatus,
    PlannedModelStage,
    default_stage_plan_for_request,
    resolve_model_run_mode_policy,
)
from text_to_sign_production.modeling.candidates.artifacts import (
    GeneratedPoseManifestArtifactSubtype,
    generated_pose_manifest_artifact_ref,
    generated_pose_surface,
)
from text_to_sign_production.modeling.candidates.runtime_support import (
    support_artifact_from_model_run_file,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    LatentDiffusionConfig,
    latent_diffusion_config_from_mapping,
    load_latent_diffusion_config,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.autoencoder import (
    build_temporal_window_autoencoder,
    decode_window_latents,
    load_autoencoder_checkpoint,
    train_latent_autoencoder_from_surfaces,
    write_autoencoder_checkpoint,
    write_autoencoder_config,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.dataset import (
    LatentSourceSample,
    build_latent_source_surface,
    build_latent_sequence_surface_from_source_surface,
    build_latent_window_surface_from_source_surface,
    build_latent_target_spec,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.io import (
    _manifest_entry_from_record as _latent_manifest_entry_from_record,
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
    LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
    LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME,
    LatentManifestEntry,
    LatentSequence,
    LatentTargetSpec,
    expected_latent_count_for_frame_count,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.model import (
    LatentDenoiser,
    LatentLengthPredictor,
    build_latent_denoiser,
    build_latent_length_predictor,
    clamp_predicted_frame_count,
    compute_denoising_loss,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.sampling import (
    LatentSamplingConfig,
    LatentSamplingTrace,
    sample_latent_sequence,
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
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    provider_active_overrides,
)
from text_to_sign_production.workflows.model.contracts.compute_profile_application import (
    compute_profile_dataloader_section,
    provider_candidate_overrides_for_profile,
    provider_compute_profile_application,
)
from text_to_sign_production.modeling.data.prepared_sample_loader import (
    ModelingManifestSample,
)
from text_to_sign_production.modeling.data import resolve_modeling_manifest_path
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurface,
    ModelDataSurfaceReader,
    ModelDataSurfaceWriter,
    load_model_data_surface,
    model_data_surface_cache_key,
    resolve_effective_manifest_count,
    sha256_file,
    surface_batch_count,
)
from text_to_sign_production.modeling.data.temporal_windows import (
    BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION,
    BfhPoseWindowSequence,
    TemporalWindowSpec,
    merge_bfh_pose_windows,
    temporal_window_starts,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticConsistencyObjectiveConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.training import (
    compute_semantic_training_loss,
    load_semantic_training_objective_for_request,
    merge_overlapping_windows_torch,
)
from text_to_sign_production.modeling.inference.generated_pose import (
    GeneratedPoseSourceIdentity,
    generated_pose_sample_from_bfh_arrays,
)
from text_to_sign_production.modeling.registry import require_model_spec
from text_to_sign_production.modeling.research import ModelKey, ModelSpec
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
from text_to_sign_production.modeling.training.stages import ProviderStageArtifactRole


@dataclass(frozen=True, slots=True)
class LatentDenoiserTrainingResult:
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_metadata_path: Path
    last_metadata_path: Path
    metrics_path: Path
    diffusion_schedule_path: Path
    denoiser_config_path: Path
    length_predictor_config_path: Path
    final_train_loss: float
    validation_denoising_loss: float
    validation_length_loss: float
    best_metric_value: float
    completed_epoch: int
    global_step: int
    runtime_trace: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class LatentDenoiserBatch:
    latent_values: torch.Tensor
    latent_mask: torch.Tensor
    source_index: torch.Tensor
    latent_index: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("latent_values", "latent_mask", "source_index", "latent_index"):
            if not isinstance(getattr(self, name), torch.Tensor):
                raise LatentDiffusionError(f"{name} must be a torch.Tensor.")
        if self.latent_values.ndim != 2:
            raise LatentDiffusionError("latent_values must have shape (batch, latent_dim).")
        if self.latent_mask.shape != self.latent_values.shape:
            raise LatentDiffusionError("latent_mask must match latent_values shape.")
        batch = self.latent_values.shape[0]
        if self.source_index.shape != (batch,) or self.latent_index.shape != (batch,):
            raise LatentDiffusionError("source_index and latent_index must have shape (batch,).")


@dataclass(frozen=True, slots=True)
class LatentGenerationResult:
    predicted_latent_manifest_path: Path
    sampling_traces_path: Path
    decoded_generated_pose_manifest_path: Path
    generated_sample_count: int
    predicted_frame_counts: Mapping[str, int]
    configured_sampling_steps: int
    effective_sampling_steps: int
    sampling_steps_source: str


@dataclass(frozen=True, slots=True)
class LatentDecodedSampleResult:
    generated: GeneratedPoseSample
    latent_values: np.ndarray
    trace: LatentSamplingTrace
    predicted_frame_count: int
    sampled_latent_count: int
    decoded_frame_count: int


class LatentDiffusionProvider:
    """Concrete Phase 7 gloss-free latent diffusion provider."""

    @property
    def spec(self) -> ModelSpec:
        return require_model_spec(ModelKey.LATENT_DIFFUSION)

    @property
    def full_data_pipeline_capability(self) -> ProviderFullDataPipelineCapability:
        return ProviderFullDataPipelineCapability(
            provider_key=LATENT_DIFFUSION_MODEL_KEY,
            full_training_data_mode="streaming_sharded",
            verified=True,
            verification_evidence=(
                "behavior:latent_sequences:surface_reader",
                "behavior:latent_windows:surface_reader",
                "behavior:denoiser_training:surface_reader",
                "behavior:autoencoder_training:surface_reader",
                "behavior:generation:incremental_writer",
                "progress:known_totals",
            ),
            covered_stages=(
                "define_latent_target",
                "cache_latents",
                "train_denoiser",
                "generate",
                "export_generated_pose",
            ),
            limitations=(),
        )

    def full_data_pipeline_capability_for_config(
        self,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ProviderFullDataPipelineCapability:
        effective = loaded_config.effective_config
        latent_autoencoder = effective.get("latent_autoencoder")
        autoencoder_active = (
            isinstance(latent_autoencoder, Mapping)
            and latent_autoencoder.get("active") is True
        )
        evidence = [
            "behavior:latent_sequences:surface_reader",
            "behavior:denoiser_training:surface_reader",
            "behavior:generation:incremental_writer",
            "progress:known_totals",
        ]
        stages = [
            "define_latent_target",
            "cache_latents",
            "train_denoiser",
            "generate",
            "export_generated_pose",
        ]
        if autoencoder_active:
            evidence.extend(
                [
                    "behavior:latent_windows:surface_reader",
                    "behavior:autoencoder_training:surface_reader",
                ]
            )
            stages.append("train_autoencoder")
        return ProviderFullDataPipelineCapability(
            provider_key=LATENT_DIFFUSION_MODEL_KEY,
            full_training_data_mode="streaming_sharded",
            verified=True,
            verification_evidence=tuple(evidence),
            covered_stages=tuple(stages),
            limitations=(),
        )

    @property
    def full_data_pipeline_declaration(self) -> ProviderFullDataPipelineDeclaration:
        capability = self.full_data_pipeline_capability
        return ProviderFullDataPipelineDeclaration(
            provider_key=LATENT_DIFFUSION_MODEL_KEY,
            mode="streaming_sharded",
            covered_stages=capability.covered_stages,
        )

    def calibration_policy(
        self,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ProviderCalibrationPolicy:
        if loaded_config.model_key is not ModelKey.LATENT_DIFFUSION:
            raise ModelStageExecutionError(
                "latent_diffusion calibration policy requires latent_diffusion config."
            )
        application = loaded_config.effective_config.get("compute_profile_application")
        if not isinstance(application, Mapping):
            raise ModelStageExecutionError(
                "latent_diffusion calibration policy requires compute_profile_application."
            )
        candidate_keys = tuple(application.get("calibration_candidate_keys", ()))
        override_targets = application.get("calibration_override_targets")
        if not isinstance(override_targets, Mapping):
            raise ModelStageExecutionError(
                "latent_diffusion calibration policy requires calibration_override_targets."
            )
        return ProviderCalibrationPolicy(
            provider_key=LATENT_DIFFUSION_MODEL_KEY,
            supports_provider_real=True,
            required_run_modes=("full",),
            required_compute_profiles=("colab_a100_80gb",),
            candidate_keys=candidate_keys,
            override_targets=dict(override_targets),
            representative_surface_kinds={
                "denoiser_batch_size": "latent_sequences",
            },
            representative_split="train",
            max_samples=64,
            warmup_batches=1,
            max_batches_per_candidate=2,
        )

    def build_representative_calibration_surface(
        self,
        *,
        context,
        candidate_key: str,
        policy: ProviderCalibrationPolicy,
        progress_session,
    ):
        from text_to_sign_production.modeling.candidates.latent_diffusion.calibration import (
            build_representative_calibration_surface,
        )

        return build_representative_calibration_surface(
            context=context,
            candidate_key=candidate_key,
            policy=policy,
            progress_session=progress_session,
        )

    def load_config(self, request: ModelRunRequest) -> ModelProviderLoadedConfig:
        if request.model_key is not ModelKey.LATENT_DIFFUSION:
            raise ModelStageExecutionError(
                "LatentDiffusionProvider only accepts latent_diffusion requests."
            )
        if request.config_path is None:
            raise ModelStageExecutionError(
                "latent_diffusion requires MODEL_CONFIG_RELATIVE_PATH="
                "configs/modeling/latent_diffusion.yaml."
            )
        config = load_latent_diffusion_config(request.config_path)
        raw_config = _read_yaml_mapping(request.config_path)
        effective = _effective_config_payload(
            config=config,
            request=request,
            source_path=request.config_path,
        )
        return ModelProviderLoadedConfig(
            model_key=ModelKey.LATENT_DIFFUSION,
            source_path=request.config_path,
            raw_config=raw_config,
            effective_config=effective,
        )

    def plan_stages(
        self,
        request: ModelRunRequest,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ModelStagePlan:
        if request.model_key is not ModelKey.LATENT_DIFFUSION:
            raise ModelStageExecutionError("latent_diffusion planning requires latent_diffusion.")
        if loaded_config.model_key is not ModelKey.LATENT_DIFFUSION:
            raise ModelStageExecutionError("loaded config has the wrong model key.")
        return default_stage_plan_for_request(request)

    def execute_stage(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
    ) -> ModelStageResult:
        config = _config_from_loaded(context.loaded_config)
        try:
            if stage.spec.kind is ModelStageKind.DEFINE_LATENT_TARGET:
                return self._define_latent_target(stage, context, config)
            if stage.spec.kind is ModelStageKind.CACHE_LATENTS:
                return self._cache_latents(stage, context, config)
            if stage.spec.kind is ModelStageKind.TRAIN_DENOISER:
                return self._train_denoiser(stage, context, config)
            if stage.spec.kind is ModelStageKind.GENERATE:
                return self._generate(stage, context, config)
            if stage.spec.kind is ModelStageKind.EXPORT_GENERATED_POSE:
                return self._export_generated_pose(stage, context, config)
        except LatentDiffusionError as exc:
            raise ModelStageExecutionError(str(exc)) from exc
        raise ModelStageExecutionError(
            f"latent_diffusion does not implement stage {stage.spec.kind.value!r}."
        )

    def write_model_reports(
        self,
        context: ModelStageExecutionContext,
        results: ModelExecutionResult,
    ) -> tuple[ModelStageArtifactRef, ...]:
        from text_to_sign_production.modeling.candidates.latent_diffusion.reports import (
            write_latent_diffusion_reports,
        )

        return write_latent_diffusion_reports(context=context, results=results)

    def runtime_support_artifacts(
        self,
        *,
        request: ModelRunRequest,
        loaded_config: ModelProviderLoadedConfig,
        execution: ModelExecutionResult,
        topology,
    ) -> tuple[ModelRuntimeSupportArtifact, ...]:
        del execution
        config = _config_from_loaded(loaded_config)
        root = topology.models.model_run_root(
            LATENT_DIFFUSION_MODEL_KEY,
            request.run_name,
        ).path
        latent = topology.models.model_intermediate_root(
            LATENT_DIFFUSION_MODEL_KEY,
            request.run_name,
            "latent",
        ).path
        denoiser = topology.models.model_intermediate_root(
            LATENT_DIFFUSION_MODEL_KEY,
            request.run_name,
            "denoiser",
        ).path
        specs: list[tuple[str, Path]] = [
            ("latent_diffusion_target_spec", latent / "target_spec.json"),
            ("latent_diffusion_standardization_stats", latent / "standardization_stats.json"),
            ("latent_diffusion_schedule", denoiser / "diffusion_schedule.json"),
            ("latent_diffusion_denoiser_config", denoiser / "denoiser_config.json"),
            ("latent_diffusion_length_predictor_config", denoiser / "length_predictor_config.json"),
        ]
        if config.latent_target.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
            specs.extend(
                (
                    (
                        "latent_diffusion_autoencoder_checkpoint",
                        latent / "autoencoder" / "window_autoencoder.pt",
                    ),
                    (
                        "latent_diffusion_autoencoder_config",
                        latent / "autoencoder" / "autoencoder_config.json",
                    ),
                )
            )
        return tuple(
            support_artifact_from_model_run_file(
                model_run_root=root,
                path=path,
                role=role,
                provider_key=LATENT_DIFFUSION_MODEL_KEY,
            )
            for role, path in specs
        )

    def infer_single_sample(
        self,
        context: ModelSingleSampleInferenceContext,
    ) -> ModelSingleSampleInferenceResult:
        config = _config_from_loaded(context.loaded_config)
        paths = _required_runtime_support_paths(
            context.topology,
            context.request.run_name,
            config=config,
        )
        _require_file(context.checkpoint_path, "selected latent denoiser checkpoint")
        target_spec = read_latent_target_spec_json(paths["target_spec"])
        stats = read_bfh_standardization_stats_json(paths["standardization_stats"])
        schedule = _read_diffusion_schedule_json(paths["diffusion_schedule"])
        checkpoint = _load_checkpoint(context.checkpoint_path)
        _verify_restored_model_configs(
            denoiser_config_path=paths["denoiser_config"],
            length_predictor_config_path=paths["length_predictor_config"],
            config=config,
            checkpoint=checkpoint,
        )
        text_encoder = build_text_encoder(config.text_encoder)
        denoiser, length_predictor = _models_from_checkpoint(
            checkpoint,
            config=config,
            latent_dim=target_spec.latent_dim,
            text_encoder_output_dim=text_encoder.output_dim,
        )
        source = _source_from_manifest_sample(context.sample)
        run_seed = _run_seed(context.loaded_config.effective_config)
        seed = _sample_seed(
            run_seed=run_seed,
            sample_id=source.sample_id,
            candidate_index=0,
            seed_policy=config.generation.seed_policy,
        )
        generated_result = _generate_one_decoded_sample(
            source=source,
            config=config,
            target_spec=target_spec,
            stats=stats,
            schedule=schedule,
            denoiser=denoiser,
            length_predictor=length_predictor,
            run_name=context.request.run_name,
            seed=seed,
            candidate_index=0,
            device=_resolve_device(config.training.device),
            autoencoder_checkpoint_path=paths.get("autoencoder_checkpoint"),
        )
        written = write_generated_pose_split_to_explicit_root(
            manifest_path=context.output_root / "manifest.jsonl",
            samples_root=context.output_root / "samples",
            split=source.split,
            samples=(generated_result.generated,),
            payload_ref_root=context.output_root.parent,
        )
        if len(written.payload_paths) != 1:
            raise ModelStageExecutionError("latent_diffusion single-sample inference wrote no payload.")
        return ModelSingleSampleInferenceResult(
            model_key=context.request.model_key,
            run_name=context.request.run_name,
            checkpoint_path=context.checkpoint_path,
            sample_id=source.sample_id,
            generated_sample=load_generated_pose_payload(written.payload_paths[0]),
            generated_payload_path=written.payload_paths[0],
            generated_manifest_path=written.manifest_path,
            metadata={
                "generation_mode": config.generation.generation_mode,
                "producer_stage": "test_model_single_sample",
                "seed": seed,
                "candidate_index": 0,
                "sampling_steps": generated_result.trace.sampling_steps,
                "length_policy": config.length.policy,
                "confidence_policy": config.generation.confidence_policy,
                "predicted_frame_count": generated_result.predicted_frame_count,
                "sampled_latent_count": generated_result.sampled_latent_count,
                "decoded_frame_count": generated_result.decoded_frame_count,
                "target_type": target_spec.target_type,
                "temporal_granularity": target_spec.temporal_granularity,
                "window_size": target_spec.window_size,
                "stride": target_spec.stride,
                "checkpoint_path": str(context.checkpoint_path),
                "target_spec_path": str(paths["target_spec"]),
                "standardization_stats_path": str(paths["standardization_stats"]),
                "diffusion_schedule_path": str(paths["diffusion_schedule"]),
                "denoiser_config_path": str(paths["denoiser_config"]),
                "length_predictor_config_path": str(paths["length_predictor_config"]),
            },
        )

    def _define_latent_target(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LatentDiffusionConfig,
    ) -> ModelStageResult:
        context.provider_progress(stage).status(
            "latent_diffusion define latent target",
            target_type=config.latent_target.target_type,
        )
        layout = default_bfh_tensor_layout()
        spec = build_latent_target_spec(
            config=config.latent_target,
            layout=layout,
            learned_latent_dim=(
                None if config.latent_autoencoder is None else config.latent_autoencoder.latent_dim
            ),
        )
        root = _latent_root(context)
        target_path = root / "target_spec.json"
        write_latent_target_spec_json(target_path, spec)
        report_payload = {
            "schema_version": "t2sp-latent-target-decision-report-v1",
            "model_key": LATENT_DIFFUSION_MODEL_KEY,
            "target_type": spec.target_type,
            "temporal_granularity": spec.temporal_granularity,
            "latent_dim": spec.latent_dim,
            "statement": (
                "The active latent target is provider-owned by latent_diffusion; "
                "learned_bfh_window_latent is a continuous temporal-window "
                "autoencoder latent when selected, not a learned_pose_token codebook."
            ),
        }
        report_path = _report_root(context) / "latent_target_report.json"
        write_json(report_path, report_payload)
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role="latent_target_spec",
                    path=target_path,
                    kind="model_latent_target_spec",
                    metadata={
                        "target_type": spec.target_type,
                        "temporal_granularity": spec.temporal_granularity,
                        "latent_dim": spec.latent_dim,
                        "split": "none",
                    },
                ),
                ModelStageArtifactRef(
                    role="latent_target_report",
                    path=report_path,
                    kind="model_latent_provider_report",
                ),
            ),
            metadata={
                **_latent_diffusion_performance_metadata(config),
                "target_spec_path": str(target_path),
                "target_type": spec.target_type,
                "temporal_granularity": spec.temporal_granularity,
                "latent_dim": spec.latent_dim,
            },
        )

    def _cache_latents(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LatentDiffusionConfig,
    ) -> ModelStageResult:
        target_stage = _require_stage_result(context, ModelStageKind.DEFINE_LATENT_TARGET)
        target_spec = read_latent_target_spec_json(Path(str(target_stage.metadata["target_spec_path"])))
        caps = _run_mode_caps(context.loaded_config.effective_config)
        progress = context.provider_progress(stage)
        train_source_surface = _materialize_latent_source_surface(
            context=context,
            config=config,
            split=context.request.train_split,
            max_samples=caps["limit_train_samples"],
            progress=progress,
            operation="materialize_train",
            label="latent_diffusion build train source surface",
            total_semantics="train source samples written to tensor surface",
        )
        validation_source_surface = _materialize_latent_source_surface(
            context=context,
            config=config,
            split=context.request.validation_split,
            max_samples=caps["limit_validation_samples"],
            progress=progress,
            operation="materialize_val",
            label="latent_diffusion build validation source surface",
            total_semantics="validation source samples written to tensor surface",
        )
        if (
            train_source_surface.metadata.loaded_sample_count <= 0
            or validation_source_surface.metadata.loaded_sample_count <= 0
        ):
            raise LatentDiffusionError("cache_latents requires non-empty train and validation sources.")
        root = _latent_root(context)
        stats_path = root / "standardization_stats.json"
        stats = read_bfh_standardization_stats_json(
            train_source_surface.root / "standardization_stats.json"
        )
        write_bfh_standardization_stats_json(stats_path, stats)
        autoencoder_checkpoint_path: Path | None = None
        autoencoder_config_path: Path | None = None
        autoencoder_metrics: dict[str, float] = {}
        autoencoder_runtime_trace: dict[str, object] = {}
        if target_spec.target_type == LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
            train_latent_surface = _materialize_latent_sequence_surface(
                context=context,
                config=config,
                source_surface=train_source_surface,
                stats=stats,
                target_spec=target_spec,
                split=context.request.train_split,
                progress=progress,
                operation="cache_train_latents",
                label="latent_diffusion cache train latent surface",
                total_semantics="train latent source shards cached",
            )
            validation_latent_surface = _materialize_latent_sequence_surface(
                context=context,
                config=config,
                source_surface=validation_source_surface,
                stats=stats,
                target_spec=target_spec,
                split=context.request.validation_split,
                progress=progress,
                operation="cache_validation_latents",
                label="latent_diffusion cache validation latent surface",
                total_semantics="validation latent source shards cached",
            )
            train_entries = _write_latent_manifest_from_surface(
                train_latent_surface,
                output_root=root / "samples" / "train",
                manifest_path=root / "manifests" / "train.jsonl",
                target_spec=target_spec,
                progress=progress,
                operation="write_train_latent_manifest",
                label="latent_diffusion write train latent manifest",
            )
            val_entries = _write_latent_manifest_from_surface(
                validation_latent_surface,
                output_root=root / "samples" / "val",
                manifest_path=root / "manifests" / "val.jsonl",
                target_spec=target_spec,
                progress=progress,
                operation="write_validation_latent_manifest",
                label="latent_diffusion write validation latent manifest",
            )
        else:
            if config.latent_autoencoder is None:
                raise LatentDiffusionError(
                    "latent_autoencoder config is required when "
                    "latent_target.target_type='learned_bfh_window_latent'."
                )
            input_dim = target_spec.window_size * target_spec.base_feature_dim
            train_window_surface = _materialize_latent_window_surface(
                context=context,
                config=config,
                source_surface=train_source_surface,
                stats=stats,
                target_spec=target_spec,
                split=context.request.train_split,
                progress=progress,
                operation="build_train_windows",
                label="latent_diffusion build train window surface",
                total_semantics="train source shards converted to temporal window surface",
            )
            validation_window_surface = _materialize_latent_window_surface(
                context=context,
                config=config,
                source_surface=validation_source_surface,
                stats=stats,
                target_spec=target_spec,
                split=context.request.validation_split,
                progress=progress,
                operation="build_validation_windows",
                label="latent_diffusion build validation window surface",
                total_semantics="validation source shards converted to temporal window surface",
            )
            autoencoder = build_temporal_window_autoencoder(
                input_dim=input_dim,
                config=config.latent_autoencoder,
            )
            device = _resolve_device(config.training.device)
            precision_policy = resolve_precision_policy(_compute_precision_policy(context), device)
            autoencoder_metrics = train_latent_autoencoder_from_surfaces(
                model=autoencoder,
                train_window_surface=train_window_surface,
                validation_window_surface=validation_window_surface,
                config=config.latent_autoencoder,
                window_size=target_spec.window_size,
                feature_dim=target_spec.base_feature_dim,
                device=device,
                progress=progress,
                precision_policy=precision_policy,
                num_workers=config.training.num_workers,
                runtime_trace=autoencoder_runtime_trace,
            )
            autoencoder_checkpoint_path = root / "autoencoder" / "window_autoencoder.pt"
            autoencoder_config_path = root / "autoencoder" / "autoencoder_config.json"
            write_autoencoder_checkpoint(
                path=autoencoder_checkpoint_path,
                model=autoencoder,
                config=config.latent_autoencoder,
                input_dim=input_dim,
                target_spec=target_spec,
                metrics=autoencoder_metrics,
            )
            write_autoencoder_config(autoencoder_config_path, config.latent_autoencoder)
            train_latent_surface = _materialize_encoded_window_latent_surface(
                context=context,
                config=config,
                window_surface=train_window_surface,
                autoencoder=autoencoder,
                target_spec=target_spec,
                split=context.request.train_split,
                device=device,
                progress=progress,
                operation="cache_train_latents",
                label="latent_diffusion cache train latent surface",
                total_semantics="train temporal windows encoded into latent surface",
            )
            validation_latent_surface = _materialize_encoded_window_latent_surface(
                context=context,
                config=config,
                window_surface=validation_window_surface,
                autoencoder=autoencoder,
                target_spec=target_spec,
                split=context.request.validation_split,
                device=device,
                progress=progress,
                operation="cache_validation_latents",
                label="latent_diffusion cache validation latent surface",
                total_semantics="validation temporal windows encoded into latent surface",
            )
            train_entries = _write_latent_manifest_from_surface(
                train_latent_surface,
                output_root=root / "samples" / "train",
                manifest_path=root / "manifests" / "train.jsonl",
                target_spec=target_spec,
                progress=progress,
                operation="write_train_latent_manifest",
                label="latent_diffusion write train latent manifest",
            )
            val_entries = _write_latent_manifest_from_surface(
                validation_latent_surface,
                output_root=root / "samples" / "val",
                manifest_path=root / "manifests" / "val.jsonl",
                target_spec=target_spec,
                progress=progress,
                operation="write_validation_latent_manifest",
                label="latent_diffusion write validation latent manifest",
            )
        train_manifest = root / "manifests" / "train.jsonl"
        val_manifest = root / "manifests" / "val.jsonl"
        write_latent_manifest_jsonl(train_manifest, train_entries)
        write_latent_manifest_jsonl(val_manifest, val_entries)
        train_latent_units = sum(entry.latent_count for entry in train_entries)
        validation_latent_units = sum(entry.latent_count for entry in val_entries)
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role="standardization_stats",
                    path=stats_path,
                    kind="model_latent_standardization_stats",
                    metadata=_standardization_metadata(stats),
                ),
                ModelStageArtifactRef(
                    role="latent_manifest",
                    path=train_manifest,
                    kind="model_latent_manifest",
                    metadata={"split": "train"},
                ),
                ModelStageArtifactRef(
                    role="latent_manifest",
                    path=val_manifest,
                    kind="model_latent_manifest",
                    metadata={"split": "val"},
                ),
                *(
                    ()
                    if autoencoder_checkpoint_path is None
                    else (
                        ModelStageArtifactRef(
                            role="latent_autoencoder_checkpoint",
                            path=autoencoder_checkpoint_path,
                            kind="model_latent_autoencoder_checkpoint",
                        ),
                        ModelStageArtifactRef(
                            role="latent_autoencoder_config",
                            path=autoencoder_config_path,
                            kind="model_intermediate",
                        ),
                    )
                ),
            ),
            metadata={
                **_latent_diffusion_performance_metadata(
                    config,
                    runtime_trace=(
                        autoencoder_runtime_trace
                        if config.latent_autoencoder is not None
                        else None
                    ),
                ),
                "standardization_stats_path": str(stats_path),
                **_standardization_metadata(stats),
                "train_latent_manifest_path": str(train_manifest),
                "validation_latent_manifest_path": str(val_manifest),
                "train_sample_count": len(train_entries),
                "validation_sample_count": len(val_entries),
                "target_type": target_spec.target_type,
                "temporal_granularity": target_spec.temporal_granularity,
                "window_size": target_spec.window_size,
                "stride": target_spec.stride,
                "latent_dim": target_spec.latent_dim,
                "train_source_sample_count": train_source_surface.metadata.loaded_sample_count,
                "validation_source_sample_count": validation_source_surface.metadata.loaded_sample_count,
                "train_latent_unit_count": train_latent_units,
                "validation_latent_unit_count": validation_latent_units,
                "train_source_surface_path": str(train_source_surface.root),
                "validation_source_surface_path": str(validation_source_surface.root),
                "train_latent_surface_path": str(train_latent_surface.root),
                "validation_latent_surface_path": str(validation_latent_surface.root),
                "autoencoder_checkpoint_path": (
                    None if autoencoder_checkpoint_path is None else str(autoencoder_checkpoint_path)
                ),
                "autoencoder_config_path": (
                    None if autoencoder_config_path is None else str(autoencoder_config_path)
                ),
                **_precision_metadata(
                    context,
                    config,
                    precision_applied=autoencoder_checkpoint_path is not None,
                ),
                **autoencoder_metrics,
            },
        )

    def _train_denoiser(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LatentDiffusionConfig,
    ) -> ModelStageResult:
        result = _train_latent_denoiser(
            context=context,
            config=config,
            progress=context.provider_progress(stage),
        )

        runtime_trace = _require_latent_surface_reader_runtime_trace(
            result.runtime_trace,
            expected_num_workers=config.training.num_workers,
            context="latent_diffusion.train_denoiser",
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role=ProviderStageArtifactRole.CHECKPOINT_BEST,
                    path=result.best_checkpoint_path,
                    kind="model_latent_denoiser_checkpoint",
                ),
                ModelStageArtifactRef(
                    role=ProviderStageArtifactRole.CHECKPOINT_LAST,
                    path=result.last_checkpoint_path,
                    kind="model_latent_denoiser_checkpoint",
                ),
                ModelStageArtifactRef(
                    role=ProviderStageArtifactRole.TRAINING_METRICS,
                    path=result.metrics_path,
                    kind="model_latent_denoiser_metrics",
                ),
                ModelStageArtifactRef(
                    role="diffusion_schedule",
                    path=result.diffusion_schedule_path,
                    kind="model_diffusion_schedule",
                ),
                ModelStageArtifactRef(
                    role="denoiser_config",
                    path=result.denoiser_config_path,
                    kind="model_intermediate",
                ),
                ModelStageArtifactRef(
                    role="length_predictor_config",
                    path=result.length_predictor_config_path,
                    kind="model_intermediate",
                ),
            ),
            metrics={
                "final_train_loss": result.final_train_loss,
                "validation_denoising_loss": result.validation_denoising_loss,
                "validation_length_loss": result.validation_length_loss,
                "best_metric_value": result.best_metric_value,
            },
            metadata={
                **_latent_diffusion_performance_metadata(
                    config,
                    runtime_trace=runtime_trace,
                ),
                "best_checkpoint_path": str(result.best_checkpoint_path),
                "last_checkpoint_path": str(result.last_checkpoint_path),
                "best_checkpoint_metadata_path": str(result.best_metadata_path),
                "last_checkpoint_metadata_path": str(result.last_metadata_path),
                "training_metrics_path": str(result.metrics_path),
                "diffusion_schedule_path": str(result.diffusion_schedule_path),
                "denoiser_config_path": str(result.denoiser_config_path),
                "length_predictor_config_path": str(result.length_predictor_config_path),
                "best_metric_name": config.checkpoints.selection_metric,
                "best_metric_value": result.best_metric_value,
                "completed_epoch": result.completed_epoch,
                "global_step": result.global_step,
                **_precision_metadata(context, config),
            },
        )

    def _generate(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LatentDiffusionConfig,
    ) -> ModelStageResult:
        progress = context.provider_progress(stage)
        result = _generate_validation_latents(context=context, config=config, progress=progress)
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role="predicted_latent_manifest",
                    path=result.predicted_latent_manifest_path,
                    kind="model_latent_generation_manifest",
                    metadata={
                        "split": "val",
                        "generation_mode": "stochastic",
                        "producer_stage": "generate",
                    },
                ),
                ModelStageArtifactRef(
                    role="sampling_trace",
                    path=result.sampling_traces_path,
                    kind="model_latent_sampling_trace",
                    metadata={"split": "val", "producer_stage": "generate"},
                ),
            ),
            generated_pose_surfaces=(
                generated_pose_surface(
                    manifest_path=diagnostic_generated_pose_paths(
                        _generation_root(context) / "decoded_pose_intermediates" / "val"
                    ).manifest_path,
                    samples_dir=diagnostic_generated_pose_paths(
                        _generation_root(context) / "decoded_pose_intermediates" / "val"
                    ).samples_root,
                    split=SampleSplit.VAL,
                    model_key=LATENT_DIFFUSION_MODEL_KEY,
                    model_run_name=context.request.run_name,
                    manifest_family=context.request.manifest_family.family_id,
                    producer_stage_id="generate",
                    artifact_subtype=GeneratedPoseManifestArtifactSubtype.DECODED_INTERMEDIATE,
                ),
            ),
            metadata={
                "predicted_latent_manifest_path": str(result.predicted_latent_manifest_path),
                "predicted_latent_sample_root": str(
                    result.predicted_latent_manifest_path.parent
                    / "predicted_latent_samples"
                    / "val"
                ),
                "predicted_latent_sample_count": result.generated_sample_count,
                "sampling_traces_path": str(result.sampling_traces_path),
                "decoded_intermediate_manifest_path": str(
                    result.decoded_generated_pose_manifest_path
                ),
                "decoded_intermediate_sample_count": result.generated_sample_count,
                "generated_sample_count": result.generated_sample_count,
                "generation_mode": "stochastic",
                "seed_policy": config.generation.seed_policy,
                "candidate_count": config.generation.candidate_count,
                "predicted_frame_counts": dict(result.predicted_frame_counts),
                "configured_sampling_steps": result.configured_sampling_steps,
                "effective_sampling_steps": result.effective_sampling_steps,
                "sampling_steps_source": result.sampling_steps_source,
                **_latent_diffusion_performance_metadata(
                    config,
                    stage="generate",
                    configured_sampling_steps=result.configured_sampling_steps,
                    effective_sampling_steps=result.effective_sampling_steps,
                    candidate_count=config.generation.candidate_count,
                ),
                **_precision_metadata(context, config, precision_applied=False),
                "generation_precision_applied": False,
                "generation_precision_reason": "sampling kept in fp32 for numerical stability",
            },
        )

    def _export_generated_pose(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LatentDiffusionConfig,
    ) -> ModelStageResult:
        generate_stage = _require_stage_result(context, ModelStageKind.GENERATE)
        raw_manifest = generate_stage.metadata.get("decoded_intermediate_manifest_path")
        if not isinstance(raw_manifest, str) or not raw_manifest:
            raise LatentDiffusionError(
                "export_generated_pose requires generate metadata decoded_intermediate_manifest_path; "
                "run generate before export."
            )
        decoded_manifest_path = Path(raw_manifest)
        decoded_count = int(generate_stage.metadata.get("decoded_intermediate_sample_count", 0))
        if decoded_count <= 0:
            raise LatentDiffusionError("export_generated_pose found no decoded pose intermediates.")
        progress = context.provider_progress(stage)
        with progress.task(
            operation="write_final_generated_pose",
            label="latent_diffusion write final generated pose split",
            unit="sample",
            total=decoded_count,
            total_semantics="final validation generated pose samples written",
        ) as task:
            written = _write_generated_pose_payloads_from_manifest_incremental(
                topology=context.topology,
                manifest_path=decoded_manifest_path,
                producer_key=LATENT_DIFFUSION_MODEL_KEY,
                run_name=context.request.run_name,
                split=context.request.validation_split,
                on_sample_written=lambda: task.advance(),
            )
        artifacts: list[ModelStageArtifactRef] = [
            generated_pose_manifest_artifact_ref(
                path=written.manifest_path,
                kind="generated_pose_manifest",
                split=context.request.validation_split,
                generation_mode=config.generation.generation_mode,
                producer_stage="export_generated_pose",
                model_key=LATENT_DIFFUSION_MODEL_KEY,
                artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
            )
        ]
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=tuple(artifacts),
            generated_pose_surfaces=(
                generated_pose_surface(
                    manifest_path=written.manifest_path,
                    samples_dir=written.manifest_path.parent / "samples",
                    split=context.request.validation_split,
                    model_key=LATENT_DIFFUSION_MODEL_KEY,
                    model_run_name=context.request.run_name,
                    manifest_family=context.request.manifest_family.family_id,
                    producer_stage_id="export_generated_pose",
                    artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
                ),
            ),
            metadata={
                **_latent_diffusion_performance_metadata(config),
                "manifest_paths": {"val": str(written.manifest_path)},
                "split_sample_counts": {"val": written.sample_count},
                "total_sample_count": written.sample_count,
                "generation_mode": config.generation.generation_mode,
                "producer_stage": "export_generated_pose",
            },
        )

def _require_latent_surface_reader_runtime_trace(
    runtime_trace: Mapping[str, object] | None,
    *,
    expected_num_workers: int,
    context: str,
) -> Mapping[str, object]:
    if expected_num_workers <= 0:
        return {} if runtime_trace is None else runtime_trace

    if not isinstance(runtime_trace, Mapping):
        raise LatentDiffusionError(
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
        raise LatentDiffusionError(
            f"{context} expected surface reader runtime trace keys {missing} "
            f"because num_workers={expected_num_workers}. This means the stage did "
            "not execute through the trace-aware ModelDataSurfaceReader.iter_batches path."
        )

    observed_num_workers = runtime_trace.get("surface_reader_num_workers_used")
    if observed_num_workers != expected_num_workers:
        raise LatentDiffusionError(
            f"{context} surface reader num_workers mismatch: "
            f"expected {expected_num_workers}, observed {observed_num_workers}."
        )

    expected_mode = "multiprocess" if expected_num_workers > 0 else "single_process"
    observed_mode = runtime_trace.get("surface_reader_worker_mode")
    if observed_mode != expected_mode:
        raise LatentDiffusionError(
            f"{context} surface reader worker mode mismatch: "
            f"expected {expected_mode!r}, observed {observed_mode!r}."
        )

    return runtime_trace


def _train_latent_denoiser(
    *,
    context: ModelStageExecutionContext,
    config: LatentDiffusionConfig,
    progress=None,
) -> LatentDenoiserTrainingResult:
    cache_stage = _require_stage_result(context, ModelStageKind.CACHE_LATENTS)
    target_stage = _require_stage_result(context, ModelStageKind.DEFINE_LATENT_TARGET)
    stats_path = Path(str(cache_stage.metadata["standardization_stats_path"]))
    autoencoder_checkpoint_path = cache_stage.metadata.get("autoencoder_checkpoint_path")
    autoencoder_config_path = cache_stage.metadata.get("autoencoder_config_path")
    target_spec_path = Path(str(target_stage.metadata["target_spec_path"]))
    target_spec = read_latent_target_spec_json(target_spec_path)
    train_latent_surface = load_model_data_surface(
        Path(str(cache_stage.metadata["train_latent_surface_path"]))
    )
    validation_latent_surface = load_model_data_surface(
        Path(str(cache_stage.metadata["validation_latent_surface_path"]))
    )
    reader_runtime_trace: dict[str, object] = {}
    first_batch = next(
        iter_latent_sequence_batches_from_surface(
            train_latent_surface,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            runtime_trace=reader_runtime_trace,
        ),
        None,
    )
    if first_batch is None:
        raise LatentDiffusionError("train latent surface must contain at least one sequence.")
    latent_dim = int(first_batch.latent_values.shape[1])
    for batch in iter_latent_sequence_batches_from_surface(
        validation_latent_surface,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    ):
        if int(batch.latent_values.shape[1]) != latent_dim:
            raise LatentDiffusionError("all latent sequences must share one latent_dim.")
    run_seed = _run_seed(context.loaded_config.effective_config)
    sampling_steps, sampling_steps_source = _effective_sampling_steps(
        context.loaded_config.effective_config,
        configured_steps=config.diffusion.sampling_steps,
    )
    _set_seed(run_seed)
    schedule = build_linear_beta_schedule(config.diffusion)
    text_encoder = build_text_encoder(config.text_encoder)
    device = _resolve_device(config.training.device)
    precision_policy = resolve_precision_policy(_compute_precision_policy(context), device)
    semantic_config = load_semantic_training_objective_for_request(
        request=context.request,
        model_key=LATENT_DIFFUSION_MODEL_KEY,
    )
    semantic_autoencoder = None
    if (
        semantic_config is not None
        and target_spec.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT
    ):
        if config.latent_autoencoder is None or autoencoder_checkpoint_path is None:
            raise LatentDiffusionError(
                "semantic_consistency learned temporal latent training requires the window autoencoder checkpoint."
            )
        semantic_autoencoder = load_autoencoder_checkpoint(
            Path(str(autoencoder_checkpoint_path)),
            config=config.latent_autoencoder,
            input_dim=target_spec.window_size * target_spec.base_feature_dim,
        ).to(device)
        semantic_autoencoder.eval()
        for parameter in semantic_autoencoder.parameters():
            parameter.requires_grad_(False)
    train_source_surface = load_model_data_surface(
        Path(str(cache_stage.metadata["train_source_surface_path"]))
    )
    validation_source_surface = load_model_data_surface(
        Path(str(cache_stage.metadata["validation_source_surface_path"]))
    )
    train_sources = _latent_sources_by_index_from_surface(train_source_surface)
    val_sources = _latent_sources_by_index_from_surface(validation_source_surface)
    text_dim = text_encoder.output_dim
    denoiser = build_latent_denoiser(
        latent_dim=latent_dim,
        text_embedding_dim=text_dim,
        config=config,
    ).to(device)
    length_predictor = build_latent_length_predictor(
        text_embedding_dim=text_dim,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (*denoiser.parameters(), *length_predictor.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    max_epochs = _effective_max_epochs(context.loaded_config.effective_config, config)
    metrics: list[TrainingMetricRecord] = []
    global_step = 0
    final_train_loss = 0.0
    for epoch in range(max_epochs):
        losses = []
        denoiser.train()
        length_predictor.train()
        with progress.task(
            operation=f"train_denoiser_epoch_{epoch + 1}",
            label=f"latent_diffusion train denoiser epoch {epoch + 1}/{max_epochs}",
            unit="batch",
            total=surface_batch_count(
                unit_count=train_latent_surface.metadata.unit_count,
                batch_size=config.training.batch_size,
            ),
            total_semantics="latent denoiser training batches in epoch",
            allowed_counters=("loss",),
        ) as task:
            for batch in iter_latent_sequence_batches_from_surface(
                train_latent_surface,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
                runtime_trace=reader_runtime_trace,
            ):
                if int(batch.latent_values.shape[1]) != latent_dim:
                    raise LatentDiffusionError("all latent sequences must share one latent_dim.")
                optimizer.zero_grad(set_to_none=True)
                losses_map = _denoiser_batch_losses(
                    batch,
                    train_sources,
                    text_encoder=text_encoder,
                    schedule=schedule,
                    denoiser=denoiser,
                    length_predictor=length_predictor,
                    device=device,
                    config=config,
                    semantic_config=semantic_config,
                    semantic_autoencoder=semantic_autoencoder,
                    precision_policy=precision_policy,
                )
                total = losses_map["total_loss"]
                total.backward()
                optimizer.step()
                loss_value = _finite_tensor_value(total, "total_loss")
                losses.append(loss_value)
                metrics.append(_metric("train", "total_loss", loss_value, epoch, global_step))
                metrics.append(
                    _metric(
                        "train",
                        "denoising_loss",
                        _finite_tensor_value(losses_map["denoising_loss"], "denoising_loss"),
                        epoch,
                        global_step,
                    )
                )
                metrics.append(
                    _metric(
                        "train",
                        "length_loss",
                        _finite_tensor_value(losses_map["length_loss"], "length_loss"),
                        epoch,
                        global_step,
                    )
                )
                for semantic_metric in (
                    "semantic_loss",
                    "semantic_weighted_loss",
                    "semantic_cosine_similarity",
                    "semantic_cosine_distance",
                ):
                    if semantic_metric in losses_map:
                        metrics.append(
                            _metric(
                                "train",
                                semantic_metric,
                                _finite_tensor_value(losses_map[semantic_metric], semantic_metric),
                                epoch,
                                global_step,
                            )
                        )
                task.advance(1, counters={"loss": f"{loss_value:.6g}"})
                global_step += 1
        final_train_loss = _finite_float(float(np.mean(losses)), "final_train_loss")
    validation = _validation_batch_losses(
        iter_latent_sequence_batches_from_surface(
            validation_latent_surface,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
        ),
        val_sources,
        text_encoder=text_encoder,
        schedule=schedule,
        denoiser=denoiser,
        length_predictor=length_predictor,
        device=device,
        config=config,
        epoch=max_epochs - 1,
        global_step=global_step,
        metrics=metrics,
        progress=progress,
        precision_policy=precision_policy,
        total=surface_batch_count(
            unit_count=validation_latent_surface.metadata.unit_count,
            batch_size=config.training.batch_size,
        ),
    )
    checkpoint_root = context.topology.models.model_checkpoints_root(
        LATENT_DIFFUSION_MODEL_KEY,
        context.request.run_name,
    ).path
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    denoiser_root = _denoiser_root(context)
    denoiser_root.mkdir(parents=True, exist_ok=True)
    schedule_path = denoiser_root / "diffusion_schedule.json"
    denoiser_config_path = denoiser_root / "denoiser_config.json"
    length_config_path = denoiser_root / "length_predictor_config.json"
    _write_diffusion_schedule_json(schedule_path, schedule)
    write_json(denoiser_config_path, {"schema_version": "t2sp-latent-denoiser-config-v1", **config.denoiser.to_dict()})
    write_json(length_config_path, {"schema_version": "t2sp-latent-length-predictor-config-v1", **config.length.to_dict()})
    payload = {
        "schema_version": "t2sp-latent-diffusion-checkpoint-v1",
        "model_key": LATENT_DIFFUSION_MODEL_KEY,
        "run_name": context.request.run_name,
        "denoiser_state_dict": denoiser.state_dict(),
        "length_predictor_state_dict": length_predictor.state_dict(),
        "latent_dim": latent_dim,
        "text_embedding_dim": text_dim,
        "text_encoder_config": config.text_encoder.to_dict(),
        "text_encoder_metadata": text_encoder_metadata(text_encoder),
        "latent_target_spec": target_spec.to_dict(),
        "latent_target_spec_path": str(target_spec_path),
        "target_type": target_spec.target_type,
        "temporal_granularity": target_spec.temporal_granularity,
        "window_size": target_spec.window_size,
        "stride": target_spec.stride,
        "latent_sequence_length_semantics": (
            "latent_count is diffusion sequence length; frame_count is length predictor target."
        ),
        "autoencoder_checkpoint_path": autoencoder_checkpoint_path,
        "autoencoder_config_path": autoencoder_config_path,
        "standardization_stats_path": str(stats_path),
        "diffusion_schedule_path": str(schedule_path),
        "provider_config": config.to_dict(),
        "run_seed": run_seed,
        "selection_metric": config.checkpoints.selection_metric,
        "selection_metric_value": validation["denoising_loss"],
    }
    payload.update(_semantic_checkpoint_extra(semantic_config))
    last_path = checkpoint_root / "last.pt"
    best_path = checkpoint_root / "best.pt"
    torch.save(payload, last_path)
    torch.save(payload, best_path)
    metrics_path = denoiser_root / "training_metrics.jsonl"
    write_training_metric_records_jsonl(metrics_path, metrics)
    best_meta = denoiser_root / "best_metadata.json"
    last_meta = denoiser_root / "last_metadata.json"
    for role, checkpoint_path, metadata_path in (
        (CheckpointRole.BEST, best_path, best_meta),
        (CheckpointRole.LAST, last_path, last_meta),
    ):
        write_checkpoint_metadata_json(
            metadata_path,
            ModelCheckpointMetadata(
                schema_version=CHECKPOINT_METADATA_SCHEMA_VERSION,
                model_key=LATENT_DIFFUSION_MODEL_KEY,
                run_name=context.request.run_name,
                role=role,
                stage_name="train_denoiser",
                checkpoint_path=checkpoint_path,
                epoch=max_epochs - 1,
                global_step=global_step,
                selection_metric=config.checkpoints.selection_metric,
                selection_metric_value=validation["denoising_loss"],
                lower_is_better=config.checkpoints.lower_is_better,
                created_at_utc=_now(),
                extra={
                    "length_loss": validation["length_loss"],
                    "sampling_steps": config.diffusion.sampling_steps,
                    "target_spec_path": str(target_spec_path),
                    "target_type": target_spec.target_type,
                    "temporal_granularity": target_spec.temporal_granularity,
                    "window_size": target_spec.window_size,
                    "stride": target_spec.stride,
                    "latent_dim": latent_dim,
                    "latent_sequence_length_semantics": (
                        "latent_count is diffusion sequence length; frame_count is length predictor target."
                    ),
                    "autoencoder_checkpoint_path": autoencoder_checkpoint_path,
                    "autoencoder_config_path": autoencoder_config_path,
                    "standardization_stats_path": str(stats_path),
                    "diffusion_schedule_path": str(schedule_path),
                    **_semantic_checkpoint_extra(semantic_config),
                },
            ),
        )
    
    _require_latent_surface_reader_runtime_trace(
        reader_runtime_trace,
        expected_num_workers=config.training.num_workers,
        context="_train_latent_denoiser",
    )

    return LatentDenoiserTrainingResult(
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
        best_metadata_path=best_meta,
        last_metadata_path=last_meta,
        metrics_path=metrics_path,
        diffusion_schedule_path=schedule_path,
        denoiser_config_path=denoiser_config_path,
        length_predictor_config_path=length_config_path,
        final_train_loss=final_train_loss,
        validation_denoising_loss=validation["denoising_loss"],
        validation_length_loss=validation["length_loss"],
        best_metric_value=validation["denoising_loss"],
        completed_epoch=max_epochs - 1,
        global_step=global_step,
        runtime_trace=dict(reader_runtime_trace),
    )


def _generate_validation_latents(
    *,
    context: ModelStageExecutionContext,
    config: LatentDiffusionConfig,
    progress=None,
) -> LatentGenerationResult:
    train_stage = _require_stage_result(context, ModelStageKind.TRAIN_DENOISER)
    cache_stage = _require_stage_result(context, ModelStageKind.CACHE_LATENTS)
    target_stage = _require_stage_result(context, ModelStageKind.DEFINE_LATENT_TARGET)
    checkpoint_path = Path(str(train_stage.metadata["best_checkpoint_path"]))
    _require_file(checkpoint_path, "selected latent denoiser checkpoint")
    target_spec = read_latent_target_spec_json(Path(str(target_stage.metadata["target_spec_path"])))
    stats = read_bfh_standardization_stats_json(Path(str(cache_stage.metadata["standardization_stats_path"])))
    raw_autoencoder_checkpoint = cache_stage.metadata.get("autoencoder_checkpoint_path")
    autoencoder_checkpoint_path = (
        None if raw_autoencoder_checkpoint is None else Path(str(raw_autoencoder_checkpoint))
    )
    schedule = _read_diffusion_schedule_json(Path(str(train_stage.metadata["diffusion_schedule_path"])))
    checkpoint = _load_checkpoint(checkpoint_path)
    text_encoder = build_text_encoder(config.text_encoder)
    denoiser, length_predictor = _models_from_checkpoint(
        checkpoint,
        config=config,
        latent_dim=target_spec.latent_dim,
        text_encoder_output_dim=text_encoder.output_dim,
    )
    sampling_steps, sampling_steps_source = _effective_sampling_steps(
        context.loaded_config.effective_config,
        configured_steps=config.diffusion.sampling_steps,
    )
    caps = _run_mode_caps(context.loaded_config.effective_config)
    validation_source_surface = load_model_data_surface(
        Path(str(cache_stage.metadata["validation_source_surface_path"]))
    )
    source_count = validation_source_surface.metadata.loaded_sample_count
    if caps["limit_prediction_samples"] is not None:
        source_count = min(source_count, int(caps["limit_prediction_samples"]))
    if source_count <= 0:
        raise LatentDiffusionError("generate requires at least one validation source sample.")
    root = _generation_root(context)
    latent_root = root / "predicted_latent_samples" / "val"
    decoded_paths = diagnostic_generated_pose_paths(root / "decoded_pose_intermediates" / "val")
    validate_diagnostic_generated_pose_paths(decoded_paths)
    traces_path = root / "sampling_traces.jsonl"
    run_seed = _run_seed(context.loaded_config.effective_config)
    device = _resolve_device(config.training.device)
    predicted_counts: dict[str, int] = {}
    manifest_path = root / "predicted_latent_manifest_val.jsonl"
    total_candidates = source_count * config.generation.candidate_count
    generated_sample_count = 0
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    latent_root.mkdir(parents=True, exist_ok=True)
    decoded_paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    decoded_paths.samples_root.mkdir(parents=True, exist_ok=True)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as latent_manifest_handle:
        with decoded_paths.manifest_path.open("w", encoding="utf-8") as decoded_manifest_handle:
            with traces_path.open("w", encoding="utf-8") as trace_handle:
                with progress.task(
                    operation="generate_validation_candidates",
                    label="latent_diffusion generate validation candidates",
                    unit="candidate",
                    total=total_candidates,
                    total_semantics="validation latent generation candidates",
                    allowed_counters=("frames",),
                ) as task:
                    for source in _iter_limited_latent_sources(
                        validation_source_surface,
                        limit=caps["limit_prediction_samples"],
                    ):
                        for candidate_index in range(config.generation.candidate_count):
                            seed = _sample_seed(
                                run_seed=run_seed,
                                sample_id=source.sample_id,
                                candidate_index=candidate_index,
                                seed_policy=config.generation.seed_policy,
                            )
                            generated_result = _generate_one_decoded_sample(
                                source=source,
                                config=config,
                                target_spec=target_spec,
                                stats=stats,
                                schedule=schedule,
                                denoiser=denoiser,
                                length_predictor=length_predictor,
                                run_name=context.request.run_name,
                                seed=seed,
                                candidate_index=candidate_index,
                                device=device,
                                autoencoder_checkpoint_path=autoencoder_checkpoint_path,
                                sampling_steps=sampling_steps,
                            )
                            latent_values = generated_result.latent_values
                            sequence = LatentSequence(
                                schema_version=LATENT_SEQUENCE_SCHEMA_VERSION,
                                sample_id=source.sample_id,
                                source_sentence_name=source.source_sentence_name,
                                split=source.split,
                                values=latent_values,
                                validity_mask=np.ones(latent_values.shape, dtype=np.bool_),
                                frame_count=generated_result.predicted_frame_count,
                                latent_count=latent_values.shape[0],
                                latent_dim=latent_values.shape[1],
                                target_spec=target_spec,
                            )
                            token = _sample_candidate_token(source.sample_id, candidate_index)
                            latent_path = latent_root / f"{token}.npz"
                            write_latent_sequence_npz(latent_path, sequence)
                            latent_entry = LatentManifestEntry(
                                schema_version=LATENT_MANIFEST_SCHEMA_VERSION,
                                sample_id=source.sample_id,
                                source_sentence_name=source.source_sentence_name,
                                split=source.split,
                                latent_path=latent_path,
                                frame_count=sequence.frame_count,
                                latent_count=sequence.latent_count,
                                latent_dim=sequence.latent_dim,
                                target_type=target_spec.target_type,
                                temporal_granularity=target_spec.temporal_granularity,
                                window_size=target_spec.window_size,
                                stride=target_spec.stride,
                                issues=(),
                            )
                            latent_manifest_handle.write(
                                json.dumps(latent_entry.to_dict(), sort_keys=True, separators=(",", ":"))
                                + "\n"
                            )
                            decoded_path, decoded_entry = _write_one_generated_sample_explicit(
                                generated_result.generated,
                                manifest_path=decoded_paths.manifest_path,
                                samples_root=decoded_paths.samples_root,
                                payload_ref_root=decoded_paths.payload_ref_root,
                            )
                            decoded_manifest_handle.write(
                                json.dumps(
                                    generated_manifest_entry_to_record(decoded_entry),
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            issues = validate_generated_pose_manifest_entries(
                                (decoded_entry,),
                                expected_split=context.request.validation_split,
                            )
                            if issues:
                                raise LatentDiffusionError(
                                    "Invalid generated-pose manifest entry: "
                                    + "; ".join(issue.message for issue in issues)
                                )
                            trace_handle.write(
                                json.dumps(generated_result.trace.to_dict(), sort_keys=True, separators=(",", ":"))
                                + "\n"
                            )
                            predicted_counts[f"{source.sample_id}:g{candidate_index}"] = (
                                generated_result.predicted_frame_count
                            )
                            generated_sample_count += 1
                            task.advance(1, counters={"frames": generated_result.predicted_frame_count})
    return LatentGenerationResult(
        predicted_latent_manifest_path=manifest_path,
        sampling_traces_path=traces_path,
        decoded_generated_pose_manifest_path=decoded_paths.manifest_path,
        generated_sample_count=generated_sample_count,
        predicted_frame_counts=predicted_counts,
        configured_sampling_steps=config.diffusion.sampling_steps,
        effective_sampling_steps=sampling_steps,
        sampling_steps_source=sampling_steps_source,
    )


def _generate_one_decoded_sample(
    *,
    source: LatentSourceSample,
    config: LatentDiffusionConfig,
    target_spec: LatentTargetSpec,
    stats: BfhStandardizationStats,
    schedule: DiffusionSchedule,
    denoiser: LatentDenoiser,
    length_predictor: LatentLengthPredictor,
    run_name: str,
    seed: int,
    candidate_index: int,
    device: torch.device,
    autoencoder_checkpoint_path: Path | None,
    sampling_steps: int | None = None,
):
    latents, trace, predicted_count = _sample_latent_values(
        source=source,
        config=config,
        target_spec=target_spec,
        schedule=schedule,
        denoiser=denoiser,
        length_predictor=length_predictor,
        seed=seed,
        candidate_index=candidate_index,
        device=device,
        sampling_steps=sampling_steps,
    )
    pose = _decode_latents_to_pose(
        latents,
        target_spec=target_spec,
        stats=stats,
        decoded_frame_count=predicted_count,
        config=config,
        autoencoder_checkpoint_path=autoencoder_checkpoint_path,
        device=device,
    )
    identity = GeneratedPoseSourceIdentity(
        canonical_id=LATENT_DIFFUSION_CANONICAL_ID,
        phase_number=LATENT_DIFFUSION_PHASE_NUMBER,
        research_role=LATENT_DIFFUSION_RESEARCH_ROLE,
        run_name=run_name,
        text=source.text,
        source_video_id=source.source_video_id,
        source_sentence_id=source.source_sentence_id,
        source_sentence_name=source.source_sentence_name,
        reference_payload_ref=source.reference_payload_ref,
        num_candidates_for_sample=config.generation.candidate_count,
        producer_type="model",
        generation_mode=config.generation.generation_mode,
        length_policy=config.length.policy,
        confidence_policy=config.generation.confidence_policy,
        seed=seed,
    )
    generated = generated_pose_sample_from_bfh_arrays(
        sample_id=source.sample_id,
        sentence_name=source.source_sentence_name,
        split=source.split,
        generation_index=candidate_index,
        arrays=pose,
        producer_key=LATENT_DIFFUSION_MODEL_KEY,
        producer_stage="latent_diffusion_generate",
        identity=identity,
    )
    return LatentDecodedSampleResult(
        generated=generated,
        latent_values=latents,
        trace=trace,
        predicted_frame_count=predicted_count,
        sampled_latent_count=int(latents.shape[0]),
        decoded_frame_count=pose.frame_count,
    )


def _write_one_generated_sample_explicit(
    sample: GeneratedPoseSample,
    *,
    manifest_path: Path,
    samples_root: Path,
    payload_ref_root: Path,
):
    payload_path = samples_root / _generated_pose_sample_filename(
        sample.sample_id,
        sample.generation_index,
    )
    payload_ref = payload_path.relative_to(payload_ref_root).as_posix()
    entry = generated_manifest_entry_from_sample(
        sample,
        generated_payload_ref=payload_ref,
    )
    write_generated_pose_payload(payload_path, sample)
    return payload_path, entry


def _write_generated_pose_payloads_incremental(
    *,
    topology,
    payload_paths: Iterable[Path],
    producer_key: str,
    run_name: str,
    split: SampleSplit,
    on_sample_written,
) -> GeneratedPoseStreamWriteResult:
    resolved_split = SampleSplit(split)
    manifest_path = topology.evaluations.generated_pose_manifest(
        producer_key,
        run_name,
        resolved_split,
    ).path
    sample_root = topology.evaluations.generated_pose_samples_root(
        producer_key,
        run_name,
        resolved_split,
    ).path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    preview_entries = []
    preview_payload_paths = []
    seen: set[tuple[str, str, int]] = set()
    sample_count = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for source_path in payload_paths:
            sample = load_generated_pose_payload(source_path)
            if sample.producer_key != producer_key or sample.run_name != run_name:
                raise LatentDiffusionError("sample producer_key/run_name does not match request.")
            if sample.split is not resolved_split:
                raise LatentDiffusionError("sample split does not match request.")
            payload_path = topology.evaluations.generated_pose_sample_file(
                producer_key,
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
                raise LatentDiffusionError("Invalid generated-pose manifest: duplicate manifest key.")
            seen.add(key)
            if issues:
                raise LatentDiffusionError(
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
            on_sample_written()
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


def _write_generated_pose_payloads_from_manifest_incremental(
    *,
    topology,
    manifest_path: Path,
    producer_key: str,
    run_name: str,
    split: SampleSplit,
    on_sample_written,
) -> GeneratedPoseStreamWriteResult:
    resolved_split = SampleSplit(split)
    source_manifest = Path(manifest_path)
    payload_ref_root = source_manifest.parent.parent
    def _payload_paths():
        with source_manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = generated_manifest_entry_from_record(json.loads(line))
                issues = validate_generated_pose_manifest_entries(
                    (entry,),
                    expected_split=resolved_split,
                )
                if issues:
                    raise LatentDiffusionError(
                        "Invalid generated-pose manifest: "
                        + "; ".join(issue.message for issue in issues)
                    )
                yield payload_ref_root / entry.generated_payload_ref

    return _write_generated_pose_payloads_incremental(
        topology=topology,
        payload_paths=_payload_paths(),
        producer_key=producer_key,
        run_name=run_name,
        split=resolved_split,
        on_sample_written=on_sample_written,
    )


def _generated_pose_sample_filename(sample_id: str, generation_index: int) -> str:
    token = str(sample_id)
    if not token.strip() or token in {".", ".."} or "/" in token or "\\" in token or "." in token:
        raise LatentDiffusionError("generated-pose sample_id must be a concrete safe path token.")
    return f"{token}__g{int(generation_index)}.npz"


def _sample_latent_values(
    *,
    source: LatentSourceSample,
    config: LatentDiffusionConfig,
    target_spec: LatentTargetSpec,
    schedule: DiffusionSchedule,
    denoiser: LatentDenoiser,
    length_predictor: LatentLengthPredictor,
    seed: int,
    candidate_index: int,
    device: torch.device,
    sampling_steps: int | None = None,
) -> tuple[np.ndarray, LatentSamplingTrace, int]:
    text_encoder = build_text_encoder(config.text_encoder)
    embedding = text_encoder.encode_one(
        source.text,
        sentence_name=source.source_sentence_name,
    ).pooled_embedding
    text = torch.as_tensor(
        np.array(embedding, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    length_predictor = length_predictor.to(device)
    length_predictor.eval()
    with torch.no_grad():
        predicted = float(length_predictor(text).detach().cpu().item())
    frame_count = clamp_predicted_frame_count(
        predicted,
        config=config.length,
        max_positions=(
            config.denoiser.max_positions
            if target_spec.target_type == LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME
            else 2_147_483_647
        ),
    )
    latent_count = expected_latent_count_for_frame_count(
        frame_count=frame_count,
        target_spec=target_spec,
    )
    sampling_config = LatentSamplingConfig(
        sampler=config.diffusion.sampler,
        sampling_steps=(
            config.diffusion.sampling_steps if sampling_steps is None else sampling_steps
        ),
        seed=seed,
        candidate_index=candidate_index,
    )
    if sampling_config.sampling_steps > schedule.timesteps:
        raise LatentDiffusionError(
            "sampling_steps must be less than or equal to restored diffusion schedule timesteps."
        )
    sample, trace = sample_latent_sequence(
        denoiser=denoiser,
        schedule=schedule,
        text_embedding=text,
        length=latent_count,
        latent_dim=target_spec.latent_dim,
        seed=sampling_config.seed,
        candidate_index=sampling_config.candidate_index,
        sampling_steps=sampling_config.sampling_steps,
        device=device,
    )
    return sample, trace, frame_count


def _decode_latents_to_pose(
    values: np.ndarray,
    *,
    target_spec: LatentTargetSpec,
    stats: BfhStandardizationStats,
    decoded_frame_count: int,
    config: LatentDiffusionConfig,
    autoencoder_checkpoint_path: Path | None,
    device: torch.device,
):
    latent_count = int(values.shape[0])
    if target_spec.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
        expected_count = expected_latent_count_for_frame_count(
            frame_count=decoded_frame_count,
            target_spec=target_spec,
        )
        if latent_count != expected_count:
            raise LatentDiffusionError(
                "latent_count must equal len(temporal_window_starts(frame_count, spec)) "
                "for learned_bfh_window_latent."
            )
        if config.latent_autoencoder is None:
            raise LatentDiffusionError(
                "latent_autoencoder config is required when "
                "latent_target.target_type='learned_bfh_window_latent'."
            )
        if autoencoder_checkpoint_path is None:
            raise LatentDiffusionError(
                "autoencoder checkpoint is required to decode learned_bfh_window_latent."
            )
        input_dim = target_spec.window_size * target_spec.base_feature_dim
        autoencoder = load_autoencoder_checkpoint(
            autoencoder_checkpoint_path,
            config=config.latent_autoencoder,
            input_dim=input_dim,
        ).to(device)
        decoded_windows = decode_window_latents(
            model=autoencoder,
            latents=values,
            batch_size=config.latent_autoencoder.batch_size,
            device=device,
        )
        expected_shape = (latent_count, input_dim)
        if decoded_windows.shape != expected_shape:
            raise LatentDiffusionError(
                f"decoded window values must have shape {expected_shape}."
            )
        window_values = decoded_windows.reshape(
            latent_count,
            target_spec.window_size,
            target_spec.layout.total_joint_count,
            target_spec.layout.coordinate_dimensions,
        )
        starts = temporal_window_starts(
            frame_count=decoded_frame_count,
            spec=target_spec.temporal_window_spec(),
        )
        real_frame_mask = np.zeros((latent_count, target_spec.window_size), dtype=np.bool_)
        source_frame_indices = np.full(
            (latent_count, target_spec.window_size),
            -1,
            dtype=np.int64,
        )
        for window_index, start in enumerate(starts):
            for offset in range(target_spec.window_size):
                source_index = int(start + offset)
                if source_index >= decoded_frame_count:
                    continue
                real_frame_mask[window_index, offset] = True
                source_frame_indices[window_index, offset] = source_index
        confidence = real_frame_mask[..., None] * np.ones(
            (
                latent_count,
                target_spec.window_size,
                target_spec.layout.total_joint_count,
            ),
            dtype=np.float32,
        )
        frame_validity = real_frame_mask.copy()
        validity = real_frame_mask[..., None] * np.ones_like(confidence, dtype=np.bool_)
        window_values = np.where(real_frame_mask[..., None, None], window_values, 0.0)
        window_sequence = BfhPoseWindowSequence(
            schema_version=BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION,
            spec=target_spec.temporal_window_spec(),
            layout=target_spec.layout,
            source_sample_id=None,
            source_frame_count=decoded_frame_count,
            values=window_values,
            validity_mask=validity,
            confidence_values=confidence,
            frame_validity_mask=frame_validity,
            real_frame_mask=real_frame_mask,
            source_frame_indices=source_frame_indices,
        )
        merged = merge_bfh_pose_windows(window_sequence).vectorized_pose
        return bfh_pose_arrays_from_vectorized(invert_bfh_standardization(merged, stats))

    if target_spec.target_type != LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
        raise LatentDiffusionError(
            "latent_target.target_type must be one of "
            "{'standardized_bfh_frame', 'learned_bfh_window_latent'}."
        )
    frame_count = decoded_frame_count
    if latent_count != frame_count:
        raise LatentDiffusionError("latent_count must equal frame_count.")
    restored = unflatten_bfh_pose_values(
        values,
        layout=target_spec.layout,
        frame_count=frame_count,
    )
    vectorized = BfhVectorizedPose(
        layout=target_spec.layout,
        values=restored,
        validity_mask=np.ones(
            (frame_count, target_spec.layout.total_joint_count),
            dtype=np.bool_,
        ),
        frame_validity_mask=np.ones((frame_count,), dtype=np.bool_),
        confidence_values=np.ones(
            (frame_count, target_spec.layout.total_joint_count),
            dtype=np.float32,
        ),
        frame_count=frame_count,
        source_sample_id=None,
    )
    return bfh_pose_arrays_from_vectorized(invert_bfh_standardization(vectorized, stats))


def _sequence_losses(
    sequence: LatentSequence,
    sources: Mapping[str, LatentSourceSample],
    *,
    text_encoder,
    schedule: DiffusionSchedule,
    denoiser: LatentDenoiser,
    length_predictor: LatentLengthPredictor,
    device: torch.device,
    config: LatentDiffusionConfig,
    semantic_config: SemanticConsistencyObjectiveConfig | None = None,
    semantic_autoencoder=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
) -> Mapping[str, torch.Tensor]:
    source = sources.get(sequence.sample_id)
    if source is None:
        raise LatentDiffusionError(f"missing latent source for sample_id={sequence.sample_id!r}.")
    x0 = torch.as_tensor(
        np.array(sequence.values, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.as_tensor(
        np.array(sequence.validity_mask, dtype=np.bool_, copy=True),
        dtype=torch.bool,
        device=device,
    )
    noise = torch.randn_like(x0)
    timesteps = torch.randint(
        low=0,
        high=schedule.timesteps,
        size=(sequence.latent_count,),
        device=device,
        dtype=torch.long,
    )
    xt = q_sample(x0=x0, t=timesteps, noise=noise, schedule=schedule)
    embedding = text_encoder.encode_one(
        source.text,
        sentence_name=source.source_sentence_name,
    ).pooled_embedding
    text = torch.as_tensor(
        np.array(embedding, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    ).repeat(
        sequence.latent_count,
        1,
    )
    positions = torch.arange(sequence.latent_count, dtype=torch.long, device=device)
    resolved_precision = precision_policy or resolve_precision_policy("auto", device)
    with autocast_context(resolved_precision):
        predicted = denoiser(xt, timesteps, text, positions)
        denoising_loss = compute_denoising_loss(
            predicted_noise=predicted,
            target_noise=noise,
            validity_mask=mask,
        )
        length_embedding = text[:1]
        predicted_length = length_predictor(length_embedding)
        target_length = torch.as_tensor(
            [float(sequence.frame_count)],
            dtype=torch.float32,
            device=device,
        )
        length_loss = torch.nn.functional.mse_loss(predicted_length, target_length)
        if not torch.isfinite(length_loss):
            raise LatentDiffusionError("length loss is not finite.")
        total_loss = denoising_loss + float(config.length.loss_weight) * length_loss
        semantic_loss = None
        if semantic_config is not None:
            alpha_bars = torch.as_tensor(
                np.array(schedule.alpha_bars, copy=True),
                dtype=xt.dtype,
                device=xt.device,
            )[timesteps].unsqueeze(-1)
            predicted_x0 = (xt - torch.sqrt(1.0 - alpha_bars) * predicted) / torch.sqrt(alpha_bars)
            if config.latent_target.target_type == LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
                semantic_pose = predicted_x0.unsqueeze(0)
                semantic_validity = mask.unsqueeze(0)
                semantic_frame_mask = torch.ones(
                    (1, sequence.latent_count),
                    dtype=torch.bool,
                    device=device,
                )
            elif config.latent_target.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
                if semantic_autoencoder is None:
                    raise LatentDiffusionError(
                        "semantic_consistency learned temporal latent training requires a loaded window autoencoder."
                    )
                decoded_windows = semantic_autoencoder.decode(predicted_x0).reshape(
                    sequence.latent_count,
                    config.latent_target.window_size,
                    -1,
                )
                starts = temporal_window_starts(
                    frame_count=sequence.frame_count,
                    spec=TemporalWindowSpec.window(
                        window_size=config.latent_target.window_size,
                        stride=config.latent_target.stride,
                    ),
                )
                if sequence.latent_count != len(starts):
                    raise LatentDiffusionError(
                        "semantic latent_count must equal "
                        "len(temporal_window_starts(frame_count, spec))."
                    )
                semantic_pose = merge_overlapping_windows_torch(
                    decoded_windows,
                    frame_count=sequence.frame_count,
                    window_size=config.latent_target.window_size,
                    stride=config.latent_target.stride,
                    start_indices=starts,
                ).unsqueeze(0)
                source_validity = np.repeat(
                    source.vectorized_pose.validity_mask
                    & source.vectorized_pose.frame_validity_mask[:, None],
                    source.vectorized_pose.layout.coordinate_dimensions,
                    axis=1,
                )
                semantic_validity = torch.as_tensor(
                    source_validity,
                    dtype=torch.bool,
                    device=device,
                ).unsqueeze(0)
                semantic_frame_mask = torch.as_tensor(
                    source.vectorized_pose.frame_validity_mask,
                    dtype=torch.bool,
                    device=device,
                ).unsqueeze(0)
            else:
                raise LatentDiffusionError("unsupported latent target type for semantic training.")
            semantic_result = compute_semantic_training_loss(
                text_embeddings=text[:1],
                pose_values=semantic_pose,
                pose_validity_mask=semantic_validity,
                frame_mask=semantic_frame_mask,
                layout=default_bfh_tensor_layout(),
                config=semantic_config,
            )
            total_loss = total_loss + semantic_result.weighted_semantic_loss
            semantic_loss = semantic_result
    if not torch.isfinite(total_loss):
        raise LatentDiffusionError("total latent diffusion loss is not finite.")
    result = {
        "total_loss": total_loss,
        "denoising_loss": denoising_loss,
        "length_loss": length_loss,
    }
    if semantic_loss is not None:
        result.update(
            {
                "semantic_loss": semantic_loss.semantic_loss,
                "semantic_weighted_loss": semantic_loss.weighted_semantic_loss,
                "semantic_cosine_similarity": semantic_loss.cosine_similarity,
                "semantic_cosine_distance": semantic_loss.cosine_distance,
            }
        )
    return result


def _denoiser_batch_losses(
    batch: LatentDenoiserBatch,
    sources: Mapping[int, LatentSourceSample],
    *,
    text_encoder,
    schedule: DiffusionSchedule,
    denoiser: LatentDenoiser,
    length_predictor: LatentLengthPredictor,
    device: torch.device,
    config: LatentDiffusionConfig,
    semantic_config: SemanticConsistencyObjectiveConfig | None = None,
    semantic_autoencoder=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
) -> Mapping[str, torch.Tensor]:
    del semantic_config, semantic_autoencoder
    x0 = batch.latent_values.to(device=device, dtype=torch.float32)
    mask = batch.latent_mask.to(device=device, dtype=torch.bool)
    source_index = batch.source_index.to(device=device, dtype=torch.long)
    latent_index = batch.latent_index.to(device=device, dtype=torch.long)
    batch_size = int(x0.shape[0])
    source_rows = [sources[int(index)] for index in batch.source_index.detach().cpu().tolist()]
    encoded = text_encoder.encode_batch(
        [source.text for source in source_rows],
        sentence_names=[source.source_sentence_name for source in source_rows],
    )
    text = torch.as_tensor(
        np.stack([item.pooled_embedding for item in encoded]).astype(np.float32, copy=False),
        dtype=torch.float32,
        device=device,
    )
    noise = torch.randn_like(x0)
    timesteps = torch.randint(
        low=0,
        high=schedule.timesteps,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )
    xt = q_sample(x0=x0, t=timesteps, noise=noise, schedule=schedule)
    resolved_precision = precision_policy or resolve_precision_policy("auto", device)
    with autocast_context(resolved_precision):
        predicted = denoiser(xt, timesteps, text, latent_index)
        denoising_loss = compute_denoising_loss(
            predicted_noise=predicted,
            target_noise=noise,
            validity_mask=mask,
        )
        predicted_length = length_predictor(text)
        target_length = torch.as_tensor(
            [float(source.frame_count) for source in source_rows],
            dtype=torch.float32,
            device=device,
        )
        length_loss = torch.nn.functional.mse_loss(predicted_length, target_length)
        if not torch.isfinite(length_loss):
            raise LatentDiffusionError("length loss is not finite.")
        total_loss = denoising_loss + float(config.length.loss_weight) * length_loss
    if not torch.isfinite(total_loss):
        raise LatentDiffusionError("total latent diffusion loss is not finite.")
    return {
        "total_loss": total_loss,
        "denoising_loss": denoising_loss,
        "length_loss": length_loss,
    }


def _validation_batch_losses(
    batches,
    sources: Mapping[int, LatentSourceSample],
    *,
    text_encoder,
    schedule: DiffusionSchedule,
    denoiser: LatentDenoiser,
    length_predictor: LatentLengthPredictor,
    device: torch.device,
    config: LatentDiffusionConfig,
    epoch: int,
    global_step: int,
    metrics: list[TrainingMetricRecord],
    progress=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
    total: int | None = None,
) -> dict[str, float]:
    denoiser.eval()
    length_predictor.eval()
    denoising = []
    length = []
    with torch.no_grad():
        with progress.task(
            operation="validate_denoiser",
            label="latent_diffusion validate denoiser",
            unit="batch",
            total=total,
            total_semantics="latent denoiser validation batches",
            allowed_counters=("loss",),
        ) as task:
            for batch in batches:
                losses = _denoiser_batch_losses(
                    batch,
                    sources,
                    text_encoder=text_encoder,
                    schedule=schedule,
                    denoiser=denoiser,
                    length_predictor=length_predictor,
                    device=device,
                    config=config,
                    precision_policy=precision_policy,
                )
                denoise = _finite_tensor_value(
                    losses["denoising_loss"],
                    "validation_denoising_loss",
                )
                denoising.append(denoise)
                length.append(_finite_tensor_value(losses["length_loss"], "validation_length_loss"))
                task.advance(1, counters={"loss": f"{denoise:.6g}"})
    denoising_loss = _finite_float(float(np.mean(denoising)), "validation_denoising_loss")
    length_loss = _finite_float(float(np.mean(length)), "validation_length_loss")
    metrics.append(
        _metric(
            "val",
            config.checkpoints.selection_metric,
            denoising_loss,
            epoch,
            global_step,
        )
    )
    metrics.append(_metric("val", "validation_length_loss", length_loss, epoch, global_step))
    return {"denoising_loss": denoising_loss, "length_loss": length_loss}


def _validation_losses(
    sequences,
    sources: Mapping[str, LatentSourceSample],
    *,
    text_encoder,
    schedule: DiffusionSchedule,
    denoiser: LatentDenoiser,
    length_predictor: LatentLengthPredictor,
    device: torch.device,
    config: LatentDiffusionConfig,
    epoch: int,
    global_step: int,
    metrics: list[TrainingMetricRecord],
    progress=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
    total: int | None = None,
) -> dict[str, float]:
    denoiser.eval()
    length_predictor.eval()
    denoising = []
    length = []
    with torch.no_grad():
        with progress.task(
            operation="validate_denoiser",
            label="latent_diffusion validate denoiser",
            unit="sequence",
            total=total if total is not None else len(sequences),
            total_semantics="latent denoiser validation sequences",
            allowed_counters=("loss",),
        ) as task:
            for sequence in sequences:
                losses = _sequence_losses(
                    sequence,
                    sources,
                    text_encoder=text_encoder,
                    schedule=schedule,
                    denoiser=denoiser,
                    length_predictor=length_predictor,
                    device=device,
                    config=config,
                    precision_policy=precision_policy,
                )
                denoise = _finite_tensor_value(
                    losses["denoising_loss"],
                    "validation_denoising_loss",
                )
                denoising.append(denoise)
                length.append(_finite_tensor_value(losses["length_loss"], "validation_length_loss"))
                task.advance(1, counters={"loss": f"{denoise:.6g}"})
    denoising_loss = _finite_float(float(np.mean(denoising)), "validation_denoising_loss")
    length_loss = _finite_float(float(np.mean(length)), "validation_length_loss")
    metrics.append(
        _metric(
            "val",
            config.checkpoints.selection_metric,
            denoising_loss,
            epoch,
            global_step,
        )
    )
    metrics.append(_metric("val", "validation_length_loss", length_loss, epoch, global_step))
    return {"denoising_loss": denoising_loss, "length_loss": length_loss}


def _models_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    config: LatentDiffusionConfig,
    latent_dim: int,
    text_encoder_output_dim: int,
) -> tuple[LatentDenoiser, LatentLengthPredictor]:
    if checkpoint.get("schema_version") != "t2sp-latent-diffusion-checkpoint-v1":
        raise LatentDiffusionError("latent denoiser checkpoint schema_version is unsupported.")
    _require_v2_temporal_checkpoint_fields(checkpoint, config=config)
    _require_checkpoint_field_match(
        checkpoint,
        field_name="target_type",
        expected=config.latent_target.target_type,
    )
    _require_checkpoint_field_match(
        checkpoint,
        field_name="temporal_granularity",
        expected=config.latent_target.temporal_granularity,
    )
    _require_checkpoint_field_match(
        checkpoint,
        field_name="window_size",
        expected=config.latent_target.window_size,
    )
    _require_checkpoint_field_match(
        checkpoint,
        field_name="stride",
        expected=config.latent_target.stride,
    )
    _require_checkpoint_field_match(
        checkpoint,
        field_name="latent_dim",
        expected=latent_dim,
    )
    text_dim = int(checkpoint["text_embedding_dim"])
    if text_dim != text_encoder_output_dim:
        raise LatentDiffusionError(
            f"checkpoint text_embedding_dim={text_dim} but current text encoder "
            f"output_dim={text_encoder_output_dim}. Use the original text encoder config or retrain."
        )
    denoiser = build_latent_denoiser(
        latent_dim=latent_dim,
        text_embedding_dim=text_dim,
        config=config,
    )
    length_predictor = build_latent_length_predictor(text_embedding_dim=text_dim, config=config)
    denoiser.load_state_dict(cast(Any, checkpoint["denoiser_state_dict"]))
    length_predictor.load_state_dict(cast(Any, checkpoint["length_predictor_state_dict"]))
    return denoiser, length_predictor


def _require_checkpoint_field_match(
    checkpoint: Mapping[str, object],
    *,
    field_name: str,
    expected: object,
) -> None:
    if field_name not in checkpoint:
        return
    observed = checkpoint[field_name]
    if observed != expected:
        raise LatentDiffusionError(
            f"checkpoint {field_name}={observed!r} but current config requires "
            f"{field_name}={expected!r}. Use the original config or retrain."
        )


def _require_v2_temporal_checkpoint_fields(
    checkpoint: Mapping[str, object],
    *,
    config: LatentDiffusionConfig,
) -> None:
    if config.latent_target.target_type != LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
        return
    for field_name in (
        "target_type",
        "temporal_granularity",
        "window_size",
        "stride",
        "latent_dim",
    ):
        if field_name not in checkpoint:
            raise LatentDiffusionError(
                f"latent_diffusion v2 checkpoint is missing required field '{field_name}'."
            )


def _verify_restored_model_configs(
    *,
    denoiser_config_path: Path,
    length_predictor_config_path: Path,
    config: LatentDiffusionConfig,
    checkpoint: Mapping[str, object],
) -> None:
    denoiser_record = _read_support_config_json(
        denoiser_config_path,
        expected_schema_version="t2sp-latent-denoiser-config-v1",
        label="latent_diffusion denoiser_config.json",
    )
    length_record = _read_support_config_json(
        length_predictor_config_path,
        expected_schema_version="t2sp-latent-length-predictor-config-v1",
        label="latent_diffusion length_predictor_config.json",
    )
    expected_denoiser = {
        "schema_version": "t2sp-latent-denoiser-config-v1",
        **config.denoiser.to_dict(),
    }
    expected_length = {
        "schema_version": "t2sp-latent-length-predictor-config-v1",
        **config.length.to_dict(),
    }
    _require_exact_support_config(
        observed=denoiser_record,
        expected=expected_denoiser,
        label="latent_diffusion denoiser_config.json",
    )
    _require_exact_support_config(
        observed=length_record,
        expected=expected_length,
        label="latent_diffusion length_predictor_config.json",
    )
    checkpoint_config = checkpoint.get("provider_config")
    if isinstance(checkpoint_config, Mapping):
        checkpoint_denoiser = _mapping_from_checkpoint_config(
            checkpoint_config,
            "denoiser",
        )
        checkpoint_length = _mapping_from_checkpoint_config(checkpoint_config, "length")
        _require_exact_support_config(
            observed=denoiser_record,
            expected={
                "schema_version": "t2sp-latent-denoiser-config-v1",
                **dict(checkpoint_denoiser),
            },
            label="latent_diffusion denoiser_config.json versus checkpoint provider_config",
        )
        _require_exact_support_config(
            observed=length_record,
            expected={
                "schema_version": "t2sp-latent-length-predictor-config-v1",
                **dict(checkpoint_length),
            },
            label=(
                "latent_diffusion length_predictor_config.json versus checkpoint "
                "provider_config"
            ),
        )


def _source_from_manifest_sample(sample: ModelingManifestSample) -> LatentSourceSample:
    from text_to_sign_production.modeling.backbones.bfh_vectorization import (
        vectorize_bfh_pose_arrays,
    )

    vectorized = vectorize_bfh_pose_arrays(sample.pose, sample_id=sample.sample_id)
    return LatentSourceSample(
        sample_id=sample.sample_id,
        source_sentence_name=sample.source_sentence_name,
        text=sample.text,
        source_video_id=sample.source_video_id,
        source_sentence_id=sample.source_sentence_id,
        reference_payload_ref=sample.payload_ref,
        split=sample.split,
        frame_count=sample.frame_count,
        vectorized_pose=vectorized,
    )


def _required_runtime_support_paths(
    topology,
    run_name: str,
    *,
    config: LatentDiffusionConfig,
) -> dict[str, Path]:
    paths = {
        "target_spec": topology.models.model_intermediate_root(
            LATENT_DIFFUSION_MODEL_KEY,
            run_name,
            "latent",
        ).path
        / "target_spec.json",
        "standardization_stats": topology.models.model_intermediate_root(
            LATENT_DIFFUSION_MODEL_KEY,
            run_name,
            "latent",
        ).path
        / "standardization_stats.json",
        "diffusion_schedule": topology.models.model_intermediate_root(
            LATENT_DIFFUSION_MODEL_KEY,
            run_name,
            "denoiser",
        ).path
        / "diffusion_schedule.json",
        "denoiser_config": topology.models.model_intermediate_root(
            LATENT_DIFFUSION_MODEL_KEY,
            run_name,
            "denoiser",
        ).path
        / "denoiser_config.json",
        "length_predictor_config": topology.models.model_intermediate_root(
            LATENT_DIFFUSION_MODEL_KEY,
            run_name,
            "denoiser",
        ).path
        / "length_predictor_config.json",
    }
    if config.latent_target.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
        paths["autoencoder_checkpoint"] = topology.models.model_intermediate_root(
            LATENT_DIFFUSION_MODEL_KEY,
            run_name,
            "latent",
        ).path / "autoencoder" / "window_autoencoder.pt"
        paths["autoencoder_config"] = topology.models.model_intermediate_root(
            LATENT_DIFFUSION_MODEL_KEY,
            run_name,
            "latent",
        ).path / "autoencoder" / "autoencoder_config.json"
    for label, path in paths.items():
        _require_file(path, f"latent_diffusion {label}")
    return paths


def _effective_config_payload(
    *,
    config: LatentDiffusionConfig,
    request: ModelRunRequest,
    source_path: Path,
) -> dict[str, object]:
    (
        config,
        dataloader_applied,
        dataloader_not_applicable,
        dataloader_unsupported,
    ) = _apply_latent_dataloader_overrides(config, request)
    config, active_overrides = _apply_latent_active_overrides(config, request)
    policy = resolve_model_run_mode_policy(request.run_mode)
    overrides = {
        "limit_train_samples": policy.limit_train_samples,
        "limit_validation_samples": policy.limit_validation_samples,
        "limit_prediction_samples": policy.limit_prediction_samples,
        "max_epochs": policy.max_epochs,
        "batch_size": policy.batch_size,
    }
    run_seed = request.seed if request.seed is not None else config.training.seed
    if run_seed is None:
        run_seed = 0
    provider_payload = config.to_dict()
    text_encoder_payload = dict(provider_payload["text_encoder"])
    text_encoder_payload["output_dim"] = resolve_text_encoder_output_dim(config.text_encoder)
    provider_payload["text_encoder"] = text_encoder_payload
    training_payload = provider_payload.get("training", {})
    candidate_overrides = provider_candidate_overrides_for_profile(
        request.compute_profile,
        provider_key=LATENT_DIFFUSION_MODEL_KEY,
    )
    candidate_allowed = {"denoiser_batch_size"}
    candidate_unknown = set(candidate_overrides) - candidate_allowed
    application = provider_compute_profile_application(
        compute_profile=request.compute_profile,
        provider_key=LATENT_DIFFUSION_MODEL_KEY,
        run_mode=request.run_mode.value,
        active_applied=active_overrides,
        dataloader_applied=dataloader_applied,
        dataloader_not_applicable=dataloader_not_applicable,
        dataloader_unsupported=dataloader_unsupported,
        candidates_applicable={
            key: candidate_overrides[key]
            for key in candidate_overrides
            if key in candidate_allowed
        },
        candidates_unsupported={
            key: "latent_diffusion current target calibrates only denoiser_batch_size."
            for key in candidate_unknown
        },
        telemetry_required_fields=("denoiser_batch_size", "num_workers"),
        calibration_candidate_keys=("denoiser_batch_size",),
        calibration_override_targets={
            "denoiser_batch_size": "training.denoiser_batch_size",
        },
    ).to_dict()
    return {
        "schema_version": "t2sp-latent-diffusion-effective-config-v1",
        "model_key": request.model_key.value,
        "run_name": request.run_name,
        "source_path": str(source_path),
        "manifest_family": request.manifest_family.family_id,
        "run_mode": request.run_mode.value,
        "compute_profile": dict(request.compute_profile),
        "compute_profile_application": application,
        "runtime_truth_contract": _latent_diffusion_runtime_truth_contract(),
        "auxiliary_objectives": [
            objective.value for objective in request.auxiliary_objectives
        ],
        "objective_config_paths": {
            objective.value: str(path)
            for objective, path in request.objective_config_paths.items()
        },
        "compute_profile_active_overrides": dict(active_overrides),
        "compute_profile_dataloader_overrides": dict(dataloader_applied),
        "run_mode_policy": policy.to_dict(),
        "run_mode_overrides": overrides,
        "training": {
            "denoiser_batch_size": (
                training_payload.get("batch_size")
                if isinstance(training_payload, Mapping)
                else None
            ),
            "num_workers": (
                training_payload.get("num_workers")
                if isinstance(training_payload, Mapping)
                else None
            ),
        },
        "latent_autoencoder": {
            "active": config.latent_autoencoder is not None,
            "reason": (
                "configured latent_autoencoder is active"
                if config.latent_autoencoder is not None
                else f"latent_target.target_type={config.latent_target.target_type}"
            ),
        },
        "generation": {
            "batch_size": None,
            "batching_supported": False,
            "batching_note": (
                "latent_diffusion generation currently samples per sample; "
                "generation_batch_size is intentionally not an active override."
            ),
        },
        "calibration_override_bindings": {
            "denoiser_batch_size": "provider_config.training.batch_size",
        },
        "provider_config": provider_payload,
        "seed_policy": {
            "request_seed": request.seed,
            "config_training_seed": config.training.seed,
            "run_seed": run_seed,
            "generation_seed_policy": config.generation.seed_policy,
            "formula": (
                "sha256(f'{run_seed}:{seed_policy}:{sample_id}:{candidate_index}') "
                "mod 2147483647"
            ),
        },
    }


def _latent_diffusion_runtime_truth_contract() -> dict[str, object]:
    return {
        "schema_version": "model.runtime_truth_contract.v1",
        "provider_key": LATENT_DIFFUSION_MODEL_KEY,
        "required_runtime_evidence": {
            "num_workers": [
                "surface_reader_num_workers_used",
                "surface_reader_worker_mode",
            ],
        },
    }


def _apply_latent_active_overrides(
    config: LatentDiffusionConfig,
    request: ModelRunRequest,
) -> tuple[LatentDiffusionConfig, Mapping[str, object]]:
    active = provider_active_overrides(
        request.compute_profile,
        provider_key=LATENT_DIFFUSION_MODEL_KEY,
        run_mode=request.run_mode.value,
    )
    if not active:
        return config, {}

    allowed = {"denoiser_batch_size"}
    unknown = set(active) - allowed
    if unknown:
        raise LatentDiffusionError(
            "latent_diffusion compute profile active overrides contain unsupported keys: "
            f"{sorted(unknown)}."
        )

    updated = config

    if "denoiser_batch_size" in active:
        value = active["denoiser_batch_size"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise LatentDiffusionError(
                "latent_diffusion denoiser_batch_size active override must be positive."
            )
        updated = replace(
            updated,
            training=replace(updated.training, batch_size=value),
        )

    return updated, dict(active)


def _apply_latent_dataloader_overrides(
    config: LatentDiffusionConfig,
    request: ModelRunRequest,
) -> tuple[
    LatentDiffusionConfig,
    Mapping[str, object],
    Mapping[str, str],
    Mapping[str, str],
]:
    requested = compute_profile_dataloader_section(request.compute_profile)
    if not requested:
        return config, {}, {}, {}
    applied: dict[str, object] = {}
    not_applicable: dict[str, str] = {}
    unsupported: dict[str, str] = {}
    updated = config
    allowed = {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "materialization_workers",
    }
    for key, value in requested.items():
        if key not in allowed:
            unsupported[key] = "latent_diffusion does not recognize this dataloader profile key."
            continue
        if key == "num_workers":
            if value is None:
                not_applicable[key] = "compute profile does not request a num_workers override."
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise LatentDiffusionError(
                    "latent_diffusion dataloader.num_workers must be non-negative."
                )
            updated = replace(
                updated,
                training=replace(updated.training, num_workers=value),
            )
            applied[key] = value
            continue
        not_applicable[key] = (
            "latent_diffusion currently wires only num_workers into runtime dataloaders."
        )
    return updated, applied, not_applicable, unsupported


def _latent_diffusion_performance_metadata(
    config: LatentDiffusionConfig,
    *,
    runtime_trace: Mapping[str, object] | None = None,
    stage: str | None = None,
    configured_sampling_steps: int | None = None,
    effective_sampling_steps: int | None = None,
    candidate_count: int | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "denoiser_batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "configured_sampling_steps": (
            config.diffusion.sampling_steps
            if configured_sampling_steps is None
            else configured_sampling_steps
        ),
        "candidate_count": (
            config.generation.candidate_count
            if candidate_count is None
            else candidate_count
        ),
        "latent_autoencoder_active": config.latent_autoencoder is not None,
    }

    if effective_sampling_steps is not None:
        metadata["effective_sampling_steps"] = effective_sampling_steps

    if stage is not None:
        metadata["latent_stage"] = stage

    if config.latent_autoencoder is not None:
        metadata["autoencoder_batch_size"] = config.latent_autoencoder.batch_size

    if runtime_trace is not None:
        for key in (
            "surface_reader_num_workers_used",
            "surface_reader_worker_mode",
            "surface_reader_prefetch_factor_used",
            "surface_reader_persistent_workers_used",
        ):
            value = runtime_trace.get(key)
            if value is not None:
                metadata[key] = value

    return metadata


def _config_from_loaded(loaded_config: ModelProviderLoadedConfig) -> LatentDiffusionConfig:
    raw = loaded_config.effective_config.get("provider_config")
    if not isinstance(raw, Mapping):
        raise ModelStageExecutionError("latent_diffusion effective_config is missing provider_config.")
    return latent_diffusion_config_from_mapping(raw)


def _standardization_metadata(stats: BfhStandardizationStats) -> dict[str, object]:
    return {
        "standardization_missing_observation_policy": stats.missing_observation_policy,
        "standardization_zero_observation_coordinate_count": (
            stats.zero_observation_coordinate_count
        ),
        "standardization_channel_fallback_coordinate_count": (
            stats.channel_fallback_coordinate_count
        ),
        "standardization_global_fallback_coordinate_count": (
            stats.global_fallback_coordinate_count
        ),
        "standardization_identity_fallback_coordinate_count": (
            stats.identity_fallback_coordinate_count
        ),
        "standardization_sample_count": stats.sample_count,
        "standardization_frame_count": stats.frame_count,
        "standardization_fallback_summary_by_channel": {
            channel: dict(summary)
            for channel, summary in stats.fallback_summary_by_channel.items()
        },
    }


def _stable_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _latent_data_surface_root(
    context: ModelStageExecutionContext,
    surface_kind: str,
    split: SampleSplit,
) -> Path:
    return (
        context.topology.repo_root
        / "runtime"
        / "model"
        / "data_surfaces"
        / context.request.run_name
        / LATENT_DIFFUSION_MODEL_KEY
        / surface_kind
        / SampleSplit(split).value
    )


def _materialize_latent_source_surface(
    *,
    context: ModelStageExecutionContext,
    config: LatentDiffusionConfig,
    split: SampleSplit,
    max_samples: int | None,
    progress,
    operation: str,
    label: str,
    total_semantics: str,
) -> ModelDataSurface:
    manifest_path = resolve_modeling_manifest_path(
        context.topology,
        context.request.manifest_family,
        split,
    )
    manifest_count = resolve_effective_manifest_count(
        manifest_path=manifest_path,
        limit_samples=max_samples,
    )
    manifest_sha = sha256_file(manifest_path)
    provider_config_sha = _stable_sha256(config.to_dict())
    cache_key = model_data_surface_cache_key(
        provider_key=LATENT_DIFFUSION_MODEL_KEY,
        surface_kind="latent_source_sequences",
        split=SampleSplit(split).value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=provider_config_sha,
        run_mode=context.request.run_mode.value,
        limit_samples=max_samples,
        data_version="standardized_bfh_source:v1",
    )
    root = _latent_data_surface_root(context, "latent_source_sequences", split)
    if (root / "metadata.json").is_file() and (root / "manifest.jsonl").is_file():
        surface = load_model_data_surface(root)
        if surface.metadata.cache_key == cache_key:
            return surface
    with progress.task(
        operation=operation,
        label=label,
        unit="sample",
        total=manifest_count,
        total_semantics=total_semantics,
        allowed_counters=("loaded",),
    ) as task:
        return build_latent_source_surface(
            topology=context.topology,
            manifest_family=context.request.manifest_family,
            split=split,
            surface_root=root,
            source_manifest_sha256=manifest_sha,
            provider_config_sha256=provider_config_sha,
            cache_key=cache_key,
            run_mode=context.request.run_mode.value,
            manifest_entry_count=manifest_count,
            max_samples=max_samples,
            standardization_missing_observation_policy=(
                config.latent_target.standardization_missing_observation_policy
            ),
            on_sample_loaded=lambda index, total, sample_id: task.advance(
                1,
                counters={"loaded": index},
            ),
        )


def _materialize_latent_sequence_surface(
    *,
    context: ModelStageExecutionContext,
    config: LatentDiffusionConfig,
    source_surface: ModelDataSurface,
    stats: BfhStandardizationStats,
    target_spec: LatentTargetSpec,
    split: SampleSplit,
    progress,
    operation: str,
    label: str,
    total_semantics: str,
) -> ModelDataSurface:
    manifest_sha = source_surface.metadata.source_manifest_sha256
    provider_config_sha = _stable_sha256(config.to_dict())
    cache_key = model_data_surface_cache_key(
        provider_key=LATENT_DIFFUSION_MODEL_KEY,
        surface_kind="latent_sequences",
        split=SampleSplit(split).value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=provider_config_sha,
        run_mode=context.request.run_mode.value,
        limit_samples=None,
        data_version=f"{target_spec.target_type}:v1",
    )
    root = _latent_data_surface_root(context, "latent_sequences", split)
    if (root / "metadata.json").is_file() and (root / "manifest.jsonl").is_file():
        surface = load_model_data_surface(root)
        if surface.metadata.cache_key == cache_key:
            return surface
    with progress.task(
        operation=operation,
        label=label,
        unit="shard",
        total=source_surface.metadata.shard_count,
        total_semantics=total_semantics,
    ) as task:
        surface = build_latent_sequence_surface_from_source_surface(
            source_surface=source_surface,
            stats=stats,
            surface_root=root,
            source_manifest_sha256=manifest_sha,
            provider_config_sha256=provider_config_sha,
            cache_key=cache_key,
            run_mode=context.request.run_mode.value,
            manifest_entry_count=source_surface.metadata.manifest_entry_count,
            target_spec=target_spec,
        )
        task.advance(source_surface.metadata.shard_count)
    _copy_surface_sources(source_surface.root, surface.root)
    return surface


def _materialize_latent_window_surface(
    *,
    context: ModelStageExecutionContext,
    config: LatentDiffusionConfig,
    source_surface: ModelDataSurface,
    stats: BfhStandardizationStats,
    target_spec: LatentTargetSpec,
    split: SampleSplit,
    progress,
    operation: str,
    label: str,
    total_semantics: str,
) -> ModelDataSurface:
    manifest_sha = source_surface.metadata.source_manifest_sha256
    provider_config_sha = _stable_sha256(config.to_dict())
    cache_key = model_data_surface_cache_key(
        provider_key=LATENT_DIFFUSION_MODEL_KEY,
        surface_kind="latent_windows",
        split=SampleSplit(split).value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=provider_config_sha,
        run_mode=context.request.run_mode.value,
        limit_samples=None,
        data_version=f"{target_spec.target_type}:windows:v1",
    )
    root = _latent_data_surface_root(context, "latent_windows", split)
    if (root / "metadata.json").is_file() and (root / "manifest.jsonl").is_file():
        surface = load_model_data_surface(root)
        if surface.metadata.cache_key == cache_key:
            return surface
    with progress.task(
        operation=operation,
        label=label,
        unit="shard",
        total=source_surface.metadata.shard_count,
        total_semantics=total_semantics,
        allowed_counters=("windows",),
    ) as task:
        surface = build_latent_window_surface_from_source_surface(
            source_surface=source_surface,
            stats=stats,
            surface_root=root,
            source_manifest_sha256=manifest_sha,
            provider_config_sha256=provider_config_sha,
            cache_key=cache_key,
            run_mode=context.request.run_mode.value,
            manifest_entry_count=source_surface.metadata.manifest_entry_count,
            target_spec=target_spec,
            on_shard_built=lambda index, total, windows: task.advance(
                1,
                counters={"windows": windows},
            ),
        )
    _copy_surface_sources(source_surface.root, surface.root)
    return surface


def _materialize_encoded_window_latent_surface(
    *,
    context: ModelStageExecutionContext,
    config: LatentDiffusionConfig,
    window_surface: ModelDataSurface,
    autoencoder,
    target_spec: LatentTargetSpec,
    split: SampleSplit,
    device: torch.device,
    progress,
    operation: str,
    label: str,
    total_semantics: str,
) -> ModelDataSurface:
    if config.latent_autoencoder is None:
        raise LatentDiffusionError("latent_autoencoder config is required to encode window surfaces.")
    manifest_sha = window_surface.metadata.source_manifest_sha256
    provider_config_sha = _stable_sha256(config.to_dict())
    cache_key = model_data_surface_cache_key(
        provider_key=LATENT_DIFFUSION_MODEL_KEY,
        surface_kind="latent_sequences",
        split=SampleSplit(split).value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=provider_config_sha,
        run_mode=context.request.run_mode.value,
        limit_samples=None,
        data_version=f"{target_spec.target_type}:encoded:v1",
    )
    root = _latent_data_surface_root(context, "latent_sequences", split)
    if (root / "metadata.json").is_file() and (root / "manifest.jsonl").is_file():
        surface = load_model_data_surface(root)
        if surface.metadata.cache_key == cache_key:
            return surface
    reader = ModelDataSurfaceReader(window_surface)
    writer = ModelDataSurfaceWriter(
        root=root,
        provider_key=LATENT_DIFFUSION_MODEL_KEY,
        surface_kind="latent_sequences",
        split=SampleSplit(split).value,
        manifest_family=reader.metadata.manifest_family,
        source_manifest_path=window_surface.metadata.source_manifest_path,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=provider_config_sha,
        cache_key=cache_key,
        manifest_entry_count=window_surface.metadata.manifest_entry_count,
        run_mode=context.request.run_mode.value,
        data_version=f"{target_spec.target_type}:encoded:v1",
    )
    resolved_device = torch.device(device)
    autoencoder = autoencoder.to(resolved_device)
    autoencoder.eval()
    with progress.task(
        operation=operation,
        label=label,
        unit="window",
        total=window_surface.metadata.unit_count,
        total_semantics=total_semantics,
    ) as task:
        with torch.no_grad():
            for shard in reader.iter_shards():
                source_index = shard["source_index"].long()
                window_index = shard["window_index"].long()
                for raw_source_index in torch.unique(source_index, sorted=True).tolist():
                    mask = source_index == int(raw_source_index)
                    order = torch.argsort(window_index[mask])
                    windows = shard["windows"][mask][order].float()
                    values = windows.reshape(windows.shape[0], -1)
                    encoded_batches: list[torch.Tensor] = []
                    for start in range(0, values.shape[0], config.latent_autoencoder.batch_size):
                        batch = values[start : start + config.latent_autoencoder.batch_size].to(
                            device=resolved_device,
                            dtype=torch.float32,
                        )
                        encoded_batches.append(autoencoder.encode(batch).detach().cpu())
                    encoded = torch.cat(encoded_batches, dim=0).to(dtype=torch.float32)
                    if not torch.isfinite(encoded).all():
                        raise LatentDiffusionError("encoded window latents contain non-finite values.")
                    latent_count = int(encoded.shape[0])
                    writer.append_units(
                        {
                            "latent_values": encoded,
                            "latent_mask": torch.ones_like(encoded, dtype=torch.bool),
                            "source_index": torch.full(
                                (latent_count,),
                                int(raw_source_index),
                                dtype=torch.int64,
                            ),
                            "latent_index": torch.arange(latent_count, dtype=torch.int64),
                        },
                        sample_count=1,
                        frame_count=latent_count,
                    )
                    task.advance(latent_count)
    surface = writer.close()
    _copy_surface_sources(window_surface.root, surface.root)
    return surface


def _write_latent_manifest_from_surface(
    surface: ModelDataSurface,
    *,
    output_root: Path,
    manifest_path: Path,
    target_spec: LatentTargetSpec,
    progress,
    operation: str,
    label: str,
) -> tuple[LatentManifestEntry, ...]:
    reader = ModelDataSurfaceReader(surface)
    sources = _latent_source_records_by_index(surface)
    entries: list[LatentManifestEntry] = []
    with progress.task(
        operation=operation,
        label=label,
        unit="sample",
        total=surface.metadata.loaded_sample_count,
        total_semantics="latent manifest entries written from tensor surface",
        allowed_counters=("frames",),
    ) as task:
        for shard in reader.iter_shards():
            source_index = shard["source_index"].long()
            latent_index = shard["latent_index"].long()
            for raw_source_index in torch.unique(source_index, sorted=True).tolist():
                source = sources[int(raw_source_index)]
                mask = source_index == int(raw_source_index)
                order = torch.argsort(latent_index[mask])
                values = shard["latent_values"][mask][order].detach().cpu().numpy().astype(np.float32)
                validity = shard["latent_mask"][mask][order].detach().cpu().numpy().astype(np.bool_)
                latent_count = int(values.shape[0])
                frame_count = int(source["frame_count"])
                sequence = LatentSequence(
                    schema_version=LATENT_SEQUENCE_SCHEMA_VERSION,
                    sample_id=str(source["sample_id"]),
                    source_sentence_name=str(source["source_sentence_name"]),
                    split=SampleSplit(str(source["split"])),
                    values=values,
                    validity_mask=validity,
                    frame_count=frame_count,
                    latent_count=latent_count,
                    latent_dim=int(values.shape[1]),
                    target_spec=target_spec,
                )
                latent_path = Path(output_root) / f"{_safe_path_token(sequence.sample_id)}.npz"
                write_latent_sequence_npz(latent_path, sequence)
                entry = LatentManifestEntry(
                    schema_version=LATENT_MANIFEST_SCHEMA_VERSION,
                    sample_id=sequence.sample_id,
                    source_sentence_name=sequence.source_sentence_name,
                    split=sequence.split,
                    latent_path=latent_path,
                    frame_count=sequence.frame_count,
                    latent_count=sequence.latent_count,
                    latent_dim=sequence.latent_dim,
                    target_type=target_spec.target_type,
                    temporal_granularity=target_spec.temporal_granularity,
                    window_size=target_spec.window_size,
                    stride=target_spec.stride,
                    issues=(),
                )
                entries.append(entry)
                task.advance(1, counters={"frames": latent_count})
    write_latent_manifest_jsonl(manifest_path, entries)
    return tuple(entries)


def _copy_surface_sources(source_root: Path, target_root: Path) -> None:
    source_path = Path(source_root) / "sources.jsonl"
    if source_path.is_file():
        target_path = Path(target_root) / "sources.jsonl"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def _latent_source_records_by_index(surface: ModelDataSurface) -> dict[int, Mapping[str, object]]:
    path = surface.root / "sources.jsonl"
    if not path.is_file():
        raise LatentDiffusionError("latent surface is missing sources.jsonl.")
    records: dict[int, Mapping[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise LatentDiffusionError("latent sources.jsonl rows must be objects.")
            records[int(record["source_index"])] = record
    return records


def _latent_sources_by_index_from_surface(surface: ModelDataSurface) -> Mapping[int, LatentSourceSample]:
    return {
        source_index: _latent_source_from_surface_record(record)
        for source_index, record in _latent_source_records_by_index(surface).items()
    }


def _latent_sources_by_id_from_surface(surface: ModelDataSurface) -> Mapping[str, LatentSourceSample]:
    sources: dict[str, LatentSourceSample] = {}
    for record in _latent_source_records_by_index(surface).values():
        source = _latent_source_from_surface_record(record)
        sources[source.sample_id] = source
    return sources


def _iter_limited_latent_sources(
    surface: ModelDataSurface,
    *,
    limit: int | None,
):
    emitted = 0
    records = _latent_source_records_by_index(surface)
    for source_index in sorted(records):
        if limit is not None and emitted >= int(limit):
            break
        yield _latent_source_from_surface_record(records[source_index])
        emitted += 1


def _latent_source_from_surface_record(record: Mapping[str, object]) -> LatentSourceSample:
    frame_count = int(record["frame_count"])
    layout = default_bfh_tensor_layout()
    sample_id = str(record["sample_id"])
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
    return LatentSourceSample(
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


def _safe_path_token(value: str) -> str:
    if value in {".", ".."} or "/" in value or "\\" in value or "." in value:
        raise LatentDiffusionError("sample_id must be a safe path token for latent output.")
    return value


def _compute_precision_policy(context: ModelStageExecutionContext) -> str:
    precision = context.request.compute_profile.get("precision")
    if isinstance(precision, Mapping):
        policy = precision.get("policy")
        if isinstance(policy, str) and policy.strip():
            return policy
    return "auto"


def _precision_metadata(
    context: ModelStageExecutionContext,
    config: LatentDiffusionConfig,
    *,
    precision_applied: bool | None = None,
) -> dict[str, object]:
    resolved = resolve_precision_policy(
        _compute_precision_policy(context),
        _resolve_device(config.training.device),
    )
    applied = resolved.autocast_enabled if precision_applied is None else (
        bool(precision_applied) and resolved.autocast_enabled
    )
    return resolved.to_metadata(precision_applied=applied)


def _run_mode_caps(effective_config: Mapping[str, object]) -> dict[str, int | None]:
    raw = effective_config.get("run_mode_overrides")
    if not isinstance(raw, Mapping):
        raise LatentDiffusionError("effective_config is missing run_mode_overrides.")
    return {
        "limit_train_samples": _optional_positive_int(raw.get("limit_train_samples"), "limit_train_samples"),
        "limit_validation_samples": _optional_positive_int(
            raw.get("limit_validation_samples"),
            "limit_validation_samples",
        ),
        "limit_prediction_samples": _optional_positive_int(
            raw.get("limit_prediction_samples"),
            "limit_prediction_samples",
        ),
    }


def _effective_manifest_total(
    *,
    context: ModelStageExecutionContext,
    split: SampleSplit,
    limit_samples: int | None,
) -> int:
    return resolve_effective_manifest_count(
        manifest_path=resolve_modeling_manifest_path(
            context.topology,
            context.request.manifest_family,
            split,
        ),
        limit_samples=limit_samples,
    )


def _effective_sampling_steps(
    effective_config: Mapping[str, object],
    *,
    configured_steps: int,
) -> tuple[int, str]:
    compute_profile = effective_config.get("compute_profile")
    if not isinstance(compute_profile, Mapping):
        return configured_steps, "config"
    profile_name = compute_profile.get("name")
    provider_overrides = compute_profile.get("provider_overrides")
    if profile_name != "colab_a100_80gb" or not isinstance(provider_overrides, Mapping):
        return configured_steps, "config"
    latent_overrides = provider_overrides.get(LATENT_DIFFUSION_MODEL_KEY)
    if not isinstance(latent_overrides, Mapping):
        return configured_steps, "config"
    run_mode = effective_config.get("run_mode")
    if run_mode == "smoke":
        value = latent_overrides.get("smoke_sampling_steps")
        source = "compute_profile_smoke_override"
    elif run_mode == "debug":
        value = latent_overrides.get("debug_sampling_steps")
        source = "compute_profile_debug_override"
    else:
        return configured_steps, "config"
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LatentDiffusionError(f"{source} must be a positive integer.")
    return min(configured_steps, value), source


def _effective_max_epochs(
    effective_config: Mapping[str, object],
    config: LatentDiffusionConfig,
) -> int:
    raw = effective_config.get("run_mode_overrides")
    if not isinstance(raw, Mapping):
        return config.training.max_epochs
    override = raw.get("max_epochs")
    if override is None:
        return config.training.max_epochs
    if not isinstance(override, int) or isinstance(override, bool) or override <= 0:
        raise LatentDiffusionError("run_mode max_epochs override must be positive.")
    return min(config.training.max_epochs, override)


def _run_seed(effective_config: Mapping[str, object]) -> int:
    raw = effective_config.get("seed_policy")
    if not isinstance(raw, Mapping):
        raise LatentDiffusionError("effective_config is missing seed_policy.")
    seed = raw.get("run_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise LatentDiffusionError("effective seed_policy.run_seed must be a non-negative integer.")
    return seed


def _sample_seed(
    *,
    run_seed: int,
    sample_id: str,
    candidate_index: int,
    seed_policy: str,
) -> int:
    if seed_policy not in {"fixed_per_run", "fixed_per_sample"}:
        raise LatentDiffusionError("unsupported latent generation seed_policy.")
    payload = f"{run_seed}:{seed_policy}:{sample_id}:{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647


def _generation_artifact_metadata_from_path(
    path: Path,
    *,
    producer_stage: str,
) -> dict[str, object]:
    token = Path(path).name.removesuffix(".npz")
    if "__g" not in token:
        raise LatentDiffusionError(
            f"generation artifact filename must include sample/candidate token: {path}"
        )
    sample_id, raw_generation = token.rsplit("__g", 1)
    if not sample_id or not raw_generation.isdigit():
        raise LatentDiffusionError(
            f"generation artifact filename has invalid sample/candidate token: {path}"
        )
    generation_index = int(raw_generation)
    return {
        "split": "val",
        "sample_id": sample_id,
        "generation_index": generation_index,
        "candidate_index": generation_index,
        "producer_stage": producer_stage,
        "generation_mode": "stochastic",
        "model_key": LATENT_DIFFUSION_MODEL_KEY,
    }


def _read_sequences(path: Path) -> tuple[LatentSequence, ...]:
    entries = read_latent_manifest_jsonl(path)
    if not entries:
        raise LatentDiffusionError(f"latent manifest is empty: {path}")
    sequences = tuple(read_latent_sequence_npz(entry.latent_path) for entry in entries)
    return sequences


def _iter_latent_sequences(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise LatentDiffusionError(
                    f"latent manifest record at line {line_number} must be an object."
                )
            entry = _latent_manifest_entry_from_record(record, line_number=line_number)
            yield read_latent_sequence_npz(entry.latent_path)


def _iter_latent_sequences_from_surface(
    surface: ModelDataSurface,
    target_spec: LatentTargetSpec,
):
    reader = ModelDataSurfaceReader(surface)
    sources = _latent_source_records_by_index(surface)
    for shard in reader.iter_shards():
        source_index = shard["source_index"].long()
        latent_index = shard["latent_index"].long()
        for raw_source_index in torch.unique(source_index, sorted=True).tolist():
            source = sources[int(raw_source_index)]
            mask = source_index == int(raw_source_index)
            order = torch.argsort(latent_index[mask])
            values = shard["latent_values"][mask][order].detach().cpu().numpy().astype(np.float32)
            validity = shard["latent_mask"][mask][order].detach().cpu().numpy().astype(np.bool_)
            latent_count = int(values.shape[0])
            yield LatentSequence(
                schema_version=LATENT_SEQUENCE_SCHEMA_VERSION,
                sample_id=str(source["sample_id"]),
                source_sentence_name=str(source["source_sentence_name"]),
                split=SampleSplit(str(source["split"])),
                values=values,
                validity_mask=validity,
                frame_count=int(source["frame_count"]),
                latent_count=latent_count,
                latent_dim=int(values.shape[1]),
                target_spec=target_spec,
            )


def iter_latent_sequence_batches_from_surface(
    surface: ModelDataSurface | ModelDataSurfaceReader,
    batch_size: int,
    *,
    num_workers: int = 0,
    runtime_trace: MutableMapping[str, object] | None = None,
):
    reader = surface if isinstance(surface, ModelDataSurfaceReader) else ModelDataSurfaceReader(surface)
    for batch in reader.iter_batches(
        batch_size=batch_size,
        num_workers=num_workers,
        runtime_trace=runtime_trace,
    ):
        required = ("latent_values", "latent_mask", "source_index", "latent_index")
        if any(name not in batch for name in required):
            raise LatentDiffusionError("latent sequence surface batch is missing required tensors.")
        yield LatentDenoiserBatch(
            latent_values=batch["latent_values"],
            latent_mask=batch["latent_mask"].bool(),
            source_index=batch["source_index"].long(),
            latent_index=batch["latent_index"].long(),
        )


def _latent_manifest_count(path: Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _validate_sequence_lengths(
    sequences: Sequence[LatentSequence],
    *,
    config: LatentDiffusionConfig,
) -> None:
    for sequence in sequences:
        if sequence.latent_count > config.denoiser.max_positions:
            raise LatentDiffusionError(
                f"latent sequence sample_id={sequence.sample_id!r} latent_count "
                f"{sequence.latent_count} exceeds denoiser.max_positions "
                f"{config.denoiser.max_positions}."
            )


def _latent_root(context: ModelStageExecutionContext) -> Path:
    return context.topology.models.model_intermediate_root(
        LATENT_DIFFUSION_MODEL_KEY,
        context.request.run_name,
        "latent",
    ).path


def _denoiser_root(context: ModelStageExecutionContext) -> Path:
    return context.topology.models.model_intermediate_root(
        LATENT_DIFFUSION_MODEL_KEY,
        context.request.run_name,
        "denoiser",
    ).path


def _generation_root(context: ModelStageExecutionContext) -> Path:
    return context.topology.models.model_intermediate_root(
        LATENT_DIFFUSION_MODEL_KEY,
        context.request.run_name,
        "generation",
    ).path


def _report_root(context: ModelStageExecutionContext) -> Path:
    return context.topology.reports.model_run_root(
        LATENT_DIFFUSION_MODEL_KEY,
        context.request.run_name,
    ).path


def _require_stage_result(
    context: ModelStageExecutionContext,
    kind: ModelStageKind,
) -> ModelStageResult:
    for result in context.stage_results:
        if result.stage.spec.kind is kind:
            return result
    raise LatentDiffusionError(f"stage {kind.value!r} must complete before this stage.")


def _sources_by_id(sources: Sequence[LatentSourceSample]) -> Mapping[str, LatentSourceSample]:
    return {source.sample_id: source for source in sources}


def _metric(
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
        value=_finite_float(value, name),
        epoch=epoch,
        global_step=global_step,
        stage_name="train_denoiser",
        created_at_utc=_now(),
        extra={},
    )


def _write_diffusion_schedule_json(path: Path, schedule: DiffusionSchedule) -> None:
    write_json(path, schedule.to_dict())


def _read_diffusion_schedule_json(path: Path) -> DiffusionSchedule:
    _require_file(path, "diffusion schedule")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LatentDiffusionError(f"malformed diffusion schedule JSON: {exc}") from exc
    if not isinstance(record, Mapping):
        raise LatentDiffusionError("diffusion schedule JSON must contain an object.")
    if record.get("schema_version") != DIFFUSION_SCHEDULE_SCHEMA_VERSION:
        raise LatentDiffusionError("diffusion schedule schema_version is unsupported.")
    return DiffusionSchedule(
        schema_version=str(record["schema_version"]),
        schedule=str(record["schedule"]),
        timesteps=int(record["timesteps"]),
        beta_start=float(record["beta_start"]),
        beta_end=float(record["beta_end"]),
        betas=np.asarray(record["betas"], dtype=np.float32),
        alphas=np.asarray(record["alphas"], dtype=np.float32),
        alpha_bars=np.asarray(record["alpha_bars"], dtype=np.float32),
    )


def _read_support_config_json(
    path: Path,
    *,
    expected_schema_version: str,
    label: str,
) -> Mapping[str, object]:
    _require_file(path, label)
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LatentDiffusionError(f"malformed {label}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise LatentDiffusionError(f"{label} must contain a JSON object.")
    if record.get("schema_version") != expected_schema_version:
        raise LatentDiffusionError(
            f"{label} schema_version must be {expected_schema_version!r}; "
            f"observed {record.get('schema_version')!r}."
        )
    if any(not isinstance(key, str) for key in record):
        raise LatentDiffusionError(f"{label} JSON keys must be strings.")
    return record


def _require_exact_support_config(
    *,
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    label: str,
) -> None:
    if dict(observed) != dict(expected):
        raise LatentDiffusionError(
            f"{label} does not match the effective latent_diffusion/checkpoint config. "
            "Restore the support config artifacts from the same model run as the selected "
            "checkpoint, or rerun the provider stage that materialized them."
        )


def _mapping_from_checkpoint_config(
    checkpoint_config: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = checkpoint_config.get(key)
    if not isinstance(value, Mapping):
        raise LatentDiffusionError(
            f"latent denoiser checkpoint provider_config is missing {key!r}; "
            "cannot verify restored support config artifacts."
        )
    return value


def _write_sampling_traces_jsonl(path: Path, traces: Sequence[LatentSamplingTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise LatentDiffusionError(f"latent denoiser checkpoint could not be loaded: {path}: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise LatentDiffusionError("latent denoiser checkpoint root must be a mapping.")
    return checkpoint


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


def _read_yaml_mapping(path: Path) -> Mapping[str, object]:
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ModelStageExecutionError(f"latent_diffusion YAML is invalid: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ModelStageExecutionError("latent_diffusion YAML root must be a mapping.")
    return dict(cast(Mapping[str, object], loaded))


def _set_seed(seed: int) -> None:
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _finite_tensor_value(value: torch.Tensor, name: str) -> float:
    return _finite_float(float(value.detach().cpu().item()), name)


def _finite_float(value: float, name: str) -> float:
    if not math.isfinite(float(value)):
        raise LatentDiffusionError(f"{name} must be finite.")
    return float(value)


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LatentDiffusionError(f"{name} must be a positive integer or null.")
    return value


def _require_file(path: Path, label: str) -> None:
    if not Path(path).is_file():
        raise LatentDiffusionError(
            f"missing {label}: {path}. Restore or run the latent_diffusion provider "
            "stage that materializes this required artifact before inference/export."
        )


def _sample_candidate_token(sample_id: str, candidate_index: int) -> str:
    token = f"{sample_id}__g{candidate_index}"
    if (
        not token.strip()
        or token in {".", ".."}
        or "/" in token
        or "\\" in token
        or "." in token
    ):
        raise LatentDiffusionError("sample_id must be a safe path token for generation output.")
    return token


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["LatentDiffusionProvider", "_latent_diffusion_performance_metadata"]
