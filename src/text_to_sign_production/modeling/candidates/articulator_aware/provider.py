"""Concrete Phase 8 provider for articulator-aware text-to-pose generation."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    load_generated_pose_payload,
    write_generated_pose_split_to_explicit_root,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    default_bfh_tensor_layout,
    vectorize_bfh_pose_arrays,
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
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    ArticulatorAwareConfig,
    articulator_aware_config_from_mapping,
    load_articulator_aware_config,
    ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP,
    ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL,
)
from text_to_sign_production.modeling.candidates.articulator_aware.dataset import (
    ArticulatorSourceSample,
    build_articulator_frame_surface_from_source_surface,
    build_articulator_source_surface,
)
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)
from text_to_sign_production.modeling.candidates.articulator_aware.exporter import (
    ArticulatorGeneratedSample,
    export_articulator_generated_pose_split,
    export_articulator_generated_pose_stream,
    generate_articulator_pose_for_source,
    generated_pose_sample_for_articulator,
    load_articulator_checkpoint_models,
)
from text_to_sign_production.modeling.candidates.articulator_aware.io import (
    read_loss_weighting_policy_json,
    read_mask_strategy_json,
    read_partition_policy_json,
    write_loss_weighting_policy_json,
    write_mask_strategy_json,
    write_partition_policy_json,
)
from text_to_sign_production.modeling.candidates.articulator_aware.losses import (
    build_channel_loss_weighting_policy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    build_articulator_partition_policy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.spec import (
    ARTICULATOR_AWARE_MODEL_KEY,
)
from text_to_sign_production.modeling.candidates.articulator_aware.trainer import (
    build_articulator_length_config_payload,
    build_articulator_model_config_payload,
    train_articulator_model_from_surfaces,
)
from text_to_sign_production.modeling.data import resolve_modeling_manifest_path
from text_to_sign_production.modeling.data_surfaces import (
    ModelDataSurface,
    load_model_data_surface,
    model_data_surface_cache_key,
    resolve_effective_manifest_count,
    sha256_file,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.training import (
    load_semantic_training_objective_for_request,
)
from text_to_sign_production.modeling.research import ModelKey, ModelSpec
from text_to_sign_production.modeling.registry import require_model_spec
from text_to_sign_production.modeling.training.precision import resolve_precision_policy
from text_to_sign_production.modeling.training.stages import ProviderStageArtifactRole
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    provider_active_overrides,
)
from text_to_sign_production.workflows.model.contracts.compute_profile_application import (
    compute_profile_dataloader_section,
    provider_candidate_overrides_for_profile,
    provider_compute_profile_application,
)


class ArticulatorAwareProvider:
    """Real provider implementing channel-aware pose heads and predicted length."""

    @property
    def spec(self) -> ModelSpec:
        return require_model_spec(ModelKey.ARTICULATOR_AWARE)

    @property
    def full_data_pipeline_capability(self) -> ProviderFullDataPipelineCapability:
        return ProviderFullDataPipelineCapability(
            provider_key=ARTICULATOR_AWARE_MODEL_KEY,
            full_training_data_mode="streaming_sharded",
            verified=True,
            verification_evidence=(
                "behavior:frame_training:surface_reader",
                "behavior:export:incremental_writer",
                "progress:known_totals",
            ),
            covered_stages=(
                "define_channel_partitions",
                "define_mask_strategy",
                "define_loss_weighting",
                "train_structure_aware",
                "export_generated_pose",
            ),
            limitations=(),
        )

    @property
    def full_data_pipeline_declaration(self) -> ProviderFullDataPipelineDeclaration:
        capability = self.full_data_pipeline_capability
        return ProviderFullDataPipelineDeclaration(
            provider_key=ARTICULATOR_AWARE_MODEL_KEY,
            mode="streaming_sharded",
            covered_stages=capability.covered_stages,
        )

    def calibration_policy(
        self,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ProviderCalibrationPolicy:
        if loaded_config.model_key is not ModelKey.ARTICULATOR_AWARE:
            raise ModelStageExecutionError(
                "articulator_aware calibration policy requires articulator_aware config."
            )
        application = loaded_config.effective_config.get("compute_profile_application")
        if not isinstance(application, Mapping):
            raise ModelStageExecutionError(
                "articulator_aware calibration policy requires compute_profile_application."
            )
        candidate_keys = tuple(application.get("calibration_candidate_keys", ()))
        override_targets = application.get("calibration_override_targets")
        if not isinstance(override_targets, Mapping):
            raise ModelStageExecutionError(
                "articulator_aware calibration policy requires calibration_override_targets."
            )
        return ProviderCalibrationPolicy(
            provider_key=ARTICULATOR_AWARE_MODEL_KEY,
            supports_provider_real=True,
            required_run_modes=("full",),
            required_compute_profiles=("colab_a100_80gb",),
            candidate_keys=candidate_keys,
            override_targets=dict(override_targets),
            representative_surface_kinds={
                "frame_batch_size": "articulator_frame_units",
                "source_batch_size": "articulator_source_sequences",
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
        from text_to_sign_production.modeling.candidates.articulator_aware.calibration import (
            build_representative_calibration_surface,
        )

        return build_representative_calibration_surface(
            context=context,
            candidate_key=candidate_key,
            policy=policy,
            progress_session=progress_session,
        )

    def load_config(self, request: ModelRunRequest) -> ModelProviderLoadedConfig:
        if request.model_key is not ModelKey.ARTICULATOR_AWARE:
            raise ModelStageExecutionError(
                "ArticulatorAwareProvider only accepts articulator_aware requests."
            )
        if request.config_path is None:
            raise ModelStageExecutionError(
                "articulator_aware requires MODEL_CONFIG_RELATIVE_PATH="
                "configs/modeling/articulator_aware.yaml."
            )
        config = load_articulator_aware_config(request.config_path)
        effective_config = _effective_provider_config(config, request=request)
        try:
            raw = yaml.safe_load(Path(request.config_path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ModelStageExecutionError(f"articulator_aware config could not be read: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ModelStageExecutionError("articulator_aware raw config must be a mapping.")
        return ModelProviderLoadedConfig(
            model_key=ModelKey.ARTICULATOR_AWARE,
            source_path=request.config_path,
            raw_config=raw,
            effective_config=effective_config,
        )

    def plan_stages(
        self,
        request: ModelRunRequest,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ModelStagePlan:
        if request.model_key is not ModelKey.ARTICULATOR_AWARE:
            raise ModelStageExecutionError("articulator_aware planning requires articulator_aware.")
        if loaded_config.model_key is not ModelKey.ARTICULATOR_AWARE:
            raise ModelStageExecutionError("articulator_aware loaded config has the wrong model key.")
        return default_stage_plan_for_request(request)

    def execute_stage(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
    ) -> ModelStageResult:
        config = _config_from_loaded(context.loaded_config)
        try:
            if stage.spec.kind is ModelStageKind.DEFINE_CHANNEL_PARTITIONS:
                return self._define_partitions(stage, context, config)
            if stage.spec.kind is ModelStageKind.DEFINE_MASK_STRATEGY:
                return self._define_mask_strategy(stage, context, config)
            if stage.spec.kind is ModelStageKind.DEFINE_LOSS_WEIGHTING:
                return self._define_loss_weighting(stage, context, config)
            if stage.spec.kind is ModelStageKind.TRAIN_STRUCTURE_AWARE:
                return self._train(stage, context, config)
            if stage.spec.kind is ModelStageKind.EXPORT_GENERATED_POSE:
                return self._export(stage, context, config)
        except ArticulatorAwareError as exc:
            raise ModelStageExecutionError(str(exc)) from exc
        raise ModelStageExecutionError(
            f"articulator_aware does not implement stage {stage.spec.kind.value!r}."
        )

    def write_model_reports(
        self,
        context: ModelStageExecutionContext,
        results: ModelExecutionResult,
    ) -> tuple[ModelStageArtifactRef, ...]:
        from text_to_sign_production.modeling.candidates.articulator_aware.reports import (
            write_articulator_aware_reports,
        )

        return write_articulator_aware_reports(context=context, results=results)

    def runtime_support_artifacts(
        self,
        *,
        request: ModelRunRequest,
        loaded_config: ModelProviderLoadedConfig,
        execution: ModelExecutionResult,
        topology,
    ) -> tuple[ModelRuntimeSupportArtifact, ...]:
        del loaded_config, execution
        root = topology.models.model_run_root(
            ARTICULATOR_AWARE_MODEL_KEY,
            request.run_name,
        ).path
        articulator = topology.models.model_intermediate_root(
            ARTICULATOR_AWARE_MODEL_KEY,
            request.run_name,
            "articulator",
        ).path
        specs = (
            ("articulator_partition_policy", articulator / "partition_policy.json"),
            ("articulator_mask_strategy", articulator / "mask_strategy.json"),
            ("articulator_loss_weighting_policy", articulator / "loss_weighting_policy.json"),
            ("articulator_model_config", articulator / "model_config.json"),
            ("articulator_length_predictor_config", articulator / "length_predictor_config.json"),
        )
        return tuple(
            support_artifact_from_model_run_file(
                model_run_root=root,
                path=path,
                role=role,
                provider_key=ARTICULATOR_AWARE_MODEL_KEY,
            )
            for role, path in specs
        )

    def infer_single_sample(
        self,
        context: ModelSingleSampleInferenceContext,
    ) -> ModelSingleSampleInferenceResult:
        config = _config_from_loaded(context.loaded_config)
        paths = _required_support_paths(context.topology, context.request.run_name)
        policy = read_partition_policy_json(paths["partition_policy"])
        mask_strategy = read_mask_strategy_json(paths["mask_strategy"])
        weighting = read_loss_weighting_policy_json(paths["loss_weighting_policy"])
        _verify_support_policy_matches_config(
            config=config,
            policy=policy,
            mask_strategy=mask_strategy,
            weighting=weighting,
        )
        text_encoder = build_text_encoder(config.text_encoder)
        model_config, length_config = _verify_support_configs(
            paths,
            config=config,
            policy=policy,
            text_embedding_dim=text_encoder.output_dim,
            text_encoder_metadata_payload=text_encoder_metadata(text_encoder),
        )
        device = _resolve_device(config.training.device)
        model, length_predictor = load_articulator_checkpoint_models(
            checkpoint_path=context.checkpoint_path,
            config=config,
            partition_policy=policy,
            device=device,
            run_name=context.request.run_name,
            restored_model_config=model_config,
            restored_length_predictor_config=length_config,
            text_encoder_output_dim=text_encoder.output_dim,
        )
        source = _source_from_manifest_sample(context.sample)
        embedding = text_encoder.encode_one(
            source.text,
            sentence_name=source.source_sentence_name,
        ).pooled_embedding
        pose, generation_metadata = generate_articulator_pose_for_source(
            model=model,
            length_predictor=length_predictor,
            text_embedding=embedding,
            source=source,
            config=config,
            partition_policy=policy,
        )
        generated = generated_pose_sample_for_articulator(
            generated=ArticulatorGeneratedSample(
                source=source,
                pose=pose,
                predicted_frame_count=int(generation_metadata["predicted_frame_count"]),
            ),
            config=config,
            run_name=context.request.run_name,
            seed=_run_seed(context.loaded_config.effective_config),
            producer_stage="test_model_single_sample_inference",
        )
        written = write_generated_pose_split_to_explicit_root(
            manifest_path=context.output_root / "manifest.jsonl",
            samples_root=context.output_root / "samples",
            payload_ref_root=context.output_root.parent,
            split=source.split,
            samples=(generated,),
        )
        if len(written.payload_paths) != 1:
            raise ModelStageExecutionError(
                "articulator_aware single-sample inference wrote no generated payload."
            )
        return ModelSingleSampleInferenceResult(
            model_key=context.request.model_key,
            run_name=context.request.run_name,
            checkpoint_path=context.checkpoint_path,
            sample_id=source.sample_id,
            generated_sample=load_generated_pose_payload(written.payload_paths[0]),
            generated_payload_path=written.payload_paths[0],
            generated_manifest_path=written.manifest_path,
            metadata={
                **generation_metadata,
                "producer_stage": "test_model_single_sample_inference",
                "checkpoint_path": str(context.checkpoint_path),
                **{f"{name}_path": str(path) for name, path in paths.items()},
            },
        )

    def _define_partitions(self, stage, context, config):
        context.provider_progress(stage).status(
            "articulator define channel partitions",
            source=config.partition_policy.source,
        )
        policy = build_articulator_partition_policy(
            config=config.partition_policy,
            layout=default_bfh_tensor_layout(),
        )
        path = _articulator_root(context) / "partition_policy.json"
        write_partition_policy_json(path, policy)
        report = _report_root(context) / "partition_policy_report.md"
        _write_stage_report(
            report,
            "Partition Policy Report",
            [
                "Partitions are derived from canonical BFH channels.",
                "Primary channels: " + ", ".join(channel.value for channel in policy.primary_channels),
            ],
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role="articulator_partition_policy",
                    path=path,
                    kind="model_articulator_partition_policy",
                    metadata={
                        "primary_channels": [channel.value for channel in policy.primary_channels],
                        "producer_stage": "define_channel_partitions",
                    },
                ),
                ModelStageArtifactRef(
                    role="partition_policy_report",
                    path=report,
                    kind="model_articulator_report",
                ),
            ),
            metadata={"partition_policy_path": str(path)},
        )

    def _define_mask_strategy(self, stage, context, config):
        context.provider_progress(stage).status(
            "articulator define mask strategy",
            missing_channel_policy=config.mask_strategy.missing_channel_policy,
        )
        _require_prior_stage(context, ModelStageKind.DEFINE_CHANNEL_PARTITIONS)
        path = _articulator_root(context) / "mask_strategy.json"
        write_mask_strategy_json(path, config.mask_strategy)
        report = _report_root(context) / "mask_strategy_report.md"
        _write_stage_report(
            report,
            "Mask Strategy Report",
            [
                f"Missing channel policy: `{config.mask_strategy.missing_channel_policy}`",
                f"All-invalid sample policy: `{config.mask_strategy.all_invalid_sample_policy}`",
            ],
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role="articulator_mask_strategy",
                    path=path,
                    kind="model_articulator_mask_strategy",
                    metadata={"producer_stage": "define_mask_strategy"},
                ),
                ModelStageArtifactRef(
                    role="mask_strategy_report", path=report, kind="model_articulator_report"
                ),
            ),
            metadata={"mask_strategy_path": str(path)},
        )

    def _define_loss_weighting(self, stage, context, config):
        context.provider_progress(stage).status(
            "articulator define loss weighting",
            policy=config.loss_weighting.policy,
        )
        _require_prior_stage(context, ModelStageKind.DEFINE_MASK_STRATEGY)
        policy = build_channel_loss_weighting_policy(config.loss_weighting)
        path = _articulator_root(context) / "loss_weighting_policy.json"
        write_loss_weighting_policy_json(path, policy)
        report = _report_root(context) / "loss_weighting_report.md"
        _write_stage_report(
            report,
            "Loss Weighting Report",
            [
                "Channel reconstruction loss uses static normalized weights.",
                f"Architecture: `{config.structure_variant.architecture}`.",
                f"Velocity loss weight: `{config.loss_weighting.velocity_weight}`.",
                f"Hand motion coordination weight: `{config.loss_weighting.symmetry_weight}`.",
                "Hand coordination is a motion prior, not a linguistic symmetry assumption.",
                f"Cross-channel temporal consistency weight: `{config.loss_weighting.cross_channel_consistency_weight}`.",
            ],
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role="articulator_loss_weighting_policy",
                    path=path,
                    kind="model_articulator_loss_weighting_policy",
                    metadata={"producer_stage": "define_loss_weighting"},
                ),
                ModelStageArtifactRef(
                    role="loss_weighting_report", path=report, kind="model_articulator_report"
                ),
            ),
            metadata={"loss_weighting_policy_path": str(path)},
        )

    def _train(self, stage, context, config):
        policy, mask_strategy, weighting = _read_stage_policies(context)
        caps = _run_mode_overrides(context.loaded_config.effective_config)
        progress = context.provider_progress(stage)
        train_source_surface = _materialize_articulator_source_surface(
            context=context,
            config=config,
            split=context.request.train_split,
            max_samples=caps["limit_train_samples"],
            progress=progress,
            operation="materialize_train_sources",
            label="articulator build train source surface",
            total_semantics="train source samples written to tensor surface",
        )
        validation_source_surface = _materialize_articulator_source_surface(
            context=context,
            config=config,
            split=context.request.validation_split,
            max_samples=caps["limit_validation_samples"],
            progress=progress,
            operation="materialize_validation_sources",
            label="articulator build validation source surface",
            total_semantics="validation source samples written to tensor surface",
        )
        train_frame_surface = _materialize_articulator_frame_surface(
            context=context,
            config=config,
            source_surface=train_source_surface,
            split=context.request.train_split,
            partition_policy=policy,
            mask_strategy=mask_strategy,
            progress=progress,
            operation="build_train_frame_surface",
            label="articulator build train frame surface",
            total_semantics="train source shards converted to frame units",
        )
        validation_frame_surface = _materialize_articulator_frame_surface(
            context=context,
            config=config,
            source_surface=validation_source_surface,
            split=context.request.validation_split,
            partition_policy=policy,
            mask_strategy=mask_strategy,
            progress=progress,
            operation="build_validation_frame_surface",
            label="articulator build validation frame surface",
            total_semantics="validation source shards converted to frame units",
        )
        result = train_articulator_model_from_surfaces(
            config=config,
            train_surface=train_frame_surface,
            validation_surface=validation_frame_surface,
            train_source_surface=train_source_surface,
            validation_source_surface=validation_source_surface,
            partition_policy=policy,
            mask_config=mask_strategy,
            weighting=weighting,
            output_root=context.topology.models.model_run_root(
                ARTICULATOR_AWARE_MODEL_KEY,
                context.request.run_name,
            ).path,
            run_name=context.request.run_name,
            seed=_run_seed(context.loaded_config.effective_config),
            semantic_config=load_semantic_training_objective_for_request(
                request=context.request,
                model_key=ARTICULATOR_AWARE_MODEL_KEY,
            ),
            progress=progress,
            precision_policy_name=_compute_precision_policy(context),
        )
        runtime_trace = _require_articulator_surface_reader_runtime_trace(
            result.runtime_trace,
            config=config,
            context="articulator_aware.train_structure_aware",
        )
        split_artifacts = (
            ("articulator_channel_loss_records", result.train_channel_loss_records_path, "model_articulator_channel_loss_records", "train"),
            ("articulator_channel_loss_records", result.validation_channel_loss_records_path, "model_articulator_channel_loss_records", "val"),
            ("articulator_channel_diagnostics", result.train_channel_diagnostics_path, "model_articulator_channel_diagnostics", "train"),
            ("articulator_channel_diagnostics", result.validation_channel_diagnostics_path, "model_articulator_channel_diagnostics", "val"),
            ("articulator_channel_diagnostics_summary", result.train_channel_diagnostics_summary_path, "model_articulator_channel_diagnostics_summary", "train"),
            ("articulator_channel_diagnostics_summary", result.validation_channel_diagnostics_summary_path, "model_articulator_channel_diagnostics_summary", "val"),
        )
        artifacts = [
            ModelStageArtifactRef(
                role=ProviderStageArtifactRole.CHECKPOINT_BEST,
                path=result.best_checkpoint_path,
                kind="model_articulator_checkpoint",
                metadata={"producer_stage": "train_structure_aware"},
            ),
            ModelStageArtifactRef(
                role=ProviderStageArtifactRole.CHECKPOINT_LAST,
                path=result.last_checkpoint_path,
                kind="model_articulator_checkpoint",
                metadata={"producer_stage": "train_structure_aware"},
            ),
            ModelStageArtifactRef(
                role="checkpoint_best_metadata",
                path=result.best_metadata_path,
                kind="model_articulator_checkpoint",
                metadata={"producer_stage": "train_structure_aware"},
            ),
            ModelStageArtifactRef(
                role="checkpoint_last_metadata",
                path=result.last_metadata_path,
                kind="model_articulator_checkpoint",
                metadata={"producer_stage": "train_structure_aware"},
            ),
            ModelStageArtifactRef(
                role=ProviderStageArtifactRole.TRAINING_METRICS,
                path=result.training_metrics_path,
                kind="model_articulator_training_metrics",
                metadata={"producer_stage": "train_structure_aware"},
            ),
            ModelStageArtifactRef(
                role="articulator_model_config",
                path=result.model_config_path,
                kind="model_articulator_model_config",
                metadata={"producer_stage": "train_structure_aware"},
            ),
            ModelStageArtifactRef(
                role="articulator_length_predictor_config",
                path=result.length_predictor_config_path,
                kind="model_articulator_length_predictor_config",
                metadata={"producer_stage": "train_structure_aware"},
            ),
        ]
        artifacts.extend(
            ModelStageArtifactRef(
                role=role,
                path=path,
                kind=kind,
                metadata={"split": split, "producer_stage": "train_structure_aware"},
            )
            for role, path, kind, split in split_artifacts
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=tuple(artifacts),
            metrics={
                "final_train_loss": result.final_train_loss,
                "validation_channel_weighted_loss": result.validation_channel_weighted_loss,
                "validation_length_loss": result.validation_length_loss,
                "best_metric_value": result.validation_channel_weighted_loss,
            },
            metadata={
               **_articulator_aware_performance_metadata(
                    config,
                    runtime_trace=runtime_trace,
                ),
                "best_checkpoint_path": str(result.best_checkpoint_path),
                "last_checkpoint_path": str(result.last_checkpoint_path),
                "best_metric_name": config.checkpoints.selection_metric,
                "best_metric_value": result.validation_channel_weighted_loss,
                "completed_epoch": result.completed_epoch,
                "global_step": result.global_step,
                "model_config_path": str(result.model_config_path),
                "length_predictor_config_path": str(result.length_predictor_config_path),
                "train_source_surface_path": str(train_source_surface.root),
                "validation_source_surface_path": str(validation_source_surface.root),
                "train_frame_surface_path": str(train_frame_surface.root),
                "validation_frame_surface_path": str(validation_frame_surface.root),
                **_precision_metadata(context, config, precision_applied=True),
            },
        )

    def _export(self, stage, context, config):
        policy, mask_strategy, weighting = _read_stage_policies(context)
        paths = _required_support_paths(context.topology, context.request.run_name)
        _verify_support_policy_matches_config(
            config=config,
            policy=policy,
            mask_strategy=mask_strategy,
            weighting=weighting,
        )
        encoder = build_text_encoder(config.text_encoder)
        model_config, length_config = _verify_support_configs(
            paths,
            config=config,
            policy=policy,
            text_embedding_dim=encoder.output_dim,
            text_encoder_metadata_payload=text_encoder_metadata(encoder),
        )
        trained = _require_prior_stage(context, ModelStageKind.TRAIN_STRUCTURE_AWARE)
        checkpoint_path = Path(str(trained.metadata["best_checkpoint_path"]))
        model, length_predictor = load_articulator_checkpoint_models(
            checkpoint_path=checkpoint_path,
            config=config,
            partition_policy=policy,
            device=_resolve_device(config.training.device),
            run_name=context.request.run_name,
            restored_model_config=model_config,
            restored_length_predictor_config=length_config,
            text_encoder_output_dim=encoder.output_dim,
        )
        max_samples = _run_mode_overrides(context.loaded_config.effective_config)["limit_prediction_samples"]
        progress = context.provider_progress(stage)
        validation_surface_path = trained.metadata.get("validation_source_surface_path")
        if isinstance(validation_surface_path, str) and validation_surface_path:
            validation_source_surface = load_model_data_surface(Path(validation_surface_path))
        else:
            validation_source_surface = _materialize_articulator_source_surface(
                context=context,
                config=config,
                split=context.request.validation_split,
                max_samples=max_samples,
                progress=progress,
                operation="build_export_source_surface",
                label="articulator build export source surface",
                total_semantics="validation source samples written for export",
            )
        source_count = validation_source_surface.metadata.loaded_sample_count
        if max_samples is not None:
            source_count = min(source_count, int(max_samples))
        if source_count <= 0:
            raise ArticulatorAwareError("export_generated_pose requires validation source samples.")
        predicted_counts: dict[str, int] = {}
        # write_final_generated_pose is streamed through the generation task to avoid a samples list.
        generate_task = progress.task(
            operation="generate_validation_poses",
            label="articulator_aware export generated poses",
            unit="sample",
            total=source_count,
            total_semantics="validation generated pose samples produced and written",
            allowed_counters=("frames",),
        )

        def generated_iter():
            for source in _iter_articulator_sources(validation_source_surface, limit=max_samples):
                encoded = encoder.encode_batch(
                    [source.text],
                    sentence_names=[source.source_sentence_name],
                )[0]
                embedding = encoded.pooled_embedding
                pose, metadata = generate_articulator_pose_for_source(
                    model=model,
                    length_predictor=length_predictor,
                    text_embedding=embedding,
                    source=source,
                    config=config,
                    partition_policy=policy,
                )
                predicted_count = int(metadata["predicted_frame_count"])
                predicted_counts[source.sample_id] = predicted_count
                yield generated_pose_sample_for_articulator(
                    generated=ArticulatorGeneratedSample(source, pose, predicted_count),
                    config=config,
                    run_name=context.request.run_name,
                    seed=_run_seed(context.loaded_config.effective_config),
                    producer_stage="export_generated_pose",
                )

        with generate_task as task:
            written = export_articulator_generated_pose_stream(
                topology=context.topology,
                run_name=context.request.run_name,
                split=context.request.validation_split,
                generated_samples=generated_iter(),
                on_sample_written=lambda: task.advance(1),
            )
        artifacts = [
            generated_pose_manifest_artifact_ref(
                path=written.manifest_path,
                kind="generated_pose_manifest",
                split=context.request.validation_split,
                generation_mode=config.generation.generation_mode,
                producer_stage="export_generated_pose",
                model_key=ARTICULATOR_AWARE_MODEL_KEY,
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
                    model_key=ARTICULATOR_AWARE_MODEL_KEY,
                    model_run_name=context.request.run_name,
                    manifest_family=context.request.manifest_family.family_id,
                    producer_stage_id="export_generated_pose",
                    artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
                ),
            ),
            metadata={
                **_articulator_aware_performance_metadata(config),
                "manifest_paths": {"val": str(written.manifest_path)},
                "split_sample_counts": {"val": written.sample_count},
                "total_sample_count": written.sample_count,
                "predicted_frame_counts": predicted_counts,
                "generation_mode": config.generation.generation_mode,
                "producer_stage": "export_generated_pose",
                **_precision_metadata(context, config, precision_applied=False),
                "generation_precision_applied": False,
                "generation_precision_reason": "generation kept in fp32 for numerical stability",
            },
        )


def _effective_provider_config(config: ArticulatorAwareConfig, *, request: ModelRunRequest) -> dict[str, object]:
    policy = resolve_model_run_mode_policy(request.run_mode)
    overrides = {
        "limit_train_samples": policy.limit_train_samples,
        "limit_validation_samples": policy.limit_validation_samples,
        "limit_prediction_samples": policy.limit_prediction_samples,
        "max_epochs": policy.max_epochs,
        "batch_size": policy.batch_size,
    }
    training = config.training
    if policy.max_epochs is not None:
        training = replace(training, max_epochs=min(training.max_epochs, policy.max_epochs))
    if policy.batch_size is not None:
        training = replace(training, batch_size=policy.batch_size)
    effective = replace(config, training=training)
    (
        effective,
        dataloader_applied,
        dataloader_not_applicable,
        dataloader_unsupported,
    ) = _apply_articulator_dataloader_overrides(effective, request)
    effective, active_overrides = _apply_articulator_active_overrides(
        effective,
        request,
    )
    training = effective.training
    run_seed = request.seed if request.seed is not None else training.seed
    if run_seed is None:
        run_seed = 0
    effective = replace(effective, training=replace(training, seed=run_seed))
    provider_payload = effective.to_dict()
    text_encoder_payload = dict(provider_payload["text_encoder"])
    text_encoder_payload["output_dim"] = resolve_text_encoder_output_dim(effective.text_encoder)
    provider_payload["text_encoder"] = text_encoder_payload
    candidate_overrides = provider_candidate_overrides_for_profile(
        request.compute_profile,
        provider_key=ARTICULATOR_AWARE_MODEL_KEY,
    )
    candidate_allowed = {"source_batch_size", "frame_batch_size"}
    candidate_unknown = set(candidate_overrides) - candidate_allowed
    telemetry_required_fields = ["source_batch_size", "frame_batch_size"]
    if (
        effective.structure_variant.architecture
        == ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP
    ):
        telemetry_required_fields.append("num_workers")
    application = provider_compute_profile_application(
        compute_profile=request.compute_profile,
        provider_key=ARTICULATOR_AWARE_MODEL_KEY,
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
            key: "articulator_aware supports candidates only for source_batch_size and frame_batch_size."
            for key in candidate_unknown
        },
        telemetry_required_fields=tuple(telemetry_required_fields),
        calibration_candidate_keys=("source_batch_size", "frame_batch_size"),
        calibration_override_targets={
            "source_batch_size": "training.source_batch_size",
            "frame_batch_size": "training.frame_batch_size",
        },
    ).to_dict()
    return {
        "schema_version": "t2sp-articulator-aware-effective-config-v1",
        "model_key": request.model_key.value,
        "run_name": request.run_name,
        "source_path": None if request.config_path is None else str(request.config_path),
        "manifest_family": request.manifest_family.family_id,
        "run_mode": request.run_mode.value,
        "compute_profile": dict(request.compute_profile),
        "compute_profile_application": application,
        "runtime_truth_contract": _articulator_aware_runtime_truth_contract(effective),
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
        "architecture": config.structure_variant.architecture,
        "velocity_loss_support": {
            "implemented": config.structure_variant.architecture == "articulator_factorized_temporal",
            "configured_weight": config.loss_weighting.velocity_weight,
        },
        "batch_semantics": {
            "batch_size": "source/sample batch size",
            "frame_batch_size": "actual frame tensor batch cap",
            "configured_frame_batch_size": effective.training.frame_batch_size,
        },
        "seed_policy": {
            "request_seed": request.seed,
            "config_training_seed": config.training.seed,
            "run_seed": run_seed,
            "generation_mode": config.generation.generation_mode,
        },
        "training": {
            "frame_batch_size": effective.training.frame_batch_size,
            "source_batch_size": effective.training.batch_size,
            "num_workers": effective.training.num_workers,
        },
        "calibration_override_bindings": {
            "frame_batch_size": "provider_config.training.frame_batch_size",
            "source_batch_size": "provider_config.training.batch_size",
        },
        "provider_config": provider_payload,
    }


def _articulator_aware_runtime_truth_contract(
    config: ArticulatorAwareConfig,
) -> dict[str, object]:
    required_runtime_evidence: dict[str, list[str]] = {}

    if (
        config.structure_variant.architecture
        == ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP
    ):
        required_runtime_evidence["num_workers"] = [
            "surface_reader_num_workers_used",
            "surface_reader_worker_mode",
        ]

    return {
        "schema_version": "model.runtime_truth_contract.v1",
        "provider_key": ARTICULATOR_AWARE_MODEL_KEY,
        "required_runtime_evidence": required_runtime_evidence,
    }


def _apply_articulator_active_overrides(
    config: ArticulatorAwareConfig,
    request: ModelRunRequest,
) -> tuple[ArticulatorAwareConfig, Mapping[str, object]]:
    active = provider_active_overrides(
        request.compute_profile,
        provider_key=ARTICULATOR_AWARE_MODEL_KEY,
        run_mode=request.run_mode.value,
    )
    if not active:
        return config, {}

    allowed = {"source_batch_size", "frame_batch_size"}
    unknown = set(active) - allowed
    if unknown:
        raise ArticulatorAwareError(
            "articulator_aware compute profile active overrides contain unsupported keys: "
            f"{sorted(unknown)}."
        )

    training = config.training

    if "source_batch_size" in active:
        value = active["source_batch_size"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ArticulatorAwareError(
                "articulator_aware source_batch_size active override must be positive."
            )
        training = replace(training, batch_size=value)

    if "frame_batch_size" in active:
        value = active["frame_batch_size"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ArticulatorAwareError(
                "articulator_aware frame_batch_size active override must be positive."
            )
        training = replace(training, frame_batch_size=value)

    return replace(config, training=training), dict(active)


def _apply_articulator_dataloader_overrides(
    config: ArticulatorAwareConfig,
    request: ModelRunRequest,
) -> tuple[
    ArticulatorAwareConfig,
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
            unsupported[key] = (
                "articulator_aware does not recognize this dataloader profile key."
            )
            continue
        if key == "num_workers":
            if value is None:
                not_applicable[key] = "compute profile does not request a num_workers override."
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ArticulatorAwareError(
                    "articulator_aware dataloader.num_workers must be non-negative."
                )

            if (
                config.structure_variant.architecture
                == ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL
            ):
                not_applicable[key] = (
                    "articulator_aware factorized temporal architecture uses "
                    "sequence-level in-memory batching and does not consume "
                    "ModelDataSurfaceReader num_workers."
                )
                continue

            updated = replace(
                updated,
                training=replace(updated.training, num_workers=value),
            )
            applied[key] = value
            continue
        not_applicable[key] = (
            "articulator_aware currently wires only num_workers into runtime dataloaders."
        )
    return updated, applied, not_applicable, unsupported

def _articulator_requires_surface_reader_runtime_trace(
    config: ArticulatorAwareConfig,
) -> bool:
    return (
        config.structure_variant.architecture
        == ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP
        and config.training.num_workers > 0
    )


def _require_articulator_surface_reader_runtime_trace(
    runtime_trace: Mapping[str, object] | None,
    *,
    config: ArticulatorAwareConfig,
    context: str,
) -> Mapping[str, object]:
    if not _articulator_requires_surface_reader_runtime_trace(config):
        return {} if runtime_trace is None else runtime_trace

    expected_num_workers = config.training.num_workers

    if not isinstance(runtime_trace, Mapping):
        raise ArticulatorAwareError(
            f"{context} expected surface reader runtime trace because "
            f"architecture={config.structure_variant.architecture!r} and "
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
        raise ArticulatorAwareError(
            f"{context} expected surface reader runtime trace keys {missing} "
            f"because architecture={config.structure_variant.architecture!r} and "
            f"num_workers={expected_num_workers}. This means the stage did not "
            "execute through the trace-aware ModelDataSurfaceReader.iter_batches path."
        )

    observed_num_workers = runtime_trace.get("surface_reader_num_workers_used")
    if observed_num_workers != expected_num_workers:
        raise ArticulatorAwareError(
            f"{context} surface reader num_workers mismatch: "
            f"expected {expected_num_workers}, observed {observed_num_workers}."
        )

    expected_mode = "multiprocess" if expected_num_workers > 0 else "single_process"
    observed_mode = runtime_trace.get("surface_reader_worker_mode")
    if observed_mode != expected_mode:
        raise ArticulatorAwareError(
            f"{context} surface reader worker mode mismatch: "
            f"expected {expected_mode!r}, observed {observed_mode!r}."
        )

    return runtime_trace


def _articulator_aware_performance_metadata(
    config: ArticulatorAwareConfig,
    *,
    runtime_trace: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source_batch_size": config.training.batch_size,
        "frame_batch_size": config.training.frame_batch_size,
        "num_workers": config.training.num_workers,
    }

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


def _config_from_loaded(loaded: ModelProviderLoadedConfig) -> ArticulatorAwareConfig:
    raw = loaded.effective_config.get("provider_config")
    if not isinstance(raw, Mapping):
        raise ModelStageExecutionError("articulator_aware effective config is missing provider_config.")
    return articulator_aware_config_from_mapping(raw)


def _articulator_root(context: ModelStageExecutionContext) -> Path:
    return context.topology.models.model_intermediate_root(
        ARTICULATOR_AWARE_MODEL_KEY, context.request.run_name, "articulator"
    ).path


def _report_root(context: ModelStageExecutionContext) -> Path:
    return context.topology.reports.model_run_root(
        ARTICULATOR_AWARE_MODEL_KEY, context.request.run_name
    ).path


def _require_prior_stage(context: ModelStageExecutionContext, kind: ModelStageKind) -> ModelStageResult:
    for result in context.stage_results:
        if result.stage.spec.kind is kind:
            return result
    raise ArticulatorAwareError(f"stage {kind.value!r} must complete before this stage.")


def _read_stage_policies(context: ModelStageExecutionContext):
    _require_prior_stage(context, ModelStageKind.DEFINE_LOSS_WEIGHTING)
    root = _articulator_root(context)
    return (
        read_partition_policy_json(root / "partition_policy.json"),
        read_mask_strategy_json(root / "mask_strategy.json"),
        read_loss_weighting_policy_json(root / "loss_weighting_policy.json"),
    )


def _run_mode_overrides(effective: Mapping[str, object]) -> dict[str, int | None]:
    raw = effective.get("run_mode_overrides")
    if not isinstance(raw, Mapping):
        raise ArticulatorAwareError("effective config is missing run_mode_overrides.")
    values: dict[str, int | None] = {}
    for name in ("limit_train_samples", "limit_validation_samples", "limit_prediction_samples"):
        value = raw.get(name)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ArticulatorAwareError(f"run mode override {name} must be positive or null.")
        values[name] = value
    return values


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


def _articulator_data_surface_root(
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
        / ARTICULATOR_AWARE_MODEL_KEY
        / surface_kind
        / SampleSplit(split).value
    )


def _materialize_articulator_source_surface(
    *,
    context: ModelStageExecutionContext,
    config: ArticulatorAwareConfig,
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
        provider_key=ARTICULATOR_AWARE_MODEL_KEY,
        surface_kind="articulator_source_sequences",
        split=SampleSplit(split).value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=provider_config_sha,
        run_mode=context.request.run_mode.value,
        limit_samples=max_samples,
        data_version="bfh_source:v1",
    )
    root = _articulator_data_surface_root(context, "articulator_source_sequences", split)
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
        return build_articulator_source_surface(
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
            on_sample_loaded=lambda index, total, sample_id: task.advance(
                1,
                counters={"loaded": index},
            ),
        )


def _materialize_articulator_frame_surface(
    *,
    context: ModelStageExecutionContext,
    config: ArticulatorAwareConfig,
    source_surface: ModelDataSurface,
    split: SampleSplit,
    partition_policy,
    mask_strategy,
    progress,
    operation: str,
    label: str,
    total_semantics: str,
) -> ModelDataSurface:
    manifest_sha = source_surface.metadata.source_manifest_sha256
    provider_config_sha = _stable_sha256(config.to_dict())
    cache_key = model_data_surface_cache_key(
        provider_key=ARTICULATOR_AWARE_MODEL_KEY,
        surface_kind="articulator_frame_units",
        split=SampleSplit(split).value,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=provider_config_sha,
        run_mode=context.request.run_mode.value,
        limit_samples=None,
        data_version="bfh_frame_masks:v1",
    )
    root = _articulator_data_surface_root(context, "articulator_frame_units", split)
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
        surface = build_articulator_frame_surface_from_source_surface(
            source_surface=source_surface,
            partition_policy=partition_policy,
            mask_config=mask_strategy,
            surface_root=root,
            source_manifest_sha256=manifest_sha,
            provider_config_sha256=provider_config_sha,
            cache_key=cache_key,
            run_mode=context.request.run_mode.value,
            manifest_entry_count=source_surface.metadata.manifest_entry_count,
        )
        task.advance(source_surface.metadata.shard_count)
    source_path = source_surface.root / "sources.jsonl"
    if source_path.is_file():
        (surface.root / "sources.jsonl").write_text(
            source_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return surface


def _articulator_sources_by_id_from_surface(surface: ModelDataSurface) -> Mapping[str, ArticulatorSourceSample]:
    records: dict[str, ArticulatorSourceSample] = {}
    path = surface.root / "sources.jsonl"
    if not path.is_file():
        raise ArticulatorAwareError("articulator surface is missing sources.jsonl.")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ArticulatorAwareError("articulator sources.jsonl rows must be objects.")
            source = _articulator_source_from_surface_record(record)
            records[source.sample_id] = source
    return records


def _iter_articulator_sources(
    surface: ModelDataSurface,
    *,
    limit: int | None,
):
    path = surface.root / "sources.jsonl"
    if not path.is_file():
        raise ArticulatorAwareError("articulator surface is missing sources.jsonl.")
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and emitted >= int(limit):
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ArticulatorAwareError("articulator sources.jsonl rows must be objects.")
            yield _articulator_source_from_surface_record(record)
            emitted += 1


def _articulator_source_from_surface_record(record: Mapping[str, object]) -> ArticulatorSourceSample:
    import numpy as np

    from text_to_sign_production.modeling.backbones.bfh_vectorization import BfhVectorizedPose

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


def _stable_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _compute_precision_policy(context: ModelStageExecutionContext) -> str:
    precision = context.request.compute_profile.get("precision")
    if isinstance(precision, Mapping):
        policy = precision.get("policy")
        if isinstance(policy, str) and policy.strip():
            return policy
    return "auto"


def _precision_metadata(
    context: ModelStageExecutionContext,
    config: ArticulatorAwareConfig,
    *,
    precision_applied: bool,
) -> dict[str, object]:
    resolved = resolve_precision_policy(
        _compute_precision_policy(context),
        _resolve_device(config.training.device),
    )
    return resolved.to_metadata(
        precision_applied=bool(precision_applied) and resolved.autocast_enabled
    )


def _run_seed(effective: Mapping[str, object]) -> int:
    seed_policy = effective.get("seed_policy")
    if not isinstance(seed_policy, Mapping):
        raise ArticulatorAwareError("effective config is missing seed_policy.")
    value = seed_policy.get("run_seed")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ArticulatorAwareError("effective run seed must be a non-negative integer.")
    return value


def _source_from_manifest_sample(sample) -> ArticulatorSourceSample:
    vectorized = vectorize_bfh_pose_arrays(sample.pose, sample_id=sample.sample_id)
    return ArticulatorSourceSample(
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


def _required_support_paths(topology, run_name: str) -> dict[str, Path]:
    root = topology.models.model_intermediate_root(
        ARTICULATOR_AWARE_MODEL_KEY, run_name, "articulator"
    ).path
    paths = {
        "partition_policy": root / "partition_policy.json",
        "mask_strategy": root / "mask_strategy.json",
        "loss_weighting_policy": root / "loss_weighting_policy.json",
        "model_config": root / "model_config.json",
        "length_predictor_config": root / "length_predictor_config.json",
    }
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.is_file()]
    if missing:
        raise ArticulatorAwareError(
            "articulator_aware inference requires restored support artifacts; missing "
            + "; ".join(missing)
        )
    return paths


def _verify_support_policy_matches_config(*, config, policy, mask_strategy, weighting):
    expected = build_articulator_partition_policy(
        config=config.partition_policy, layout=default_bfh_tensor_layout()
    )
    if policy.to_dict() != expected.to_dict():
        raise ArticulatorAwareError("restored partition_policy.json does not match effective config.")
    if mask_strategy.to_dict() != config.mask_strategy.to_dict():
        raise ArticulatorAwareError("restored mask_strategy.json does not match effective config.")
    expected_weighting = build_channel_loss_weighting_policy(config.loss_weighting)
    if weighting.to_dict() != expected_weighting.to_dict():
        raise ArticulatorAwareError(
            "restored loss_weighting_policy.json does not match effective config."
        )


def _verify_support_configs(
    paths,
    *,
    config,
    policy,
    text_embedding_dim: int | None = None,
    text_encoder_metadata_payload: Mapping[str, object] | None = None,
):
    model = _read_json(paths["model_config"], "model_config.json")
    length = _read_json(paths["length_predictor_config"], "length_predictor_config.json")
    expected_model = build_articulator_model_config_payload(
        config,
        policy,
        text_embedding_dim=text_embedding_dim,
        text_encoder_metadata_payload=text_encoder_metadata_payload,
    )
    expected_length = build_articulator_length_config_payload(
        config,
        text_embedding_dim=text_embedding_dim,
    )
    if model != expected_model:
        raise ArticulatorAwareError(
            "restored model_config.json does not exactly match the effective articulator config."
        )
    if length != expected_length:
        raise ArticulatorAwareError(
            "restored length_predictor_config.json does not exactly match the effective articulator config."
        )
    return model, length


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArticulatorAwareError(f"restored {label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ArticulatorAwareError(f"restored {label} must contain a JSON object.")
    return value


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _write_stage_report(path: Path, title: str, lines: list[str]) -> None:
    from text_to_sign_production.modeling.candidates.articulator_aware.reports import (
        write_articulator_markdown_report,
    )

    write_articulator_markdown_report(path, title=title, lines=lines)


__all__ = ["ArticulatorAwareProvider", "_articulator_aware_performance_metadata"]
