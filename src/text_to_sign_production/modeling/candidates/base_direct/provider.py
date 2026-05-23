"""Stage 4 provider implementation for the direct M0 baseline."""

from __future__ import annotations

from pathlib import Path

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
)
from text_to_sign_production.modeling.candidates.artifacts import (
    GeneratedPoseManifestArtifactSubtype,
    generated_pose_manifest_artifact_ref,
    generated_pose_surface,
)
from text_to_sign_production.modeling.candidates.runtime_support import (
    support_artifact_from_model_run_file,
)
from text_to_sign_production.modeling.candidates.base_direct.config import (
    BaseDirectRunConfig,
    base_direct_config_from_effective_dict,
    base_direct_config_to_dict,
    load_base_direct_config,
)
from text_to_sign_production.modeling.candidates.base_direct.spec import require_base_direct_spec
from text_to_sign_production.modeling.research import ModelKey, ModelSpec
from text_to_sign_production.modeling.training.stages import ProviderStageArtifactRole


class BaseDirectProvider:
    """Concrete M0 direct text/transcript-to-pose baseline provider."""

    @property
    def spec(self) -> ModelSpec:
        return require_base_direct_spec()

    @property
    def full_data_pipeline_capability(self) -> ProviderFullDataPipelineCapability:
        return ProviderFullDataPipelineCapability(
            provider_key="base_direct",
            full_training_data_mode="lazy_dataloader",
            verified=True,
            verification_evidence=(
                "behavior:lazy_dataloader:no_full_payload_load_before_first_batch",
                "progress:known_totals",
            ),
            covered_stages=("train", "generate", "export_generated_pose"),
            limitations=(),
        )

    @property
    def full_data_pipeline_declaration(self) -> ProviderFullDataPipelineDeclaration:
        capability = self.full_data_pipeline_capability
        return ProviderFullDataPipelineDeclaration(
            provider_key="base_direct",
            mode="lazy_dataloader",
            covered_stages=capability.covered_stages,
        )

    def calibration_policy(
        self,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ProviderCalibrationPolicy:
        if loaded_config.model_key is not ModelKey.BASE_DIRECT:
            raise ModelStageExecutionError("base_direct calibration policy requires base_direct config.")
        application = loaded_config.effective_config.get("compute_profile_application")
        if not isinstance(application, dict):
            raise ModelStageExecutionError(
                "base_direct calibration policy requires compute_profile_application."
            )
        candidate_keys = tuple(application.get("calibration_candidate_keys", ()))
        override_targets = application.get("calibration_override_targets")
        if not isinstance(override_targets, dict):
            raise ModelStageExecutionError(
                "base_direct calibration policy requires calibration_override_targets."
            )
        return ProviderCalibrationPolicy(
            provider_key="base_direct",
            supports_provider_real=True,
            required_run_modes=("full",),
            required_compute_profiles=("colab_a100_80gb",),
            candidate_keys=candidate_keys,
            override_targets=dict(override_targets),
            representative_surface_kinds={"batch_size": "base_direct_lazy_batch"},
            representative_split="train",
            max_samples=32,
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
        from text_to_sign_production.modeling.candidates.base_direct.calibration import (
            build_representative_calibration_surface,
        )

        return build_representative_calibration_surface(
            context=context,
            candidate_key=candidate_key,
            policy=policy,
            progress_session=progress_session,
        )

    def load_config(self, request: ModelRunRequest) -> ModelProviderLoadedConfig:
        if request.model_key is not ModelKey.BASE_DIRECT:
            raise ModelStageExecutionError("BaseDirectProvider only accepts base_direct requests.")
        if request.auxiliary_objectives:
            raise ModelStageExecutionError("base_direct does not support auxiliary objectives.")
        config = load_base_direct_config(request.config_path, request=request)
        effective_config = {
            **base_direct_config_to_dict(config),
            "schema_version": "t2sp-base-direct-effective-config-v1",
            "model_key": request.model_key.value,
            "run_name": request.run_name,
            "source_path": None if config.source_path is None else str(config.source_path),
            "manifest_family": request.manifest_family.family_id,
            "run_mode": request.run_mode.value,
            "compute_profile": dict(request.compute_profile),
            "auxiliary_objectives": [
                objective.value for objective in request.auxiliary_objectives
            ],
            "objective_config_paths": {
                objective.value: str(path)
                for objective, path in request.objective_config_paths.items()
            },
            "compute_profile_active_overrides": dict(
                config.compute_profile_active_overrides
            ),
        }
        return ModelProviderLoadedConfig(
            model_key=ModelKey.BASE_DIRECT,
            source_path=config.source_path,
            raw_config=config.raw_config,
            effective_config=effective_config,
        )

    def plan_stages(
        self,
        request: ModelRunRequest,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ModelStagePlan:
        if request.model_key is not ModelKey.BASE_DIRECT:
            raise ModelStageExecutionError(
                "base_direct stage planning requires base_direct request."
            )
        if loaded_config.model_key is not ModelKey.BASE_DIRECT:
            raise ModelStageExecutionError("base_direct loaded config has the wrong model key.")
        return default_stage_plan_for_request(request)

    def execute_stage(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
    ) -> ModelStageResult:
        config = self._effective_config(context)
        if stage.spec.kind is ModelStageKind.TRAIN:
            return self._execute_train(stage, context, config)
        if stage.spec.kind is ModelStageKind.GENERATE:
            return self._execute_generate(stage, context, config)
        if stage.spec.kind is ModelStageKind.EXPORT_GENERATED_POSE:
            return self._execute_export(stage, context, config)
        raise ModelStageExecutionError(
            f"base_direct does not implement stage {stage.spec.kind.value!r}."
        )

    def write_model_reports(
        self,
        context: ModelStageExecutionContext,
        results: ModelExecutionResult,
    ) -> tuple[ModelStageArtifactRef, ...]:
        from text_to_sign_production.modeling.candidates.base_direct.reports import (
            write_base_direct_reports,
        )

        return write_base_direct_reports(context=context, results=results)

    def runtime_support_artifacts(
        self,
        *,
        request: ModelRunRequest,
        loaded_config: ModelProviderLoadedConfig,
        execution: ModelExecutionResult,
        topology,
    ) -> tuple[ModelRuntimeSupportArtifact, ...]:
        del loaded_config, execution
        path = (
            topology.models.model_intermediate_root(
                "base_direct",
                request.run_name,
                "config",
            ).path
            / "baseline_training_compat.yaml"
        )
        return (
            support_artifact_from_model_run_file(
                model_run_root=topology.models.model_run_root(
                    "base_direct",
                    request.run_name,
                ).path,
                path=path,
                role="base_direct_compatibility_config",
                provider_key="base_direct",
            ),
        )

    def infer_single_sample(
        self,
        context: ModelSingleSampleInferenceContext,
    ) -> ModelSingleSampleInferenceResult:
        from text_to_sign_production.modeling.candidates.base_direct.exporter import (
            infer_base_direct_single_sample,
        )

        config = base_direct_config_from_effective_dict(
            context.loaded_config.effective_config,
            source_path=_effective_source_path(context.loaded_config.effective_config),
        )
        return infer_base_direct_single_sample(context=context, config=config)
    def _effective_config(self, context: ModelStageExecutionContext) -> BaseDirectRunConfig:
        return base_direct_config_from_effective_dict(
            context.loaded_config.effective_config,
            source_path=_effective_source_path(context.loaded_config.effective_config),
        )

    def _execute_train(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: BaseDirectRunConfig,
    ) -> ModelStageResult:
        from text_to_sign_production.modeling.candidates.base_direct.trainer import (
            run_base_direct_training_stage,
        )

        result = run_base_direct_training_stage(context=context, config=config)
        artifacts: list[ModelStageArtifactRef] = [
            ModelStageArtifactRef(
                role=ProviderStageArtifactRole.CHECKPOINT_LAST,
                path=result.last_checkpoint_path,
                kind="model_checkpoint",
            ),
            ModelStageArtifactRef(
                role=ProviderStageArtifactRole.TRAINING_METRICS,
                path=result.metrics_path,
                kind="model_training_metrics",
            ),
            ModelStageArtifactRef(
                role="training_summary",
                path=result.summary_path,
                kind="model_training_summary",
            ),
            ModelStageArtifactRef(
                role="training_live_log",
                path=result.live_log_path,
                kind="model_training_log",
            ),
        ]
        if result.best_checkpoint_path is not None:
            artifacts.append(
                ModelStageArtifactRef(
                    role=ProviderStageArtifactRole.CHECKPOINT_BEST,
                    path=result.best_checkpoint_path,
                    kind="model_checkpoint",
                )
            )
        if result.compatibility_config_path is not None:
            artifacts.append(
                ModelStageArtifactRef(
                    role="compatibility_config",
                    path=result.compatibility_config_path,
                    kind="model_intermediate",
                )
            )
        if result.target_standardization_path is not None:
            artifacts.append(
                ModelStageArtifactRef(
                    role="target_standardization",
                    path=result.target_standardization_path,
                    kind="model_intermediate",
                )
            )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=tuple(artifacts),
            metrics={
                key: value
                for key, value in (
                    ("final_train_loss", result.final_train_loss),
                    ("final_validation_loss", result.final_validation_loss),
                    ("best_metric_value", result.best_metric_value),
                )
                if value is not None
            },
            metadata={
                **_base_direct_performance_metadata(config),
                "train_sample_count": result.train_sample_count,
                "validation_sample_count": result.validation_sample_count,
                "final_train_loss": result.final_train_loss,
                "final_validation_loss": result.final_validation_loss,
                "best_metric_name": result.best_metric_name,
                "best_metric_value": result.best_metric_value,
                "completed_epoch": result.completed_epoch,
                "last_checkpoint_path": str(result.last_checkpoint_path),
                "best_checkpoint_path": (
                    None
                    if result.best_checkpoint_path is None
                    else str(result.best_checkpoint_path)
                ),
                "config_hash": result.config_hash,
            },
        )

    def _execute_generate(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: BaseDirectRunConfig,
    ) -> ModelStageResult:
        from text_to_sign_production.core.io import write_json
        from text_to_sign_production.modeling.candidates.base_direct.predictor import (
            select_base_direct_checkpoint,
        )

        checkpoint_path = select_base_direct_checkpoint(context)
        plan_path = (
            context.topology.models.model_intermediate_root(
                "base_direct",
                context.request.run_name,
                "generation",
            ).path
            / "plan.json"
        )
        write_json(
            plan_path,
            {
                "model_key": "base_direct",
                "run_name": context.request.run_name,
                "checkpoint_path": str(checkpoint_path),
                "prediction_splits": [split.value for split in config.data.prediction_splits],
                "generation_mode": "deterministic",
                "length_policy": config.identity.length_policy,
                "channel_policy": config.identity.channel_policy,
                "confidence_policy": config.identity.confidence_policy,
            },
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role="generation_plan",
                    path=plan_path,
                    kind="model_intermediate",
                ),
            ),
            metadata={
                **_base_direct_performance_metadata(config),
                "checkpoint_path": str(checkpoint_path),
            },
        )

    def _execute_export(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: BaseDirectRunConfig,
    ) -> ModelStageResult:
        from text_to_sign_production.modeling.candidates.base_direct.exporter import (
            export_base_direct_generated_pose,
        )

        result = export_base_direct_generated_pose(context=context, config=config)
        artifacts: list[ModelStageArtifactRef] = []
        for split in result.splits:
            artifacts.append(
                generated_pose_manifest_artifact_ref(
                    path=split.manifest_path,
                    split=split.split,
                    artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
                    generation_mode="deterministic",
                    producer_stage="export_generated_pose",
                    model_key="base_direct",
                    kind="generated_pose_manifest",
                    description=f"Generated-pose manifest for split {split.split.value!r}.",
                )
            )
        surfaces = tuple(
            generated_pose_surface(
                manifest_path=split.manifest_path,
                samples_dir=split.manifest_path.parent / "samples",
                split=split.split,
                model_key="base_direct",
                model_run_name=context.request.run_name,
                manifest_family=context.request.manifest_family.family_id,
                producer_stage_id="export_generated_pose",
                artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
            )
            for split in result.splits
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=tuple(artifacts),
            generated_pose_surfaces=surfaces,
            metadata={
                **_base_direct_performance_metadata(config),
                "checkpoint_path": str(result.checkpoint_path),
                "total_sample_count": result.total_sample_count,
                "split_sample_counts": {
                    split.split.value: split.sample_count for split in result.splits
                },
                "manifest_paths": {
                    split.split.value: str(split.manifest_path) for split in result.splits
                },
            },
        )


def _effective_source_path(effective_config) -> Path | None:
    value = dict(effective_config).get("source_path")
    if isinstance(value, str) and value.strip():
        return Path(value)
    return None


def _base_direct_performance_metadata(
    config: BaseDirectRunConfig,
) -> dict[str, object]:
    return {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "persistent_workers": config.training.persistent_workers,
        "prefetch_factor": config.training.prefetch_factor,
        "non_blocking_transfers": config.training.non_blocking_transfers,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "mixed_precision": config.training.mixed_precision,
    }


__all__ = ["BaseDirectProvider", "_base_direct_performance_metadata"]
