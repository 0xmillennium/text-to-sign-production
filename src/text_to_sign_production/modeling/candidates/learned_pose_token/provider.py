"""Phase 6 provider implementation for learned_pose_token."""

from __future__ import annotations

import json
import random
import hashlib
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    read_bfh_standardization_stats_json,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    default_bfh_tensor_layout,
    flatten_bfh_vectorized_pose,
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
)
from text_to_sign_production.modeling.candidates.artifacts import (
    GeneratedPoseManifestArtifactSubtype,
    generated_pose_manifest_artifact_ref,
    generated_pose_surface,
)
from text_to_sign_production.modeling.candidates.runtime_support import (
    support_artifact_from_model_run_file,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    LearnedPoseTokenConfig,
    learned_pose_token_config_from_effective_dict,
    load_learned_pose_token_config,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.dataset import (
    PoseTokenSourceSample,
    PoseTokenTrainingSample,
    build_pose_token_training_surface,
    build_pose_token_training_samples,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.exporter import (
    DecodedPoseSample,
    decode_token_sequence_to_bfh_pose,
    export_predicted_pose_samples,
    export_predicted_pose_samples_incremental,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.io import (
    read_pose_token_manifest_jsonl,
    read_pose_token_sequence_npz,
    write_pose_token_manifest_jsonl,
    write_pose_token_sequence_npz,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.model import (
    LearnedPoseTokenizer,
    build_learned_pose_tokenizer,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.predictor import (
    TextToTokenPredictor,
    build_text_to_token_predictor,
    compute_text_to_token_losses,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.spec import (
    LEARNED_POSE_TOKEN_MODEL_KEY,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.tokens import (
    POSE_TOKEN_MANIFEST_SCHEMA_VERSION,
    POSE_TOKEN_SCHEMA_VERSION,
    PoseTokenManifestEntry,
    PoseTokenSequence,
    generated_frame_count_from_token_count,
    pose_token_temporal_spec,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.trainer import (
    PoseTokenizerTrainingResult,
    train_pose_tokenizer,
)
from text_to_sign_production.modeling.artifacts import (
    GeneratedPoseSplitWriteResult,
    GeneratedPoseStreamWriteResult,
    generated_manifest_entry_from_record,
    generated_manifest_entry_from_sample,
    generated_manifest_entry_to_record,
    load_generated_pose_payload,
    validate_generated_pose_manifest_entries,
    write_generated_pose_payload,
    write_generated_pose_split,
)
from text_to_sign_production.modeling.data.temporal_windows import temporal_window_starts
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
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticConsistencyObjectiveConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.training import (
    compute_semantic_training_loss,
    load_semantic_training_objective_for_request,
    merge_overlapping_windows_torch,
)
from text_to_sign_production.modeling.research import ModelKey, ModelSpec
from text_to_sign_production.modeling.registry import require_model_spec
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
class TextToTokenTrainingResult:
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_metadata_path: Path
    last_metadata_path: Path
    metrics_path: Path
    predictor_config_path: Path
    final_train_loss: float
    validation_token_loss: float
    completed_epoch: int
    global_step: int
    runtime_trace: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class DecodeToPoseResult:
    predicted_token_manifest_path: Path
    predicted_token_sample_count: int
    decoded_generated_pose_manifest_path: Path
    decoded_samples: tuple[DecodedPoseSample, ...] = ()
    decoded_sample_count: int = 0
    predicted_token_counts: Mapping[str, int] | None = None
    decoded_frame_counts: Mapping[str, int] | None = None


class LearnedPoseTokenProvider:
    """Concrete Phase 6 learned pose-token provider."""

    @property
    def spec(self) -> ModelSpec:
        return require_model_spec(ModelKey.LEARNED_POSE_TOKEN)

    @property
    def full_data_pipeline_capability(self) -> ProviderFullDataPipelineCapability:
        return ProviderFullDataPipelineCapability(
            provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
            full_training_data_mode="streaming_sharded",
            verified=True,
            verification_evidence=(
                "behavior:fit_representation:surface_reader",
                "behavior:train_text_to_token:surface_reader",
                "behavior:evaluate_reconstruction:incremental_writer",
                "behavior:decode_to_pose:incremental_writer",
                "progress:known_totals",
            ),
            covered_stages=(
                "fit_representation",
                "evaluate_reconstruction",
                "train_text_to_token",
                "decode_to_pose",
                "export_generated_pose",
            ),
            limitations=(),
        )

    @property
    def full_data_pipeline_declaration(self) -> ProviderFullDataPipelineDeclaration:
        capability = self.full_data_pipeline_capability
        return ProviderFullDataPipelineDeclaration(
            provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
            mode="streaming_sharded",
            covered_stages=capability.covered_stages,
        )

    def calibration_policy(
        self,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ProviderCalibrationPolicy:
        if loaded_config.model_key is not ModelKey.LEARNED_POSE_TOKEN:
            raise ModelStageExecutionError(
                "learned_pose_token calibration policy requires learned_pose_token config."
            )
        application = loaded_config.effective_config.get("compute_profile_application")
        if not isinstance(application, Mapping):
            raise ModelStageExecutionError(
                "learned_pose_token calibration policy requires compute_profile_application."
            )
        candidate_keys = tuple(application.get("calibration_candidate_keys", ()))
        override_targets = application.get("calibration_override_targets")
        if not isinstance(override_targets, Mapping):
            raise ModelStageExecutionError(
                "learned_pose_token calibration policy requires calibration_override_targets."
            )
        return ProviderCalibrationPolicy(
            provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
            supports_provider_real=True,
            required_run_modes=("full",),
            required_compute_profiles=("colab_a100_80gb",),
            candidate_keys=candidate_keys,
            override_targets=dict(override_targets),
            representative_surface_kinds={
                "tokenizer_batch_size": "pose_token_units",
                "text_to_token_batch_size": "text_to_token_sequences",
                "reconstruction_batch_size": "pose_token_units",
                "decode_batch_size": "text_to_token_sequences",
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
        from text_to_sign_production.modeling.candidates.learned_pose_token.calibration import (
            build_representative_calibration_surface,
        )

        return build_representative_calibration_surface(
            context=context,
            candidate_key=candidate_key,
            policy=policy,
            progress_session=progress_session,
        )

    def load_config(self, request: ModelRunRequest) -> ModelProviderLoadedConfig:
        if request.model_key is not ModelKey.LEARNED_POSE_TOKEN:
            raise ModelStageExecutionError(
                "LearnedPoseTokenProvider only accepts learned_pose_token requests."
            )
        if request.config_path is None:
            raise ModelStageExecutionError(
                "learned_pose_token requires an explicit config snapshot path. "
                "Set MODEL_CONFIG_RELATIVE_PATH to configs/modeling/learned_pose_token.yaml."
            )
        config = load_learned_pose_token_config(request.config_path, request=request)
        effective = config.to_dict()

        text_encoder_payload = dict(effective["text_encoder"])
        text_encoder_payload["output_dim"] = resolve_text_encoder_output_dim(config.text_encoder)
        effective["text_encoder"] = text_encoder_payload

        return ModelProviderLoadedConfig(
            model_key=ModelKey.LEARNED_POSE_TOKEN,
            source_path=config.source_path,
            raw_config=config.raw_config,
            effective_config={
                **effective,
                "schema_version": "t2sp-learned-pose-token-effective-config-v1",
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
                "compute_profile_active_overrides": _compute_profile_active_from_config(config),
                "compute_profile_dataloader_overrides": dict(
                    config.compute_profile_dataloader_overrides
                ),
                "compute_profile_application": dict(config.compute_profile_application),
            },
        )

    def plan_stages(
        self,
        request: ModelRunRequest,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ModelStagePlan:
        if request.model_key is not ModelKey.LEARNED_POSE_TOKEN:
            raise ModelStageExecutionError("learned_pose_token planning requires learned_pose_token.")
        if loaded_config.model_key is not ModelKey.LEARNED_POSE_TOKEN:
            raise ModelStageExecutionError("loaded config has the wrong model key.")
        return default_stage_plan_for_request(request)

    def execute_stage(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
    ) -> ModelStageResult:
        config = self._effective_config(context.loaded_config)
        try:
            if stage.spec.kind is ModelStageKind.FIT_REPRESENTATION:
                return self._fit_representation(stage, context, config)
            if stage.spec.kind is ModelStageKind.EVALUATE_RECONSTRUCTION:
                return self._evaluate_reconstruction(stage, context, config)
            if stage.spec.kind is ModelStageKind.TRAIN_TEXT_TO_TOKEN:
                return self._train_text_to_token(stage, context, config)
            if stage.spec.kind is ModelStageKind.DECODE_TO_POSE:
                return self._decode_to_pose(stage, context, config)
            if stage.spec.kind is ModelStageKind.EXPORT_GENERATED_POSE:
                return self._export_generated_pose(stage, context, config)
        except LearnedPoseTokenError as exc:
            raise ModelStageExecutionError(str(exc)) from exc
        raise ModelStageExecutionError(
            f"learned_pose_token does not implement stage {stage.spec.kind.value!r}."
        )

    def write_model_reports(
        self,
        context: ModelStageExecutionContext,
        results: ModelExecutionResult,
    ) -> tuple[ModelStageArtifactRef, ...]:
        from text_to_sign_production.modeling.candidates.learned_pose_token.reports import (
            write_learned_pose_token_reports,
        )

        return write_learned_pose_token_reports(context=context, results=results)

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
            LEARNED_POSE_TOKEN_MODEL_KEY,
            request.run_name,
        ).path
        representation = topology.models.model_intermediate_root(
            LEARNED_POSE_TOKEN_MODEL_KEY,
            request.run_name,
            "representation",
        ).path
        return (
            support_artifact_from_model_run_file(
                model_run_root=root,
                path=representation / "checkpoints" / "tokenizer.pt",
                role="learned_pose_token_tokenizer_checkpoint",
                provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
            ),
            support_artifact_from_model_run_file(
                model_run_root=root,
                path=representation / "standardization_stats.json",
                role="learned_pose_token_standardization_stats",
                provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
            ),
        )

    def infer_single_sample(
        self,
        context: ModelSingleSampleInferenceContext,
    ) -> ModelSingleSampleInferenceResult:
        config = self._effective_config(context.loaded_config)
        if context.checkpoint_path.is_file() is not True:
            raise ModelStageExecutionError(
                f"learned_pose_token selected text-to-token checkpoint is missing: "
                f"{context.checkpoint_path}. Restore the model run and selected checkpoint."
            )
        checkpoint = _load_torch_checkpoint(context.checkpoint_path)
        tokenizer_checkpoint_path = _canonical_restored_support_path(
            checkpoint,
            "tokenizer_checkpoint_path",
            canonical_path=context.topology.models.model_intermediate_root(
                LEARNED_POSE_TOKEN_MODEL_KEY,
                context.request.run_name,
                "representation",
            ).path
            / "checkpoints"
            / "tokenizer.pt",
            runtime_root=context.topology.repo_root,
        )
        standardization_stats_path = _canonical_restored_support_path(
            checkpoint,
            "standardization_stats_path",
            canonical_path=context.topology.models.model_intermediate_root(
                LEARNED_POSE_TOKEN_MODEL_KEY,
                context.request.run_name,
                "representation",
            ).path
            / "standardization_stats.json",
            runtime_root=context.topology.repo_root,
        )
        tokenizer = _load_tokenizer(tokenizer_checkpoint_path, config=config)
        stats = read_bfh_standardization_stats_json(standardization_stats_path)
        text_encoder = build_text_encoder(config.text_encoder)
        predictor = _predictor_from_checkpoint(
            checkpoint,
            config=config,
            text_encoder_output_dim=text_encoder.output_dim,
        )
        source = _source_from_manifest_sample(context.sample)
        embedding = text_encoder.encode_one(
            source.text,
            sentence_name=source.source_sentence_name,
        ).pooled_embedding
        token_ids, lengths = _predict_tokens(
            predictor,
            np.asarray([embedding], dtype=np.float32),
        )
        predicted_count = int(lengths[0])
        sequence = PoseTokenSequence(
            schema_version=POSE_TOKEN_SCHEMA_VERSION,
            sample_id=source.sample_id,
            source_sentence_name=source.source_sentence_name,
            split=source.split,
            token_ids=token_ids[0, :predicted_count],
            frame_count=generated_frame_count_from_token_count(
                predicted_count,
                config.tokenizer.temporal_granularity,
                config.tokenizer.window_size,
                config.tokenizer.stride,
            ),
            token_count=predicted_count,
            codebook_size=config.codebook.size,
            temporal_granularity=config.tokenizer.temporal_granularity,
            window_size=config.tokenizer.window_size,
            stride=config.tokenizer.stride,
        )
        pose = decode_token_sequence_to_bfh_pose(
            tokenizer=tokenizer,
            token_sequence=sequence,
            standardization_stats=stats,
            source=source,
            confidence_policy=config.generation.confidence_policy,
        )
        decoded = DecodedPoseSample(source=source, token_sequence=sequence, pose=pose)
        written = export_predicted_pose_samples(
            decoded_samples=(decoded,),
            output_root=context.output_root,
            producer_key=LEARNED_POSE_TOKEN_MODEL_KEY,
            producer_stage="test_model_single_sample",
            run_name=context.request.run_name,
            split=source.split,
            generation_mode=config.generation.generation_mode,
            seed=context.request.seed,
        )
        if len(written.payload_paths) != 1:
            raise ModelStageExecutionError("learned_pose_token single-sample inference wrote no payload.")
        return ModelSingleSampleInferenceResult(
            model_key=context.request.model_key,
            run_name=context.request.run_name,
            checkpoint_path=context.checkpoint_path,
            sample_id=source.sample_id,
            generated_sample=_load_generated_sample_for_result(written.payload_paths[0]),
            generated_payload_path=written.payload_paths[0],
            generated_manifest_path=written.manifest_path,
            metadata={
                "predicted_token_count": predicted_count,
                "decoded_frame_count": generated_frame_count_from_token_count(
                    predicted_count,
                    config.tokenizer.temporal_granularity,
                    config.tokenizer.window_size,
                    config.tokenizer.stride,
                ),
                "temporal_granularity": config.tokenizer.temporal_granularity,
                "window_size": config.tokenizer.window_size,
                "stride": config.tokenizer.stride,
                "generation_mode": config.generation.generation_mode,
                "producer_stage": "test_model_single_sample",
                "length_policy": config.generation.length_policy,
                "confidence_policy": config.generation.confidence_policy,
                "token_selection": config.generation.token_selection,
                "tokenizer_checkpoint_path": str(tokenizer_checkpoint_path),
                "text_to_token_checkpoint_path": str(context.checkpoint_path),
            },
        )

    def _effective_config(self, loaded_config: ModelProviderLoadedConfig) -> LearnedPoseTokenConfig:
        return learned_pose_token_config_from_effective_dict(
            loaded_config.effective_config,
            source_path=loaded_config.source_path,
        )

    def _fit_representation(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LearnedPoseTokenConfig,
    ) -> ModelStageResult:
        progress = context.provider_progress(stage)
        train_surface, train_cache = _materialize_pose_token_surface(
            context=context,
            config=config,
            split=context.request.train_split,
            max_samples=config.data.limit_train_samples,
            progress=progress,
            operation="materialize_train",
            label="learned_pose_token fit train sources",
            total_semantics="training manifest sources loaded",
        )
        validation_surface, validation_cache = _materialize_pose_token_surface(
            context=context,
            config=config,
            split=context.request.validation_split,
            max_samples=config.data.limit_validation_samples,
            progress=progress,
            operation="materialize_val",
            label="learned_pose_token fit validation sources",
            total_semantics="validation manifest sources loaded",
        )
        result = train_pose_tokenizer(
            config=config,
            train_surface=train_surface,
            validation_surface=validation_surface,
            output_root=_representation_root(context),
            run_name=context.request.run_name,
            seed=context.request.seed,
            progress=progress,
            precision_policy_name=_compute_precision_policy(context),
        )
        artifacts = _representation_artifacts(result)
        stats = read_bfh_standardization_stats_json(result.standardization_stats_path)
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=artifacts,
            metadata={
                "train_frame_count": train_surface.metadata.unit_count,
                "validation_frame_count": validation_surface.metadata.unit_count,
                "train_token_unit_count": train_surface.metadata.unit_count,
                "validation_token_unit_count": validation_surface.metadata.unit_count,
                "train_source_frame_count": train_surface.metadata.unit_count,
                "validation_source_frame_count": validation_surface.metadata.unit_count,
                "temporal_granularity": config.tokenizer.temporal_granularity,
                "window_size": config.tokenizer.window_size,
                "stride": config.tokenizer.stride,
                "tokenizer_checkpoint_path": str(result.tokenizer_checkpoint_path),
                "standardization_stats_path": str(result.standardization_stats_path),
                **_standardization_metadata(stats),
                "train_token_manifest_path": str(result.train_token_manifest_path),
                "validation_token_manifest_path": str(result.validation_token_manifest_path),
                "train_data_surface_path": str(train_surface.root),
                "validation_data_surface_path": str(validation_surface.root),
                **_learned_pose_token_performance_metadata(
                    config,
                    runtime_trace=result.runtime_trace,
                ),
                **_surface_token_unit_metadata(train_surface, validation_surface),
                **_cache_stage_metadata(train_cache, validation_cache),
                **_precision_metadata(context),
            },
        )

    def _evaluate_reconstruction(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LearnedPoseTokenConfig,
    ) -> ModelStageResult:
        rep = _require_representation_result(context)
        tokenizer_checkpoint_path = Path(str(rep.metadata["tokenizer_checkpoint_path"]))
        stats_path = Path(str(rep.metadata["standardization_stats_path"]))
        val_manifest_path = Path(str(rep.metadata["validation_token_manifest_path"]))
        tokenizer = _load_tokenizer(tokenizer_checkpoint_path, config=config)
        stats = read_bfh_standardization_stats_json(stats_path)
        progress = context.provider_progress(stage)
        validation_surface = load_model_data_surface(
            Path(str(rep.metadata["validation_data_surface_path"]))
        )
        sources = _surface_sources_by_id(ModelDataSurfaceReader(validation_surface))
        entries = read_pose_token_manifest_jsonl(val_manifest_path)
        decoded_count = 0
        token_counts: dict[str, int] = {}
        frame_counts: dict[str, int] = {}

        def decoded_iter():
            nonlocal decoded_count
            with progress.task(
                operation="decode_samples",
                label="learned_pose_token reconstruction decode samples",
                unit="sample",
                total=len(entries),
                total_semantics="validation token samples decoded and written",
            ) as task:
                for entry in entries:
                    decoded = _decode_entry(
                        entry,
                        sources=sources,
                        tokenizer=tokenizer,
                        stats=stats,
                        config=config,
                    )
                    decoded_count += 1
                    token_counts[decoded.source.sample_id] = decoded.token_sequence.token_count
                    frame_counts[decoded.source.sample_id] = decoded.token_sequence.frame_count
                    task.advance()
                    yield decoded

        with progress.task(
            operation="write_generated_poses",
            label="learned_pose_token reconstruction write generated poses",
            unit="sample",
            total=len(entries),
            total_semantics="reconstruction generated pose samples written",
        ) as task:
            written = export_predicted_pose_samples_incremental(
                decoded_samples=decoded_iter(),
                output_root=_reconstruction_root(context) / "generated_pose",
                producer_key=LEARNED_POSE_TOKEN_MODEL_KEY,
                producer_stage="evaluate_reconstruction",
                run_name=context.request.run_name,
                split=context.request.validation_split,
                generation_mode="reconstruction",
                seed=context.request.seed,
            )
            task.advance(written.sample_count)
        report_path = _reconstruction_root(context) / "reconstruction_summary.md"
        report_path.write_text(
            "# Learned Pose-Token Reconstruction\n\n"
            "Reconstruction quality is not semantic adequacy. This artifact covers "
            "pose -> learned token/code -> reconstructed pose only.\n",
            encoding="utf-8",
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                generated_pose_manifest_artifact_ref(
                    path=written.manifest_path,
                    kind="model_representation_reconstruction_manifest",
                    split=context.request.validation_split,
                    generation_mode="reconstruction",
                    producer_stage="evaluate_reconstruction",
                    model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
                    artifact_subtype=GeneratedPoseManifestArtifactSubtype.RECONSTRUCTION,
                ),
                ModelStageArtifactRef(
                    role="reconstruction_report",
                    path=report_path,
                    kind="model_representation_reconstruction_report",
                ),
            ),
            generated_pose_surfaces=(
                generated_pose_surface(
                    manifest_path=written.manifest_path,
                    samples_dir=written.manifest_path.parent / "samples",
                    split=context.request.validation_split,
                    model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
                    model_run_name=context.request.run_name,
                    manifest_family=context.request.manifest_family.family_id,
                    producer_stage_id="evaluate_reconstruction",
                    artifact_subtype=GeneratedPoseManifestArtifactSubtype.RECONSTRUCTION,
                ),
            ),
            metadata={
                "reconstruction_generated_pose_manifest_path": str(written.manifest_path),
                "reconstruction_sample_count": written.sample_count,
                "predicted_token_counts": token_counts,
                "decoded_frame_counts": frame_counts,
                "unit_type": "sample",
                "unit_count": written.sample_count,
                **_learned_pose_token_performance_metadata(config),
                **_cache_stage_metadata(
                    {
                        "cache_enabled": True,
                        "cache_hit_count": 1,
                        "materialization_cache_key": validation_surface.metadata.cache_key,
                    }
                ),
                **_precision_metadata(context, precision_applied=False),
            },
        )

    def _train_text_to_token(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LearnedPoseTokenConfig,
    ) -> ModelStageResult:
        rep = _require_representation_result(context)
        train_manifest = Path(str(rep.metadata["train_token_manifest_path"]))
        val_manifest = Path(str(rep.metadata["validation_token_manifest_path"]))
        progress = context.provider_progress(stage)
        train_source_surface = load_model_data_surface(
            Path(str(rep.metadata["train_data_surface_path"]))
        )
        validation_source_surface = load_model_data_surface(
            Path(str(rep.metadata["validation_data_surface_path"]))
        )
        train_surface, train_cache = _materialize_text_to_token_surface(
            context=context,
            config=config,
            token_manifest_path=train_manifest,
            source_surface=train_source_surface,
            split=context.request.train_split,
            progress=progress,
            operation="build_train_text_to_token_surface",
            label="learned_pose_token build train text-to-token surface",
            total_semantics="train token sequences written to tensor surface",
        )
        validation_surface, validation_cache = _materialize_text_to_token_surface(
            context=context,
            config=config,
            token_manifest_path=val_manifest,
            source_surface=validation_source_surface,
            split=context.request.validation_split,
            progress=progress,
            operation="build_validation_text_to_token_surface",
            label="learned_pose_token build validation text-to-token surface",
            total_semantics="validation token sequences written to tensor surface",
        )
        result = _train_text_to_token_from_surfaces(
            config=config,
            train_surface=train_surface,
            validation_surface=validation_surface,
            context=context,
            tokenizer_checkpoint_path=Path(str(rep.metadata["tokenizer_checkpoint_path"])),
            standardization_stats_path=Path(str(rep.metadata["standardization_stats_path"])),
            progress=progress,
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role=ProviderStageArtifactRole.CHECKPOINT_BEST,
                    path=result.best_checkpoint_path,
                    kind="model_checkpoint",
                ),
                ModelStageArtifactRef(
                    role=ProviderStageArtifactRole.CHECKPOINT_LAST,
                    path=result.last_checkpoint_path,
                    kind="model_checkpoint",
                ),
                ModelStageArtifactRef(
                    role=ProviderStageArtifactRole.TRAINING_METRICS,
                    path=result.metrics_path,
                    kind="model_text_to_token_metrics",
                ),
                ModelStageArtifactRef(
                    role="predictor_config",
                    path=result.predictor_config_path,
                    kind="model_intermediate",
                ),
            ),
            metrics={
                "final_train_loss": result.final_train_loss,
                "validation_token_loss": result.validation_token_loss,
                "best_metric_value": result.validation_token_loss,
            },
            metadata={
                "last_checkpoint_path": str(result.last_checkpoint_path),
                "best_checkpoint_path": str(result.best_checkpoint_path),
                "best_metric_name": config.checkpoints.selection_metric,
                "best_metric_value": result.validation_token_loss,
                "completed_epoch": result.completed_epoch,
                "global_step": result.global_step,
                "unit_type": "token_sequence",
                "unit_count": train_surface.metadata.unit_count,
                "train_text_to_token_surface_path": str(train_surface.root),
                "validation_text_to_token_surface_path": str(validation_surface.root),
                **_learned_pose_token_performance_metadata(
                    config,
                    runtime_trace=result.runtime_trace,
                ),
                "train_sequence_count": train_surface.metadata.unit_count,
                "validation_sequence_count": validation_surface.metadata.unit_count,
                "token_unit_count_train": train_surface.metadata.unit_count,
                "token_unit_count_validation": validation_surface.metadata.unit_count,
                **_cache_stage_metadata(train_cache, validation_cache),
                **_precision_metadata(context),
            },
        )

    def _decode_to_pose(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LearnedPoseTokenConfig,
    ) -> ModelStageResult:
        progress = context.provider_progress(stage)
        result, cache_metadata = _decode_validation_text(
            context=context,
            config=config,
            progress=progress,
        )
        return ModelStageResult(
            stage=stage,
            status=ModelStageStatus.COMPLETED,
            artifacts=(
                ModelStageArtifactRef(
                    role="predicted_token_manifest:val",
                    path=result.predicted_token_manifest_path,
                    kind="model_text_to_token_predicted_token_manifest",
                    metadata={
                        "split": "val",
                        "predicted_token_sample_root": str(
                            result.predicted_token_manifest_path.parent
                            / "predicted_token_samples"
                            / "val"
                        ),
                        "predicted_token_sample_count": result.predicted_token_sample_count,
                    },
                ),
                generated_pose_manifest_artifact_ref(
                    path=result.decoded_generated_pose_manifest_path,
                    kind="generated_pose_manifest",
                    split=SampleSplit.VAL,
                    producer_stage="decode_to_pose",
                    generation_mode=config.generation.generation_mode,
                    model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
                    artifact_subtype=GeneratedPoseManifestArtifactSubtype.DECODED_INTERMEDIATE,
                ),
            ),
            generated_pose_surfaces=(
                generated_pose_surface(
                    manifest_path=result.decoded_generated_pose_manifest_path,
                    samples_dir=result.decoded_generated_pose_manifest_path.parent / "samples",
                    split=SampleSplit.VAL,
                    model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
                    model_run_name=context.request.run_name,
                    manifest_family=context.request.manifest_family.family_id,
                    producer_stage_id="decode_to_pose",
                    artifact_subtype=GeneratedPoseManifestArtifactSubtype.DECODED_INTERMEDIATE,
                ),
            ),
            metadata={
                "predicted_token_manifest_path": str(result.predicted_token_manifest_path),
                "decoded_generated_pose_manifest_path": str(
                    result.decoded_generated_pose_manifest_path
                ),
                "decoded_sample_count": result.decoded_sample_count,
                "predicted_token_counts": dict(result.predicted_token_counts or {}),
                "decoded_frame_counts": dict(result.decoded_frame_counts or {}),
                "temporal_granularity": config.tokenizer.temporal_granularity,
                "window_size": config.tokenizer.window_size,
                "stride": config.tokenizer.stride,
                "unit_type": "sample",
                "unit_count": result.decoded_sample_count,
                "decode_batching_limited": False,
                **_learned_pose_token_performance_metadata(config),
                **_cache_stage_metadata(cache_metadata),
                **_precision_metadata(context, precision_applied=False),
                "generation_precision_applied": False,
                "generation_precision_reason": "decode sampling kept in fp32 for numerical stability",
            },
        )

    def _export_generated_pose(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
        config: LearnedPoseTokenConfig,
    ) -> ModelStageResult:
        decoded_stage = _require_stage_result(context, ModelStageKind.DECODE_TO_POSE)
        raw_manifest_path = decoded_stage.metadata.get("decoded_generated_pose_manifest_path")
        if not isinstance(raw_manifest_path, str) or not raw_manifest_path:
            raise ModelStageExecutionError(
                "export_generated_pose requires decode_to_pose metadata "
                "'decoded_generated_pose_manifest_path'. Run decode_to_pose before export."
            )
        decoded_count = int(decoded_stage.metadata.get("decoded_sample_count", 0))
        if decoded_count <= 0:
            raise ModelStageExecutionError(
                "export_generated_pose requires at least one decoded pose payload from decode_to_pose."
            )
        progress = context.provider_progress(stage)
        with progress.task(
            operation="write_final_generated_pose",
            label="learned_pose_token write final generated pose split",
            unit="sample",
            total=decoded_count,
            total_semantics="final validation generated pose samples written",
        ) as task:
            written = _write_generated_pose_payloads_from_manifest_incremental(
                topology=context.topology,
                manifest_path=Path(raw_manifest_path),
                producer_key=LEARNED_POSE_TOKEN_MODEL_KEY,
                run_name=context.request.run_name,
                split=context.request.validation_split,
                on_sample_written=lambda: task.advance(),
            )
        artifacts = [
            generated_pose_manifest_artifact_ref(
                path=written.manifest_path,
                kind="generated_pose_manifest",
                description="Generated validation pose manifest for learned_pose_token.",
                split=context.request.validation_split,
                generation_mode=config.generation.generation_mode,
                producer_stage="export_generated_pose",
                model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
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
                    model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
                    model_run_name=context.request.run_name,
                    manifest_family=context.request.manifest_family.family_id,
                    producer_stage_id="export_generated_pose",
                    artifact_subtype=GeneratedPoseManifestArtifactSubtype.FINAL_VALIDATION,
                ),
            ),
            metadata={
                "manifest_paths": {"val": str(written.manifest_path)},
                "split_sample_counts": {"val": written.sample_count},
                "total_sample_count": written.sample_count,
                "decoded_generated_pose_manifest_path": str(
                    decoded_stage.metadata.get("decoded_generated_pose_manifest_path", "")
                ),
            },
        )


def _compute_profile_active_from_config(
    config: LearnedPoseTokenConfig,
) -> dict[str, object]:
    overrides = config.run_mode_overrides
    active = overrides.get("compute_profile_active")
    if isinstance(active, Mapping):
        return dict(active)
    return {}


def _representation_artifacts(
    result: PoseTokenizerTrainingResult,
) -> tuple[ModelStageArtifactRef, ...]:
    return (
        ModelStageArtifactRef(
            role=ProviderStageArtifactRole.REPRESENTATION_ARTIFACT,
            path=result.tokenizer_checkpoint_path,
            kind="model_representation_tokenizer_checkpoint",
        ),
        ModelStageArtifactRef(
            role="codebook_metadata",
            path=result.codebook_metadata_path,
            kind="model_representation_codebook_metadata",
        ),
        ModelStageArtifactRef(
            role="standardization_stats",
            path=result.standardization_stats_path,
            kind="model_representation_standardization_stats",
        ),
        ModelStageArtifactRef(
            role="token_manifest:train",
            path=result.train_token_manifest_path,
            kind="model_representation_token_manifest",
            metadata={"split": "train"},
        ),
        ModelStageArtifactRef(
            role="token_manifest:val",
            path=result.validation_token_manifest_path,
            kind="model_representation_token_manifest",
            metadata={"split": "val"},
        ),
        ModelStageArtifactRef(
            role=ProviderStageArtifactRole.TRAINING_METRICS,
            path=result.training_metrics_path,
            kind="model_training_metrics",
        ),
        ModelStageArtifactRef(
            role="codebook_stability:train",
            path=result.train_codebook_stability_path,
            kind="model_representation_codebook_stability",
            metadata={"split": "train"},
        ),
        ModelStageArtifactRef(
            role="codebook_stability:val",
            path=result.validation_codebook_stability_path,
            kind="model_representation_codebook_stability",
            metadata={"split": "val"},
        ),
        ModelStageArtifactRef(
            role="tokenizer_checkpoint_metadata",
            path=result.tokenizer_checkpoint_metadata_path,
            kind="model_intermediate",
        ),
        ModelStageArtifactRef(
            role="tokenizer_design_report",
            path=result.tokenizer_design_report_path,
            kind="model_report",
        ),
        ModelStageArtifactRef(
            role="token_manifest_summary",
            path=result.token_manifest_summary_path,
            kind="model_report",
        ),
        ModelStageArtifactRef(
            role="validation_reconstruction_summary_manifest",
            path=result.validation_reconstruction_summary_manifest_path,
            kind="model_representation_reconstruction_manifest",
            metadata={"split": "val"},
        ),
        ModelStageArtifactRef(
            role="representation_reconstruction_summary",
            path=result.reconstruction_summary_path,
            kind="model_representation_reconstruction_report",
        ),
    )


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
                raise LearnedPoseTokenError("sample producer_key/run_name does not match request.")
            if sample.split is not resolved_split:
                raise LearnedPoseTokenError("sample split does not match request.")
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
                raise LearnedPoseTokenError("Invalid generated-pose manifest: duplicate manifest key.")
            seen.add(key)
            if issues:
                raise LearnedPoseTokenError(
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
                    raise LearnedPoseTokenError(
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


def _unique_pose_token_sources(
    samples: Sequence[PoseTokenTrainingSample],
) -> tuple[PoseTokenSourceSample, ...]:
    sources: OrderedDict[str, PoseTokenSourceSample] = OrderedDict()
    for sample in samples:
        sources.setdefault(sample.source.sample_id, sample.source)
    return tuple(sources.values())


def _learned_pose_token_performance_metadata(
    config: LearnedPoseTokenConfig,
    *,
    runtime_trace: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "tokenizer_batch_size": config.training.effective_tokenizer_batch_size,
        "text_to_token_batch_size": config.training.effective_text_to_token_batch_size,
        "reconstruction_batch_size": config.training.effective_reconstruction_batch_size,
        "decode_batch_size": config.training.effective_decode_batch_size,
        "cache_materialized_sources": config.training.cache_materialized_sources,
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


def _token_unit_metadata(
    train: Sequence[PoseTokenTrainingSample],
    validation: Sequence[PoseTokenTrainingSample],
) -> dict[str, object]:
    train_sources = _unique_pose_token_sources(train)
    validation_sources = _unique_pose_token_sources(validation)
    counts = [len(tuple(group)) for group in _samples_grouped_by_source(train).values()]
    return {
        "token_unit_count_train": len(train),
        "token_unit_count_validation": len(validation),
        "train_sequence_count": len(train_sources),
        "validation_sequence_count": len(validation_sources),
        "average_token_units_per_sample": (
            None if not counts else float(sum(counts)) / float(len(counts))
        ),
        "max_token_units_per_sample": None if not counts else max(counts),
    }


def _samples_grouped_by_source(
    samples: Sequence[PoseTokenTrainingSample],
) -> OrderedDict[str, list[PoseTokenTrainingSample]]:
    grouped: OrderedDict[str, list[PoseTokenTrainingSample]] = OrderedDict()
    for sample in samples:
        grouped.setdefault(sample.sample_id, []).append(sample)
    return grouped


def _surface_token_unit_metadata(
    train: ModelDataSurface,
    validation: ModelDataSurface,
) -> dict[str, object]:
    average = (
        None
        if train.metadata.loaded_sample_count <= 0
        else float(train.metadata.unit_count) / float(train.metadata.loaded_sample_count)
    )
    return {
        "token_unit_count_train": train.metadata.unit_count,
        "token_unit_count_validation": validation.metadata.unit_count,
        "train_sequence_count": train.metadata.loaded_sample_count,
        "validation_sequence_count": validation.metadata.loaded_sample_count,
        "average_token_units_per_sample": average,
        "max_token_units_per_sample": None,
        "sharded_surface_missing": False,
    }


def _materialize_pose_token_surface(
    *,
    context: ModelStageExecutionContext,
    config: LearnedPoseTokenConfig,
    split: SampleSplit,
    max_samples: int | None,
    progress,
    operation: str,
    label: str,
    total_semantics: str,
) -> tuple[ModelDataSurface, Mapping[str, object]]:
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
    split_name = SampleSplit(split).value
    cache_key = model_data_surface_cache_key(
        provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        surface_kind="pose_token_units",
        split=split_name,
        source_manifest_sha256=manifest_sha,
        provider_config_sha256=provider_config_sha,
        run_mode=context.request.run_mode.value,
        limit_samples=max_samples,
        data_version=(
            f"{config.tokenizer.temporal_granularity}:"
            f"{config.tokenizer.window_size}:{config.tokenizer.stride}"
        ),
    )
    root = _data_surface_root(context, "pose_token_units", split_name)
    metadata: dict[str, object] = {
        "cache_enabled": True,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "cache_read_seconds": 0.0,
        "cache_write_seconds": 0.0,
        "materialization_elapsed_seconds": 0.0,
        "materialization_cache_key": cache_key,
    }
    if (root / "metadata.json").is_file() and (root / "manifest.jsonl").is_file():
        started = time.perf_counter()
        surface = load_model_data_surface(root)
        if surface.metadata.cache_key == cache_key:
            metadata["cache_hit_count"] = 1
            metadata["cache_read_seconds"] = time.perf_counter() - started
            return surface, metadata
    started = time.perf_counter()
    metadata["cache_miss_count"] = 1
    with progress.task(
        operation=operation,
        label=label,
        unit="sample",
        total=manifest_count,
        total_semantics=total_semantics,
        allowed_counters=("loaded", "shards"),
    ) as task:
        surface = build_pose_token_training_surface(
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
            temporal_spec=config.tokenizer.temporal_window_spec(),
            standardization_missing_observation_policy=(
                config.representation.standardization_missing_observation_policy
            ),
            on_sample_loaded=lambda index, total, sample_id: task.advance(
                1,
                counters={"loaded": index},
            ),
        )
    metadata["materialization_elapsed_seconds"] = time.perf_counter() - started
    metadata["cache_write_seconds"] = metadata["materialization_elapsed_seconds"]
    metadata["cache_write_units_per_second"] = (
        None
        if not metadata["cache_write_seconds"]
        else surface.metadata.unit_count / float(metadata["cache_write_seconds"])
    )
    return surface, metadata


def _materialize_text_to_token_surface(
    *,
    context: ModelStageExecutionContext,
    config: LearnedPoseTokenConfig,
    token_manifest_path: Path,
    source_surface: ModelDataSurface,
    split: SampleSplit,
    progress,
    operation: str,
    label: str,
    total_semantics: str,
) -> tuple[ModelDataSurface, Mapping[str, object]]:
    entries = read_pose_token_manifest_jsonl(token_manifest_path)
    manifest_count = len(entries)
    source_reader = ModelDataSurfaceReader(source_surface)
    sources = _surface_sources_by_id(source_reader)
    provider_config_sha = _stable_sha256(config.to_dict())
    split_name = SampleSplit(split).value
    token_manifest_sha = sha256_file(token_manifest_path)
    cache_key = model_data_surface_cache_key(
        provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        surface_kind="text_to_token_sequences",
        split=split_name,
        source_manifest_sha256=token_manifest_sha,
        provider_config_sha256=provider_config_sha,
        run_mode=context.request.run_mode.value,
        limit_samples=None,
        data_version="text_embedding_to_pose_tokens:v1",
    )
    root = _data_surface_root(context, "text_to_token_sequences", split_name)
    metadata: dict[str, object] = {
        "cache_enabled": True,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "cache_read_seconds": 0.0,
        "cache_write_seconds": 0.0,
        "materialization_elapsed_seconds": 0.0,
        "materialization_cache_key": cache_key,
    }
    if (root / "metadata.json").is_file() and (root / "manifest.jsonl").is_file():
        started = time.perf_counter()
        surface = load_model_data_surface(root)
        if surface.metadata.cache_key == cache_key:
            metadata["cache_hit_count"] = 1
            metadata["cache_read_seconds"] = time.perf_counter() - started
            return surface, metadata
    metadata["cache_miss_count"] = 1
    started = time.perf_counter()
    text_encoder = build_text_encoder(config.text_encoder)
    max_positions = max((entry.token_count for entry in entries), default=0)
    if max_positions <= 0:
        raise LearnedPoseTokenError("text-to-token surface requires at least one token sequence.")
    writer = ModelDataSurfaceWriter(
        root=root,
        provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        surface_kind="text_to_token_sequences",
        split=split_name,
        manifest_family=context.request.manifest_family.family_id,
        source_manifest_path=token_manifest_path,
        source_manifest_sha256=token_manifest_sha,
        provider_config_sha256=provider_config_sha,
        cache_key=cache_key,
        manifest_entry_count=manifest_count,
        run_mode=context.request.run_mode.value,
        data_version="text_embedding_to_pose_tokens:v1",
        max_units_per_shard=10_000,
        max_source_samples_per_shard=512,
    )
    with progress.task(
        operation=operation,
        label=label,
        unit="sample",
        total=manifest_count,
        total_semantics=total_semantics,
        allowed_counters=("token_units",),
    ) as task:
        for source_index, entry in enumerate(entries):
            source = sources.get(entry.sample_id)
            if source is None:
                raise LearnedPoseTokenError(
                    f"text-to-token surface missing source metadata for {entry.sample_id!r}."
                )
            sequence = read_pose_token_sequence_npz(entry.token_path)
            count = int(sequence.token_count)
            target = np.zeros((1, max_positions), dtype=np.int64)
            mask = np.zeros((1, max_positions), dtype=np.bool_)
            target[0, :count] = np.asarray(sequence.token_ids, dtype=np.int64)
            mask[0, :count] = True
            encoded = text_encoder.encode_one(
                source.text,
                sentence_name=source.source_sentence_name,
            ).pooled_embedding.astype(np.float32, copy=False)
            writer.append_units(
                {
                    "text_embedding": encoded.reshape(1, -1),
                    "target_token_ids": target,
                    "target_mask": mask,
                    "source_index": np.asarray([source_index], dtype=np.int64),
                    "sequence_length": np.asarray([count], dtype=np.int64),
                },
                sample_count=1,
                frame_count=count,
            )
            task.advance(1, counters={"token_units": count})
    surface = writer.close()
    metadata["materialization_elapsed_seconds"] = time.perf_counter() - started
    metadata["cache_write_seconds"] = metadata["materialization_elapsed_seconds"]
    return surface, metadata


def _data_surface_root(
    context: ModelStageExecutionContext,
    surface_kind: str,
    split: str,
) -> Path:
    return (
        context.topology.repo_root
        / "runtime"
        / "model"
        / "data_surfaces"
        / context.request.run_name
        / LEARNED_POSE_TOKEN_MODEL_KEY
        / surface_kind
        / split
    )


def _materialize_pose_token_samples(
    *,
    context: ModelStageExecutionContext,
    config: LearnedPoseTokenConfig,
    split: SampleSplit,
    max_samples: int | None,
    progress,
    operation: str,
    label: str,
    total_semantics: str,
) -> tuple[tuple[PoseTokenTrainingSample, ...], Mapping[str, object]]:
    cache_key = _materialization_cache_key(
        context=context,
        config=config,
        split=split,
        max_samples=max_samples,
    )
    cache_root = _materialization_cache_root(context) / cache_key
    split_name = SampleSplit(split).value
    sample_path = cache_root / f"{split_name}_token_units.pt"
    sources_path = cache_root / f"{split_name}_sources.pt"
    metadata_path = cache_root / f"{split_name}_metadata.json"
    cache_metadata: dict[str, object] = {
        "cache_enabled": config.training.cache_materialized_sources,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "cache_read_seconds": 0.0,
        "cache_write_seconds": 0.0,
        "materialization_elapsed_seconds": 0.0,
        "materialization_cache_key": cache_key,
    }
    if config.training.cache_materialized_sources and sample_path.is_file():
        started = time.perf_counter()
        try:
            loaded = torch.load(sample_path, map_location="cpu", weights_only=False)
            metadata = _read_json(metadata_path)
            if not isinstance(metadata, Mapping) or metadata.get("cache_key") != cache_key:
                raise LearnedPoseTokenError("materialization cache metadata key mismatch.")
            if not isinstance(loaded, tuple):
                raise LearnedPoseTokenError("materialization cache payload is invalid.")
            samples = tuple(_sample_from_cache_record(record) for record in loaded)
        except Exception as exc:
            raise LearnedPoseTokenError(
                f"corrupted learned_pose_token materialization cache: {sample_path}"
            ) from exc
        cache_metadata["cache_hit_count"] = 1
        cache_metadata["cache_read_seconds"] = time.perf_counter() - started
        return samples, cache_metadata
    started = time.perf_counter()
    cache_metadata["cache_miss_count"] = 1 if config.training.cache_materialized_sources else 0
    with progress.task(
        operation=operation,
        label=label,
        unit="sample",
        total=max_samples,
        total_semantics=total_semantics,
        allowed_counters=("loaded",),
    ) as task:
        samples = build_pose_token_training_samples(
            topology=context.topology,
            manifest_family=context.request.manifest_family,
            split=split,
            max_samples=max_samples,
            temporal_spec=config.tokenizer.temporal_window_spec(),
            on_sample_loaded=lambda index, total, sample_id: task.advance(
                1,
                counters={"loaded": index},
            ),
        )
    cache_metadata["materialization_elapsed_seconds"] = time.perf_counter() - started
    if config.training.cache_materialized_sources:
        write_started = time.perf_counter()
        cache_root.mkdir(parents=True, exist_ok=True)
        torch.save(tuple(_sample_to_cache_record(sample) for sample in samples), sample_path)
        torch.save(
            tuple(_source_to_cache_record(source) for source in _unique_pose_token_sources(samples)),
            sources_path,
        )
        _write_json(metadata_path, _materialization_cache_metadata(
            context=context,
            config=config,
            split=split,
            cache_key=cache_key,
            samples=samples,
        ))
        cache_metadata["cache_write_seconds"] = time.perf_counter() - write_started
    return samples, cache_metadata


def _materialization_cache_root(context: ModelStageExecutionContext) -> Path:
    return (
        context.topology.repo_root
        / "cache"
        / context.request.run_name
        / LEARNED_POSE_TOKEN_MODEL_KEY
    )


def _materialization_cache_key(
    *,
    context: ModelStageExecutionContext,
    config: LearnedPoseTokenConfig,
    split: SampleSplit,
    max_samples: int | None,
) -> str:
    manifest_path = resolve_modeling_manifest_path(
        context.topology,
        context.request.manifest_family,
        split,
    )
    payload = {
        "manifest_path": str(manifest_path.resolve(strict=False)),
        "manifest_sha256": _file_sha256(manifest_path) if manifest_path.is_file() else None,
        "manifest_family": context.request.manifest_family.family_id,
        "split": SampleSplit(split).value,
        "provider_config_sha256": _stable_sha256(config.to_dict()),
        "standardization_policy": config.representation.to_dict(),
        "bfh_schema_vectorization": {
            "layout": default_bfh_tensor_layout().to_dict(),
            "flattening": "flatten_bfh_vectorized_pose",
        },
        "run_mode": context.request.run_mode.value,
        "max_samples": max_samples,
    }
    return _stable_sha256(payload)


def _materialization_cache_metadata(
    *,
    context: ModelStageExecutionContext,
    config: LearnedPoseTokenConfig,
    split: SampleSplit,
    cache_key: str,
    samples: Sequence[PoseTokenTrainingSample],
) -> dict[str, object]:
    manifest_path = resolve_modeling_manifest_path(
        context.topology,
        context.request.manifest_family,
        split,
    )
    sources = _unique_pose_token_sources(samples)
    return {
        "schema_version": "model.materialization_cache.v1",
        "provider_key": LEARNED_POSE_TOKEN_MODEL_KEY,
        "run_name": context.request.run_name,
        "split": SampleSplit(split).value,
        "cache_key": cache_key,
        "source_manifest_sha256": _file_sha256(manifest_path) if manifest_path.is_file() else None,
        "provider_config_sha256": _stable_sha256(config.to_dict()),
        "created_at": datetime.now(UTC).isoformat(),
        "sample_count": len(sources),
        "frame_count": sum(source.frame_count for source in sources),
    }


def _cache_stage_metadata(*entries: Mapping[str, object]) -> dict[str, object]:
    result = {
        "cache_enabled": any(bool(entry.get("cache_enabled")) for entry in entries),
        "cache_hit_count": sum(int(entry.get("cache_hit_count") or 0) for entry in entries),
        "cache_miss_count": sum(int(entry.get("cache_miss_count") or 0) for entry in entries),
        "cache_read_seconds": sum(float(entry.get("cache_read_seconds") or 0.0) for entry in entries),
        "cache_write_seconds": sum(float(entry.get("cache_write_seconds") or 0.0) for entry in entries),
        "materialization_elapsed_seconds": sum(
            float(entry.get("materialization_elapsed_seconds") or 0.0)
            for entry in entries
        ),
    }
    keys = [entry.get("materialization_cache_key") for entry in entries if entry.get("materialization_cache_key")]
    if keys:
        result["materialization_cache_keys"] = tuple(str(key) for key in keys)
    return result


def _sample_to_cache_record(sample: PoseTokenTrainingSample) -> dict[str, object]:
    return {
        "source": _source_to_cache_record(sample.source),
        "sample_id": sample.sample_id,
        "source_sentence_name": sample.source_sentence_name,
        "split": sample.split.value,
        "frame_index": sample.frame_index,
        "values": np.asarray(sample.values, dtype=np.float32),
        "validity_mask": np.asarray(sample.validity_mask, dtype=np.bool_),
        "token_index": sample.token_index,
        "source_frame_indices": tuple(sample.source_frame_indices or ()),
        "real_frame_mask": tuple(sample.real_frame_mask or ()),
        "temporal_granularity": sample.temporal_granularity,
        "window_size": sample.window_size,
        "stride": sample.stride,
    }


def _sample_from_cache_record(record: object) -> PoseTokenTrainingSample:
    if not isinstance(record, Mapping):
        raise LearnedPoseTokenError("materialization cache sample record must be a mapping.")
    return PoseTokenTrainingSample(
        source=_source_from_cache_record(record.get("source")),
        sample_id=str(record["sample_id"]),
        source_sentence_name=str(record["source_sentence_name"]),
        split=SampleSplit(str(record["split"])),
        frame_index=int(record["frame_index"]),
        values=np.asarray(record["values"], dtype=np.float32),
        validity_mask=np.asarray(record["validity_mask"], dtype=np.bool_),
        token_index=int(record["token_index"]),
        source_frame_indices=tuple(int(value) for value in record["source_frame_indices"]),
        real_frame_mask=tuple(bool(value) for value in record["real_frame_mask"]),
        temporal_granularity=str(record["temporal_granularity"]),
        window_size=int(record["window_size"]),
        stride=int(record["stride"]),
    )


def _source_to_cache_record(source: PoseTokenSourceSample) -> dict[str, object]:
    return {
        "sample_id": source.sample_id,
        "source_sentence_name": source.source_sentence_name,
        "text": source.text,
        "source_video_id": source.source_video_id,
        "source_sentence_id": source.source_sentence_id,
        "reference_payload_ref": source.reference_payload_ref,
        "split": source.split.value,
        "frame_count": source.frame_count,
        "vectorized_pose": {
            "values": np.asarray(source.vectorized_pose.values, dtype=np.float32),
            "validity_mask": np.asarray(source.vectorized_pose.validity_mask, dtype=np.bool_),
            "frame_validity_mask": np.asarray(source.vectorized_pose.frame_validity_mask, dtype=np.bool_),
            "confidence_values": np.asarray(source.vectorized_pose.confidence_values, dtype=np.float32),
            "source_sample_id": source.vectorized_pose.source_sample_id,
        },
    }


def _source_from_cache_record(record: object) -> PoseTokenSourceSample:
    if not isinstance(record, Mapping):
        raise LearnedPoseTokenError("materialization cache source record must be a mapping.")
    vectorized = record.get("vectorized_pose")
    if not isinstance(vectorized, Mapping):
        raise LearnedPoseTokenError("materialization cache vectorized_pose must be a mapping.")
    return PoseTokenSourceSample(
        sample_id=str(record["sample_id"]),
        source_sentence_name=str(record["source_sentence_name"]),
        text=str(record["text"]),
        source_video_id=str(record["source_video_id"]),
        source_sentence_id=str(record["source_sentence_id"]),
        reference_payload_ref=str(record["reference_payload_ref"]),
        split=SampleSplit(str(record["split"])),
        frame_count=int(record["frame_count"]),
        vectorized_pose=BfhVectorizedPose(
            layout=default_bfh_tensor_layout(),
            values=np.asarray(vectorized["values"], dtype=np.float32),
            validity_mask=np.asarray(vectorized["validity_mask"], dtype=np.bool_),
            frame_validity_mask=np.asarray(vectorized["frame_validity_mask"], dtype=np.bool_),
            confidence_values=np.asarray(vectorized["confidence_values"], dtype=np.float32),
            frame_count=int(record["frame_count"]),
            source_sample_id=(
                None
                if vectorized.get("source_sample_id") is None
                else str(vectorized["source_sample_id"])
            ),
        ),
    )


def _stable_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standardization_metadata(stats) -> dict[str, object]:
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


def _compute_precision_policy(context: ModelStageExecutionContext) -> str:
    compute_profile = context.request.compute_profile
    precision = compute_profile.get("precision")
    if isinstance(precision, Mapping):
        policy = precision.get("policy")
        if isinstance(policy, str) and policy.strip():
            return policy
    return "auto"


def _precision_metadata(
    context: ModelStageExecutionContext,
    *,
    precision_applied: bool | None = None,
) -> dict[str, object]:
    policy = _compute_precision_policy(context)
    device = _resolve_device(
        learned_pose_token_config_from_effective_dict(
            context.loaded_config.effective_config,
            source_path=context.loaded_config.source_path,
        ).training.device
    )
    resolved = resolve_precision_policy(policy, device)
    applied = resolved.autocast_enabled if precision_applied is None else (
        bool(precision_applied) and resolved.autocast_enabled
    )
    return resolved.to_metadata(precision_applied=applied)


def _train_text_to_token(
    *,
    config: LearnedPoseTokenConfig,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    train_sources: Mapping[str, PoseTokenSourceSample],
    validation_sources: Mapping[str, PoseTokenSourceSample],
    context: ModelStageExecutionContext,
    tokenizer_checkpoint_path: Path,
    standardization_stats_path: Path,
    progress=None,
) -> TextToTokenTrainingResult:
    _set_seed(config.training.seed if context.request.seed is None else context.request.seed)
    train_sequences = _read_sequences(train_manifest_path)
    validation_sequences = _read_sequences(validation_manifest_path)
    max_positions = max(sequence.token_count for sequence in (*train_sequences, *validation_sequences))
    text_encoder = build_text_encoder(config.text_encoder)
    text_dim = text_encoder.output_dim
    predictor = build_text_to_token_predictor(
        text_embedding_dim=text_dim,
        codebook_size=config.codebook.size,
        config=config,
        max_positions=max_positions,
    )
    device = _resolve_device(config.training.device)
    precision_policy = resolve_precision_policy(_compute_precision_policy(context), device)
    predictor.to(device)
    semantic_config = load_semantic_training_objective_for_request(
        request=context.request,
        model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
    )
    tokenizer = None
    if semantic_config is not None:
        tokenizer = _load_tokenizer(tokenizer_checkpoint_path, config=config).to(device)
        tokenizer.eval()
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    metrics: list[TrainingMetricRecord] = []
    global_step = 0
    final_train_loss = 0.0
    train_batches = _sequence_batches(
        train_sequences,
        config.training.effective_text_to_token_batch_size,
    )
    for epoch in range(config.training.max_epochs):
        predictor.train()
        batch_losses = []
        with progress.task(
            operation=f"text_to_token_train_epoch_{epoch + 1}",
            label=(
                "learned_pose_token text-to-token train epoch "
                f"{epoch + 1}/{config.training.max_epochs}"
            ),
            unit="batch",
            total=len(train_batches),
            total_semantics="text-to-token training batches in epoch",
            allowed_counters=("loss",),
        ) as task:
            for batch in train_batches:
                text_embeddings, target_ids, lengths, mask = _predictor_batch(
                    batch,
                    train_sources,
                    text_encoder=text_encoder,
                    max_positions=max_positions,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision_policy):
                    output = predictor(text_embeddings)
                    losses = compute_text_to_token_losses(
                        output=output,
                        target_token_ids=target_ids,
                        target_lengths=lengths,
                        token_mask=mask,
                        token_loss_weight=config.text_to_token.token_loss_weight,
                        length_loss_weight=config.text_to_token.length_loss_weight,
                    )
                    total_loss = losses["total_loss"]
                    semantic_result = None
                    if semantic_config is not None:
                        if tokenizer is None:
                            raise LearnedPoseTokenError("semantic tokenizer was not loaded.")
                        semantic_result = _soft_semantic_loss_for_token_batch(
                            tokenizer=tokenizer,
                            output=output,
                            batch=batch,
                            sources=train_sources,
                            text_embeddings=text_embeddings,
                            token_mask=mask,
                            config=config,
                            semantic_config=semantic_config,
                        )
                        total_loss = total_loss + semantic_result.weighted_semantic_loss
                total_loss.backward()
                optimizer.step()
                loss_value = float(total_loss.detach().cpu().item())
                batch_losses.append(loss_value)
                metrics.append(_metric("train", "total_loss", loss_value, epoch, global_step))
                if semantic_result is not None:
                    for name, tensor in {
                        "semantic_loss": semantic_result.semantic_loss,
                        "semantic_weighted_loss": semantic_result.weighted_semantic_loss,
                        "semantic_cosine_similarity": semantic_result.cosine_similarity,
                        "semantic_cosine_distance": semantic_result.cosine_distance,
                    }.items():
                        metrics.append(
                            _metric("train", name, float(tensor.detach().cpu().item()), epoch, global_step)
                        )
                task.advance(1, counters={"loss": f"{loss_value:.6g}"})
                global_step += 1
        final_train_loss = float(np.mean(batch_losses))
    validation_token_loss = _validation_token_loss(
        predictor,
        validation_sequences,
        validation_sources,
        text_encoder=text_encoder,
        max_positions=max_positions,
        device=device,
        config=config,
        progress=progress,
        precision_policy=precision_policy,
    )
    metrics.append(
        _metric(
            "val",
            config.checkpoints.selection_metric,
            validation_token_loss,
            config.training.max_epochs - 1,
            global_step,
        )
    )
    checkpoint_root = context.topology.models.model_checkpoints_root(
        LEARNED_POSE_TOKEN_MODEL_KEY,
        context.request.run_name,
    ).path
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_root / "last.pt"
    best_path = checkpoint_root / "best.pt"
    payload = {
        "schema_version": "t2sp-learned-pose-text-to-token-checkpoint-v1",
        "model_key": LEARNED_POSE_TOKEN_MODEL_KEY,
        "run_name": context.request.run_name,
        "predictor_config": {
            "text_embedding_dim": text_dim,
            "codebook_size": config.codebook.size,
            "max_positions": max_positions,
            "hidden_dim": config.text_to_token.hidden_dim,
            "position_dim": config.text_to_token.position_dim,
            "dropout": config.text_to_token.dropout,
            "min_generated_tokens": config.text_to_token.min_generated_tokens,
            "max_generated_tokens": (
                config.text_to_token.max_generated_tokens or max_positions
            ),
        },
        "state_dict": predictor.state_dict(),
        "text_embedding_dim": text_dim,
        "text_encoder_config": config.text_encoder.to_dict(),
        "text_encoder_metadata": text_encoder_metadata(text_encoder),
        "tokenizer_checkpoint_path": str(tokenizer_checkpoint_path),
        "standardization_stats_path": str(standardization_stats_path),
        "tokenizer_config": config.tokenizer.to_dict(),
        "generation": config.generation.to_dict(),
        "validation_token_loss": validation_token_loss,
    }
    payload.update(_semantic_checkpoint_extra(semantic_config))
    torch.save(payload, last_path)
    torch.save(payload, best_path)
    metadata_root = context.topology.models.model_intermediate_root(
        LEARNED_POSE_TOKEN_MODEL_KEY,
        context.request.run_name,
        "text_to_token",
    ).path
    metadata_root.mkdir(parents=True, exist_ok=True)
    predictor_config_path = metadata_root / "predictor_config.json"
    _write_json(predictor_config_path, payload["predictor_config"] | {"schema_version": "t2sp-text-to-token-predictor-config-v1"})
    metrics_path = metadata_root / "training_metrics.jsonl"
    write_training_metric_records_jsonl(metrics_path, metrics)
    best_meta = metadata_root / "best_checkpoint_metadata.json"
    last_meta = metadata_root / "last_checkpoint_metadata.json"
    for role, path, metadata_path in (
        (CheckpointRole.BEST, best_path, best_meta),
        (CheckpointRole.LAST, last_path, last_meta),
    ):
        write_checkpoint_metadata_json(
            metadata_path,
            ModelCheckpointMetadata(
                schema_version=CHECKPOINT_METADATA_SCHEMA_VERSION,
                model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
                run_name=context.request.run_name,
                role=role,
                stage_name="train_text_to_token",
                checkpoint_path=path,
                epoch=config.training.max_epochs - 1,
                global_step=global_step,
                selection_metric=config.checkpoints.selection_metric,
                selection_metric_value=validation_token_loss,
                lower_is_better=config.checkpoints.lower_is_better,
                created_at_utc=_now(),
                extra={
                    "length_policy": config.generation.length_policy,
                    "token_selection": config.generation.token_selection,
                    **_semantic_checkpoint_extra(semantic_config),
                },
            ),
        )
    return TextToTokenTrainingResult(
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
        best_metadata_path=best_meta,
        last_metadata_path=last_meta,
        metrics_path=metrics_path,
        predictor_config_path=predictor_config_path,
        final_train_loss=final_train_loss,
        validation_token_loss=validation_token_loss,
        completed_epoch=config.training.max_epochs - 1,
        global_step=global_step,
    )


def _train_text_to_token_from_surfaces(
    *,
    config: LearnedPoseTokenConfig,
    train_surface: ModelDataSurface,
    validation_surface: ModelDataSurface,
    context: ModelStageExecutionContext,
    tokenizer_checkpoint_path: Path,
    standardization_stats_path: Path,
    progress=None,
) -> TextToTokenTrainingResult:
    _set_seed(config.training.seed if context.request.seed is None else context.request.seed)
    train_reader = ModelDataSurfaceReader(train_surface)
    validation_reader = ModelDataSurfaceReader(validation_surface)
    if train_reader.metadata.unit_count <= 0 or validation_reader.metadata.unit_count <= 0:
        raise LearnedPoseTokenError("text-to-token surfaces must be non-empty.")
    max_positions = max(
        _text_to_token_surface_positions(train_reader),
        _text_to_token_surface_positions(validation_reader),
    )
    text_dim = _text_to_token_surface_text_dim(train_reader)
    predictor = build_text_to_token_predictor(
        text_embedding_dim=text_dim,
        codebook_size=config.codebook.size,
        config=config,
        max_positions=max_positions,
    )
    device = _resolve_device(config.training.device)
    precision_policy = resolve_precision_policy(_compute_precision_policy(context), device)
    predictor.to(device)
    semantic_config = load_semantic_training_objective_for_request(
        request=context.request,
        model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
    )
    if semantic_config is not None:
        raise LearnedPoseTokenError(
            "semantic_consistency for learned_pose_token text-to-token training "
            "requires source pose batches and is not enabled for sharded text surfaces."
        )
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    metrics: list[TrainingMetricRecord] = []
    global_step = 0
    final_train_loss = 0.0
    batch_size = config.training.effective_text_to_token_batch_size
    train_batch_total = surface_batch_count(
        unit_count=train_reader.metadata.unit_count,
        batch_size=batch_size,
    )
    reader_runtime_trace: dict[str, object] = {}
    for epoch in range(config.training.max_epochs):
        predictor.train()
        batch_losses = []
        with progress.task(
            operation=f"text_to_token_train_epoch_{epoch + 1}",
            label=(
                "learned_pose_token text-to-token train epoch "
                f"{epoch + 1}/{config.training.max_epochs}"
            ),
            unit="batch",
            total=train_batch_total,
            total_semantics="text-to-token training batches in epoch",
            allowed_counters=("loss",),
        ) as task:
            for batch in train_reader.iter_batches(
                batch_size=batch_size,
                shuffle_shards=True,
                shuffle_units=True,
                num_workers=config.training.num_workers,
                runtime_trace=reader_runtime_trace,
            ):
                text_embeddings, target_ids, lengths, mask = _predictor_surface_batch(
                    batch,
                    max_positions=max_positions,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision_policy):
                    output = predictor(text_embeddings)
                    losses = compute_text_to_token_losses(
                        output=output,
                        target_token_ids=target_ids,
                        target_lengths=lengths,
                        token_mask=mask,
                        token_loss_weight=config.text_to_token.token_loss_weight,
                        length_loss_weight=config.text_to_token.length_loss_weight,
                    )
                    total_loss = losses["total_loss"]
                total_loss.backward()
                optimizer.step()
                loss_value = float(total_loss.detach().cpu().item())
                batch_losses.append(loss_value)
                metrics.append(_metric("train", "total_loss", loss_value, epoch, global_step))
                task.advance(1, counters={"loss": f"{loss_value:.6g}"})
                global_step += 1
        final_train_loss = float(np.mean(batch_losses))
    validation_token_loss = _validation_token_loss_from_surface(
        predictor,
        validation_reader,
        max_positions=max_positions,
        device=device,
        config=config,
        progress=progress,
        precision_policy=precision_policy,
    )
    metrics.append(
        _metric(
            "val",
            config.checkpoints.selection_metric,
            validation_token_loss,
            config.training.max_epochs - 1,
            global_step,
        )
    )
    checkpoint_root = context.topology.models.model_checkpoints_root(
        LEARNED_POSE_TOKEN_MODEL_KEY,
        context.request.run_name,
    ).path
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_root / "last.pt"
    best_path = checkpoint_root / "best.pt"
    payload = {
        "schema_version": "t2sp-learned-pose-text-to-token-checkpoint-v1",
        "model_key": LEARNED_POSE_TOKEN_MODEL_KEY,
        "run_name": context.request.run_name,
        "predictor_config": {
            "text_embedding_dim": text_dim,
            "codebook_size": config.codebook.size,
            "max_positions": max_positions,
            "hidden_dim": config.text_to_token.hidden_dim,
            "position_dim": config.text_to_token.position_dim,
            "dropout": config.text_to_token.dropout,
            "min_generated_tokens": config.text_to_token.min_generated_tokens,
            "max_generated_tokens": (
                config.text_to_token.max_generated_tokens or max_positions
            ),
        },
        "state_dict": predictor.state_dict(),
        "text_embedding_dim": text_dim,
        "text_encoder_config": config.text_encoder.to_dict(),
        "text_encoder_metadata": {"streaming_surface": True, "output_dim": text_dim},
        "tokenizer_checkpoint_path": str(tokenizer_checkpoint_path),
        "standardization_stats_path": str(standardization_stats_path),
        "tokenizer_config": config.tokenizer.to_dict(),
        "generation": config.generation.to_dict(),
        "validation_token_loss": validation_token_loss,
        "train_text_to_token_surface_path": str(train_surface.root),
        "validation_text_to_token_surface_path": str(validation_surface.root),
    }
    torch.save(payload, last_path)
    torch.save(payload, best_path)
    metadata_root = context.topology.models.model_intermediate_root(
        LEARNED_POSE_TOKEN_MODEL_KEY,
        context.request.run_name,
        "text_to_token",
    ).path
    metadata_root.mkdir(parents=True, exist_ok=True)
    predictor_config_path = metadata_root / "predictor_config.json"
    _write_json(
        predictor_config_path,
        payload["predictor_config"] | {"schema_version": "t2sp-text-to-token-predictor-config-v1"},
    )
    metrics_path = metadata_root / "training_metrics.jsonl"
    write_training_metric_records_jsonl(metrics_path, metrics)
    best_meta = metadata_root / "best_checkpoint_metadata.json"
    last_meta = metadata_root / "last_checkpoint_metadata.json"
    for role, path, metadata_path in (
        (CheckpointRole.BEST, best_path, best_meta),
        (CheckpointRole.LAST, last_path, last_meta),
    ):
        write_checkpoint_metadata_json(
            metadata_path,
            ModelCheckpointMetadata(
                schema_version=CHECKPOINT_METADATA_SCHEMA_VERSION,
                model_key=LEARNED_POSE_TOKEN_MODEL_KEY,
                run_name=context.request.run_name,
                role=role,
                stage_name="train_text_to_token",
                checkpoint_path=path,
                epoch=config.training.max_epochs - 1,
                global_step=global_step,
                selection_metric=config.checkpoints.selection_metric,
                selection_metric_value=validation_token_loss,
                lower_is_better=config.checkpoints.lower_is_better,
                created_at_utc=_now(),
                extra={
                    "length_policy": config.generation.length_policy,
                    "token_selection": config.generation.token_selection,
                    "streaming_text_to_token_surface": True,
                },
            ),
        )
    return TextToTokenTrainingResult(
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
        best_metadata_path=best_meta,
        last_metadata_path=last_meta,
        metrics_path=metrics_path,
        predictor_config_path=predictor_config_path,
        final_train_loss=final_train_loss,
        validation_token_loss=validation_token_loss,
        completed_epoch=config.training.max_epochs - 1,
        global_step=global_step,
        runtime_trace=dict(reader_runtime_trace),
    )


def _decode_validation_text(
    *,
    context: ModelStageExecutionContext,
    config: LearnedPoseTokenConfig,
    progress=None,
) -> tuple[DecodeToPoseResult, Mapping[str, object]]:
    text_stage = _require_stage_result(context, ModelStageKind.TRAIN_TEXT_TO_TOKEN)
    rep_stage = _require_representation_result(context)
    checkpoint_path = Path(str(text_stage.metadata["best_checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise LearnedPoseTokenError(
            f"selected text-to-token checkpoint is missing: {checkpoint_path}"
        )
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    tokenizer = _load_tokenizer(
        Path(str(rep_stage.metadata["tokenizer_checkpoint_path"])),
        config=config,
    )
    stats = read_bfh_standardization_stats_json(
        Path(str(rep_stage.metadata["standardization_stats_path"]))
    )
    text_encoder = build_text_encoder(config.text_encoder)
    predictor = _predictor_from_checkpoint(
        checkpoint,
        config=config,
        text_encoder_output_dim=text_encoder.output_dim,
    )
    prediction_surface, cache_metadata = _materialize_pose_token_surface(
        context=context,
        config=config,
        split=context.request.validation_split,
        max_samples=config.data.limit_prediction_samples,
        progress=progress,
        operation="materialize_prediction_surface",
        label="learned_pose_token decode build validation source surface",
        total_semantics="validation prediction source surface samples loaded",
    )
    source_count = prediction_surface.metadata.loaded_sample_count
    if source_count <= 0:
        raise LearnedPoseTokenError("validation prediction source selection is empty.")
    token_root = _decode_root(context) / "predicted_token_samples" / "val"
    manifest_path = _decode_root(context) / "predicted_token_manifest_val.jsonl"
    token_counts: dict[str, int] = {}
    frame_counts: dict[str, int] = {}
    predicted_token_sample_count = 0

    def decoded_iter():
        nonlocal predicted_token_sample_count
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        token_root.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as manifest_handle:
            with progress.task(
                operation="predict_and_decode_token_samples",
                label="learned_pose_token predict and decode validation samples",
                unit="sample",
                total=source_count,
                total_semantics="validation token samples predicted and decoded",
                allowed_counters=("token_units",),
            ) as task:
                for source_batch in _surface_source_batches(
                    ModelDataSurfaceReader(prediction_surface),
                    config.training.effective_decode_batch_size,
                ):
                    encoded_text = text_encoder.encode_batch(
                        [source.text for source in source_batch],
                        sentence_names=[source.source_sentence_name for source in source_batch],
                    )
                    embeddings = np.stack(
                        [output.pooled_embedding for output in encoded_text]
                    ).astype(np.float32)
                    token_ids, lengths = _predict_tokens(predictor, embeddings)
                    for row, source in enumerate(source_batch):
                        count = int(lengths[row])
                        sequence = PoseTokenSequence(
                            schema_version=POSE_TOKEN_SCHEMA_VERSION,
                            sample_id=source.sample_id,
                            source_sentence_name=source.source_sentence_name,
                            split=source.split,
                            token_ids=token_ids[row, :count],
                            frame_count=generated_frame_count_from_token_count(
                                count,
                                config.tokenizer.temporal_granularity,
                                config.tokenizer.window_size,
                                config.tokenizer.stride,
                            ),
                            token_count=count,
                            codebook_size=config.codebook.size,
                            temporal_granularity=config.tokenizer.temporal_granularity,
                            window_size=config.tokenizer.window_size,
                            stride=config.tokenizer.stride,
                        )
                        path = token_root / f"{source.sample_id}.npz"
                        write_pose_token_sequence_npz(path, sequence)
                        predicted_token_sample_count += 1
                        entry = PoseTokenManifestEntry(
                            schema_version=POSE_TOKEN_MANIFEST_SCHEMA_VERSION,
                            sample_id=source.sample_id,
                            source_sentence_name=source.source_sentence_name,
                            split=source.split,
                            token_path=path,
                            token_count=sequence.token_count,
                            frame_count=sequence.frame_count,
                            codebook_size=sequence.codebook_size,
                            temporal_granularity=sequence.temporal_granularity,
                            window_size=sequence.window_size,
                            stride=sequence.stride,
                            issues=(),
                        )
                        manifest_handle.write(
                            json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                        pose = decode_token_sequence_to_bfh_pose(
                            tokenizer=tokenizer,
                            token_sequence=sequence,
                            standardization_stats=stats,
                            source=source,
                            confidence_policy=config.generation.confidence_policy,
                        )
                        token_counts[source.sample_id] = sequence.token_count
                        frame_counts[source.sample_id] = sequence.frame_count
                        task.advance(1, counters={"token_units": sequence.token_count})
                        yield DecodedPoseSample(source=source, token_sequence=sequence, pose=pose)

    with progress.task(
        operation="write_decoded_intermediate_poses",
        label="learned_pose_token write decoded intermediate poses",
        unit="sample",
        total=source_count,
        total_semantics="decoded intermediate generated pose samples written",
    ) as task:
        decoded_written = export_predicted_pose_samples_incremental(
            decoded_samples=decoded_iter(),
            diagnostic_root=_decode_root(context) / "decoded_pose_intermediates" / "val",
            producer_key=LEARNED_POSE_TOKEN_MODEL_KEY,
            producer_stage="decode_to_pose",
            run_name=context.request.run_name,
            split=context.request.validation_split,
            generation_mode=config.generation.generation_mode,
            seed=context.request.seed,
        )
        task.advance(decoded_written.sample_count)
    return (
        DecodeToPoseResult(
            predicted_token_manifest_path=manifest_path,
            predicted_token_sample_count=predicted_token_sample_count,
            decoded_generated_pose_manifest_path=decoded_written.manifest_path,
            decoded_sample_count=decoded_written.sample_count,
            predicted_token_counts=token_counts,
            decoded_frame_counts=frame_counts,
        ),
        cache_metadata,
    )


def _validation_token_loss(
    predictor: TextToTokenPredictor,
    sequences: Sequence[PoseTokenSequence],
    sources: Mapping[str, PoseTokenSourceSample],
    *,
    text_encoder,
    max_positions: int,
    device: torch.device,
    config: LearnedPoseTokenConfig,
    progress=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
) -> float:
    losses = []
    predictor.eval()
    resolved_precision = precision_policy or resolve_precision_policy("auto", device)
    with torch.no_grad():
        batches = _sequence_batches(
            sequences,
            config.training.effective_text_to_token_batch_size,
        )
        with progress.task(
            operation="text_to_token_validation",
            label="learned_pose_token text-to-token validation",
            unit="batch",
            total=len(batches),
            total_semantics="text-to-token validation batches",
            allowed_counters=("loss",),
        ) as task:
            for batch in batches:
                text_embeddings, target_ids, lengths, mask = _predictor_batch(
                    batch,
                    sources,
                    text_encoder=text_encoder,
                    max_positions=max_positions,
                    device=device,
                )
                with autocast_context(resolved_precision):
                    output = predictor(text_embeddings)
                    computed = compute_text_to_token_losses(
                        output=output,
                        target_token_ids=target_ids,
                        target_lengths=lengths,
                        token_mask=mask,
                        token_loss_weight=config.text_to_token.token_loss_weight,
                        length_loss_weight=config.text_to_token.length_loss_weight,
                    )
                loss = float(computed["token_loss"].detach().cpu().item())
                losses.append(loss)
                task.advance(1, counters={"loss": f"{loss:.6g}"})
    return float(np.mean(losses))


def _soft_semantic_loss_for_token_batch(
    *,
    tokenizer: LearnedPoseTokenizer,
    output,
    batch: Sequence[PoseTokenSequence],
    sources: Mapping[str, PoseTokenSourceSample],
    text_embeddings: torch.Tensor,
    token_mask: torch.Tensor,
    config: LearnedPoseTokenConfig,
    semantic_config: SemanticConsistencyObjectiveConfig,
):
    probabilities = torch.softmax(output.token_logits, dim=-1)
    codebook = tokenizer.quantizer.embedding.weight.detach()
    expected_latents = probabilities @ codebook
    decoded_units = tokenizer.decoder(expected_latents.reshape(-1, codebook.shape[-1])).reshape(
        expected_latents.shape[0],
        expected_latents.shape[1],
        -1,
    )
    feature_dim = default_bfh_tensor_layout().total_feature_dim
    max_frames = max(sequence.frame_count for sequence in batch)
    pose_rows: list[torch.Tensor] = []
    validity_rows: list[torch.Tensor] = []
    frame_rows: list[torch.Tensor] = []
    for row, sequence in enumerate(batch):
        source = sources.get(sequence.sample_id)
        if source is None:
            raise LearnedPoseTokenError(f"missing source identity for token sample {sequence.sample_id!r}.")
        token_count = int(sequence.token_count)
        if config.tokenizer.temporal_granularity == "frame":
            pose = decoded_units[row, : sequence.frame_count, :]
        else:
            if sequence.temporal_granularity != config.tokenizer.temporal_granularity:
                raise LearnedPoseTokenError(
                    "semantic token temporal_granularity does not match config."
                )
            if (
                sequence.window_size != config.tokenizer.window_size
                or sequence.stride != config.tokenizer.stride
            ):
                raise LearnedPoseTokenError(
                    "semantic token window_size/stride does not match config."
                )
            starts = temporal_window_starts(
                frame_count=sequence.frame_count,
                spec=pose_token_temporal_spec(
                    sequence.temporal_granularity,
                    sequence.window_size,
                    sequence.stride,
                ),
            )
            if token_count != len(starts):
                raise LearnedPoseTokenError(
                    "semantic token_count must equal "
                    "len(temporal_window_starts(frame_count, spec))."
                )
            window_values = decoded_units[row, :token_count, :].reshape(
                token_count,
                config.tokenizer.window_size,
                feature_dim,
            )
            pose = merge_overlapping_windows_torch(
                window_values,
                frame_count=sequence.frame_count,
                window_size=config.tokenizer.window_size,
                stride=config.tokenizer.stride,
                start_indices=starts,
            )
        if pose.shape != (sequence.frame_count, feature_dim):
            raise LearnedPoseTokenError("semantic soft decode did not reconstruct frame-level pose values.")
        pad = max_frames - sequence.frame_count
        if pad > 0:
            pose = torch.cat((pose, pose.new_zeros((pad, feature_dim))), dim=0)
        pose_rows.append(pose)
        source_validity = np.repeat(
            source.vectorized_pose.validity_mask & source.vectorized_pose.frame_validity_mask[:, None],
            source.vectorized_pose.layout.coordinate_dimensions,
            axis=1,
        )
        validity = torch.as_tensor(source_validity, dtype=torch.bool, device=pose.device)
        frame = torch.as_tensor(
            source.vectorized_pose.frame_validity_mask,
            dtype=torch.bool,
            device=pose.device,
        )
        if pad > 0:
            validity = torch.cat((validity, torch.zeros((pad, feature_dim), dtype=torch.bool, device=pose.device)), dim=0)
            frame = torch.cat((frame, torch.zeros((pad,), dtype=torch.bool, device=pose.device)), dim=0)
        validity_rows.append(validity)
        frame_rows.append(frame)
    if not torch.any(token_mask):
        raise LearnedPoseTokenError("semantic text-to-token batch has no valid token observations.")
    return compute_semantic_training_loss(
        text_embeddings=text_embeddings,
        pose_values=torch.stack(pose_rows, dim=0),
        pose_validity_mask=torch.stack(validity_rows, dim=0),
        frame_mask=torch.stack(frame_rows, dim=0),
        layout=default_bfh_tensor_layout(),
        config=semantic_config,
    )


def _predictor_batch(
    batch: Sequence[PoseTokenSequence],
    sources: Mapping[str, PoseTokenSourceSample],
    *,
    text_encoder,
    max_positions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = []
    target_ids = np.zeros((len(batch), max_positions), dtype=np.int64)
    mask = np.zeros((len(batch), max_positions), dtype=np.bool_)
    lengths = np.zeros((len(batch),), dtype=np.float32)
    for row, sequence in enumerate(batch):
        source = sources.get(sequence.sample_id)
        if source is None:
            raise LearnedPoseTokenError(f"missing source identity for token sample {sequence.sample_id!r}.")
        embeddings.append(
            text_encoder.encode_one(
                source.text,
                sentence_name=source.source_sentence_name,
            ).pooled_embedding
        )
        count = sequence.token_count
        target_ids[row, :count] = np.asarray(sequence.token_ids, dtype=np.int64)
        mask[row, :count] = True
        lengths[row] = float(count)
    return (
        torch.as_tensor(np.stack(embeddings).astype(np.float32), device=device),
        torch.as_tensor(target_ids, dtype=torch.long, device=device),
        torch.as_tensor(lengths, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


def _predictor_surface_batch(
    batch: Mapping[str, object],
    *,
    max_positions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = _required_surface_tensor(batch, "text_embedding").float()
    target_ids = _required_surface_tensor(batch, "target_token_ids").long()
    mask = _required_surface_tensor(batch, "target_mask").bool()
    lengths = _required_surface_tensor(batch, "sequence_length").float()
    if target_ids.shape[1] < max_positions:
        pad = max_positions - int(target_ids.shape[1])
        target_ids = torch.nn.functional.pad(target_ids, (0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
    elif target_ids.shape[1] > max_positions:
        target_ids = target_ids[:, :max_positions]
        mask = mask[:, :max_positions]
    return (
        embeddings.to(device),
        target_ids.to(device),
        lengths.to(device),
        mask.to(device),
    )


def _validation_token_loss_from_surface(
    predictor: TextToTokenPredictor,
    reader: ModelDataSurfaceReader,
    *,
    max_positions: int,
    device: torch.device,
    config: LearnedPoseTokenConfig,
    progress=None,
    precision_policy: ResolvedPrecisionPolicy | None = None,
) -> float:
    losses = []
    predictor.eval()
    resolved_precision = precision_policy or resolve_precision_policy("auto", device)
    batch_size = config.training.effective_text_to_token_batch_size
    with torch.no_grad():
        with progress.task(
            operation="text_to_token_validation",
            label="learned_pose_token text-to-token validation",
            unit="batch",
            total=surface_batch_count(unit_count=reader.metadata.unit_count, batch_size=batch_size),
            total_semantics="text-to-token validation batches",
            allowed_counters=("loss",),
        ) as task:
            for batch in reader.iter_batches(
                batch_size=batch_size,
                num_workers=config.training.num_workers,
            ):
                text_embeddings, target_ids, lengths, mask = _predictor_surface_batch(
                    batch,
                    max_positions=max_positions,
                    device=device,
                )
                with autocast_context(resolved_precision):
                    output = predictor(text_embeddings)
                    computed = compute_text_to_token_losses(
                        output=output,
                        target_token_ids=target_ids,
                        target_lengths=lengths,
                        token_mask=mask,
                        token_loss_weight=config.text_to_token.token_loss_weight,
                        length_loss_weight=config.text_to_token.length_loss_weight,
                    )
                loss = float(computed["token_loss"].detach().cpu().item())
                losses.append(loss)
                task.advance(1, counters={"loss": f"{loss:.6g}"})
    if not losses:
        raise LearnedPoseTokenError("validation text-to-token surface contains no batches.")
    return float(np.mean(losses))


def _text_to_token_surface_positions(reader: ModelDataSurfaceReader) -> int:
    for shard in reader.iter_shards():
        target = shard.get("target_token_ids")
        if isinstance(target, torch.Tensor) and target.ndim == 2:
            return int(target.shape[1])
    raise LearnedPoseTokenError("text-to-token surface is missing target_token_ids.")


def _text_to_token_surface_text_dim(reader: ModelDataSurfaceReader) -> int:
    for shard in reader.iter_shards():
        embeddings = shard.get("text_embedding")
        if isinstance(embeddings, torch.Tensor) and embeddings.ndim == 2:
            return int(embeddings.shape[1])
    raise LearnedPoseTokenError("text-to-token surface is missing text_embedding.")


def _required_surface_tensor(batch: Mapping[str, object], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise LearnedPoseTokenError(f"surface batch is missing tensor field {name!r}.")
    return value


def _read_sequences(path: Path) -> tuple[PoseTokenSequence, ...]:
    entries = read_pose_token_manifest_jsonl(path)
    return tuple(read_pose_token_sequence_npz(entry.token_path) for entry in entries)


def _decode_entry(
    entry: PoseTokenManifestEntry,
    *,
    sources: Mapping[str, PoseTokenSourceSample],
    tokenizer: LearnedPoseTokenizer,
    stats,
    config: LearnedPoseTokenConfig,
) -> DecodedPoseSample:
    source = sources.get(entry.sample_id)
    if source is None:
        raise LearnedPoseTokenError(f"missing source identity for {entry.sample_id!r}.")
    sequence = read_pose_token_sequence_npz(entry.token_path)
    pose = decode_token_sequence_to_bfh_pose(
        tokenizer=tokenizer,
        token_sequence=sequence,
        standardization_stats=stats,
        source=source,
        confidence_policy=config.generation.confidence_policy,
    )
    return DecodedPoseSample(source=source, token_sequence=sequence, pose=pose)


def _sources_by_id(samples: Sequence[PoseTokenTrainingSample]) -> Mapping[str, PoseTokenSourceSample]:
    sources: OrderedDict[str, PoseTokenSourceSample] = OrderedDict()
    for sample in samples:
        sources.setdefault(sample.source.sample_id, sample.source)
    return sources


def _surface_sources_by_id(reader: ModelDataSurfaceReader) -> Mapping[str, PoseTokenSourceSample]:
    sources: OrderedDict[str, PoseTokenSourceSample] = OrderedDict()
    for record in _surface_source_records(reader):
        source = _pose_token_source_from_surface_record(record)
        sources[source.sample_id] = source
    return sources


def _surface_source_records(reader: ModelDataSurfaceReader) -> tuple[Mapping[str, object], ...]:
    path = reader.surface.root / "sources.jsonl"
    if not path.is_file():
        raise LearnedPoseTokenError("pose-token surface is missing sources.jsonl.")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise LearnedPoseTokenError("surface sources.jsonl rows must be objects.")
            records.append(record)
    return tuple(records)


def _surface_source_batches(
    reader: ModelDataSurfaceReader,
    batch_size: int,
):
    if batch_size <= 0:
        raise LearnedPoseTokenError("surface source batch_size must be positive.")
    batch: list[PoseTokenSourceSample] = []
    path = reader.surface.root / "sources.jsonl"
    if not path.is_file():
        raise LearnedPoseTokenError("pose-token surface is missing sources.jsonl.")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise LearnedPoseTokenError("surface sources.jsonl rows must be objects.")
            batch.append(_pose_token_source_from_surface_record(record))
            if len(batch) >= batch_size:
                yield tuple(batch)
                batch.clear()
    if batch:
        yield tuple(batch)


def _pose_token_source_from_surface_record(record: Mapping[str, object]) -> PoseTokenSourceSample:
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
    return PoseTokenSourceSample(
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


def _source_from_manifest_sample(sample) -> PoseTokenSourceSample:
    from text_to_sign_production.modeling.backbones.bfh_vectorization import vectorize_bfh_pose_arrays

    vectorized = vectorize_bfh_pose_arrays(sample.pose, sample_id=sample.sample_id)
    return PoseTokenSourceSample(
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


def _load_tokenizer(path: Path, *, config: LearnedPoseTokenConfig) -> LearnedPoseTokenizer:
    checkpoint = _load_torch_checkpoint(path)
    checkpoint_tokenizer = checkpoint.get("config")
    if isinstance(checkpoint_tokenizer, Mapping):
        checkpoint_tokenizer = checkpoint_tokenizer.get("tokenizer")
    if isinstance(checkpoint_tokenizer, Mapping):
        _validate_checkpoint_temporal_config(
            checkpoint_tokenizer,
            config=config,
            owner="checkpoint tokenizer",
        )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise LearnedPoseTokenError(f"tokenizer checkpoint is missing state_dict: {path}")
    weight = state.get("encoder.net.0.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise LearnedPoseTokenError("tokenizer checkpoint does not expose encoder input dimension.")
    from text_to_sign_production.modeling.backbones.bfh_vectorization import (
        default_bfh_tensor_layout,
    )

    expected_input_dim = (
        default_bfh_tensor_layout().total_feature_dim
        * config.tokenizer.temporal_window_spec().window_size
    )
    if int(weight.shape[1]) != expected_input_dim:
        raise LearnedPoseTokenError(
            f"checkpoint tokenizer input_dim={int(weight.shape[1])} but current config "
            f"requires {expected_input_dim}. Use the original config or retrain."
        )
    tokenizer = build_learned_pose_tokenizer(input_dim=int(weight.shape[1]), config=config)
    tokenizer.load_state_dict(state)
    tokenizer.eval()
    return tokenizer


def _predictor_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    config: LearnedPoseTokenConfig,
    text_encoder_output_dim: int,
) -> TextToTokenPredictor:
    raw = checkpoint.get("predictor_config")
    if not isinstance(raw, Mapping):
        raise LearnedPoseTokenError("text-to-token checkpoint is missing predictor_config.")
    checkpoint_text_dim = int(raw["text_embedding_dim"])
    if checkpoint_text_dim != text_encoder_output_dim:
        raise LearnedPoseTokenError(
            f"checkpoint text_embedding_dim={checkpoint_text_dim} but current text encoder "
            f"output_dim={text_encoder_output_dim}. Use the original text encoder config or retrain."
        )
    tokenizer_config = checkpoint.get("tokenizer_config")
    if isinstance(tokenizer_config, Mapping):
        _validate_checkpoint_temporal_config(
            tokenizer_config,
            config=config,
            owner="checkpoint tokenizer",
        )
    predictor = build_text_to_token_predictor(
        text_embedding_dim=checkpoint_text_dim,
        codebook_size=int(raw["codebook_size"]),
        config=config,
        max_positions=int(raw["max_positions"]),
    )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise LearnedPoseTokenError("text-to-token checkpoint is missing state_dict.")
    predictor.load_state_dict(state)
    predictor.eval()
    return predictor


def _validate_checkpoint_temporal_config(
    raw: Mapping[str, object],
    *,
    config: LearnedPoseTokenConfig,
    owner: str,
) -> None:
    checkpoint_granularity = str(raw.get("temporal_granularity", ""))
    if checkpoint_granularity != config.tokenizer.temporal_granularity:
        raise LearnedPoseTokenError(
            f"{owner} temporal_granularity={checkpoint_granularity!r} but current "
            f"config requires {config.tokenizer.temporal_granularity!r}. Use the original config or retrain."
        )
    for field_name in ("window_size", "stride"):
        checkpoint_value = int(raw.get(field_name, -1))
        current_value = int(getattr(config.tokenizer, field_name))
        if checkpoint_value != current_value:
            raise LearnedPoseTokenError(
                f"{owner} {field_name}={checkpoint_value} but current config requires "
                f"{current_value}. Use the original config or retrain."
            )


def _predict_tokens(predictor: TextToTokenPredictor, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = next(predictor.parameters()).device
    predictor.eval()
    with torch.no_grad():
        ids, lengths = predictor.generate_token_ids(
            torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        )
    return ids.detach().cpu().numpy().astype(np.int64), lengths.detach().cpu().numpy().astype(np.int64)


def _path_from_checkpoint(
    checkpoint: Mapping[str, object],
    key: str,
    *,
    fallback: Path | None = None,
) -> Path:
    value = checkpoint.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LearnedPoseTokenError(f"text-to-token checkpoint is missing {key}.")
    path = Path(value)
    if not path.is_file() and fallback is not None and fallback.is_file():
        return fallback
    if not path.is_file():
        raise LearnedPoseTokenError(
            f"text-to-token checkpoint references missing {key}: {path}. "
            "Restore the learned_pose_token provider artifacts for the selected model run."
        )
    return path


def _canonical_restored_support_path(
    checkpoint: Mapping[str, object],
    key: str,
    *,
    canonical_path: Path,
    runtime_root: Path,
) -> Path:
    path = Path(canonical_path)
    if not path.is_file():
        raise LearnedPoseTokenError(
            f"learned_pose_token restore is missing required support artifact {key}: {path}. "
            "Restore the selected model run before running test_model inference."
        )
    advisory = checkpoint.get(key)
    if isinstance(advisory, str) and advisory.strip():
        advisory_path = Path(advisory).expanduser().resolve(strict=False)
        resolved_runtime = Path(runtime_root).expanduser().resolve(strict=False)
        try:
            advisory_path.relative_to(resolved_runtime)
        except ValueError:
            return path
    return path


def _load_torch_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise LearnedPoseTokenError(f"could not load checkpoint {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise LearnedPoseTokenError(f"checkpoint payload must be a mapping: {path}")
    return loaded


def _require_representation_result(context: ModelStageExecutionContext) -> ModelStageResult:
    return _require_stage_result(context, ModelStageKind.FIT_REPRESENTATION)


def _require_stage_result(
    context: ModelStageExecutionContext,
    kind: ModelStageKind,
) -> ModelStageResult:
    for result in context.stage_results:
        if result.stage.spec.kind is kind and result.status is ModelStageStatus.COMPLETED:
            return result
    raise LearnedPoseTokenError(f"required completed stage result is missing: {kind.value}.")


def _sequence_batches(
    sequences: Sequence[PoseTokenSequence],
    batch_size: int,
) -> tuple[tuple[PoseTokenSequence, ...], ...]:
    return _chunked(sequences, batch_size)


def _chunked(
    values: Sequence,
    batch_size: int,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(values[index : index + batch_size])
        for index in range(0, len(values), batch_size)
    )


def _array_batches(values: np.ndarray, batch_size: int) -> tuple[np.ndarray, ...]:
    return tuple(values[index : index + batch_size] for index in range(0, len(values), batch_size))


def _representation_root(context: ModelStageExecutionContext) -> Path:
    return context.topology.models.model_intermediate_root(
        LEARNED_POSE_TOKEN_MODEL_KEY,
        context.request.run_name,
        "representation",
    ).path


def _reconstruction_root(context: ModelStageExecutionContext) -> Path:
    return context.topology.models.model_intermediate_root(
        LEARNED_POSE_TOKEN_MODEL_KEY,
        context.request.run_name,
        "reconstruction",
    ).path


def _decode_root(context: ModelStageExecutionContext) -> Path:
    return context.topology.models.model_intermediate_root(
        LEARNED_POSE_TOKEN_MODEL_KEY,
        context.request.run_name,
        "decode_to_pose",
    ).path


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
        value=value,
        epoch=epoch,
        global_step=global_step,
        stage_name="train_text_to_token",
        created_at_utc=_now(),
        extra={"length_policy": "predicted_length"},
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearnedPoseTokenError(f"could not read JSON payload: {path}") from exc


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _now() -> str:
    return datetime.now(UTC).isoformat()


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


def _load_generated_sample_for_result(path: Path):
    from text_to_sign_production.modeling.artifacts import load_generated_pose_payload

    return load_generated_pose_payload(path)


__all__ = ["LearnedPoseTokenProvider"]
