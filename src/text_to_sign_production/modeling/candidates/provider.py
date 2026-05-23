"""Provider interface, stage plans, and generic sequential execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.modeling.candidates.calibration_surfaces import (
    RepresentativeCalibrationSurfaceResult,
)
from text_to_sign_production.modeling.candidates.configs import (
    ModelRunRequest,
    validate_objective_attachments,
)
from text_to_sign_production.modeling.candidates.errors import (
    ModelProviderRegistrationError,
    ModelStageExecutionError,
    ModelStagePlanError,
)
from text_to_sign_production.modeling.candidates.objectives import (
    ObjectiveAttachmentPlan,
    build_objective_attachment_plans,
)
from text_to_sign_production.modeling.candidates.results import (
    ModelSingleSampleInferenceResult,
    ModelExecutionResult,
    ModelStageArtifactRef,
    ModelStageResult,
    ModelStageStatus,
)
from text_to_sign_production.modeling.candidates.runtime_support import (
    ModelRuntimeSupportArtifact,
)
from text_to_sign_production.modeling.candidates.stages import (
    MODEL_STAGE_SPECS,
    PlannedModelStage,
    stage_sequence_from_values,
    validate_stage_sequence_for_spec,
)
from text_to_sign_production.modeling.registry import ModelingRegistryError, require_model_spec
from text_to_sign_production.modeling.data.prepared_sample_loader import ModelingManifestSample
from text_to_sign_production.modeling.research import ModelKey, ModelSpec


@dataclass(frozen=True, slots=True)
class ProviderCalibrationPolicy:
    """Provider-owned contract for runtime compute calibration."""

    provider_key: str
    supports_provider_real: bool
    required_run_modes: tuple[str, ...]
    required_compute_profiles: tuple[str, ...]
    candidate_keys: tuple[str, ...]
    override_targets: Mapping[str, str]
    representative_surface_kinds: Mapping[str, str]
    representative_split: Literal["train", "validation"]
    max_samples: int
    warmup_batches: int
    max_batches_per_candidate: int
    reuse_if_artifacts_match: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.provider_key, str) or not self.provider_key.strip():
            raise ModelStagePlanError("calibration policy provider_key must be non-empty.")
        if not isinstance(self.supports_provider_real, bool):
            raise ModelStagePlanError("calibration policy supports_provider_real must be boolean.")
        for field_name in ("required_run_modes", "required_compute_profiles", "candidate_keys"):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ModelStagePlanError(f"calibration policy {field_name} must contain text.")
            object.__setattr__(self, field_name, values)
        targets = _immutable_string_keyed_mapping(self.override_targets, "override_targets")
        missing = tuple(key for key in self.candidate_keys if key not in targets)
        if missing:
            raise ModelStagePlanError(
                f"calibration policy candidate keys missing override targets: {missing!r}."
            )
        surface_kinds = _immutable_string_keyed_mapping(
            self.representative_surface_kinds,
            "representative_surface_kinds",
        )
        missing_surface = tuple(key for key in self.candidate_keys if key not in surface_kinds)
        if missing_surface:
            raise ModelStagePlanError(
                "calibration policy candidate keys missing representative surface kinds: "
                f"{missing_surface!r}."
            )
        for key, value in targets.items():
            if not isinstance(value, str) or not value.strip():
                raise ModelStagePlanError(
                    f"calibration policy override target for {key!r} must be non-empty text."
                )
        for key, value in surface_kinds.items():
            if not isinstance(value, str) or not value.strip():
                raise ModelStagePlanError(
                    "calibration policy representative surface kind for "
                    f"{key!r} must be non-empty text."
                )
        if self.representative_split not in {"train", "validation"}:
            raise ModelStagePlanError(
                "calibration policy representative_split must be train or validation."
            )
        if (
            not isinstance(self.max_samples, int)
            or isinstance(self.max_samples, bool)
            or self.max_samples <= 0
        ):
            raise ModelStagePlanError("calibration policy max_samples must be positive.")
        if (
            not isinstance(self.warmup_batches, int)
            or isinstance(self.warmup_batches, bool)
            or self.warmup_batches < 0
        ):
            raise ModelStagePlanError("calibration policy warmup_batches must be non-negative.")
        if (
            not isinstance(self.max_batches_per_candidate, int)
            or isinstance(self.max_batches_per_candidate, bool)
            or self.max_batches_per_candidate <= 0
        ):
            raise ModelStagePlanError(
                "calibration policy max_batches_per_candidate must be positive."
            )
        if not isinstance(self.reuse_if_artifacts_match, bool):
            raise ModelStagePlanError("calibration policy reuse_if_artifacts_match must be boolean.")
        object.__setattr__(self, "override_targets", targets)
        object.__setattr__(self, "representative_surface_kinds", surface_kinds)


@dataclass(frozen=True, slots=True)
class ModelProviderLoadedConfig:
    """Provider-resolved configuration for one run request."""

    model_key: ModelKey
    source_path: Path | None
    raw_config: Mapping[str, object]
    effective_config: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "model_key", ModelKey(self.model_key))
        except (TypeError, ValueError) as exc:
            raise ModelStagePlanError(f"unknown model key: {self.model_key!r}") from exc
        if self.source_path is not None and not isinstance(self.source_path, Path):
            raise ModelStagePlanError("source_path must be a Path or None.")
        object.__setattr__(
            self,
            "raw_config",
            _immutable_string_keyed_mapping(self.raw_config, "raw_config"),
        )
        object.__setattr__(
            self,
            "effective_config",
            _immutable_string_keyed_mapping(self.effective_config, "effective_config"),
        )


@dataclass(frozen=True, slots=True)
class ProviderFullDataPipelineCapability:
    """Provider declaration for full-run training data safety."""

    provider_key: str
    full_training_data_mode: Literal[
        "streaming_sharded",
        "lazy_dataloader",
        "unsafe_eager",
        "partial_streaming",
        "unknown_pending_audit",
    ]
    verified: bool
    covered_stages: tuple[str, ...]
    verification_evidence: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provider_key, str) or not self.provider_key.strip():
            raise ModelStagePlanError("provider_key must be non-empty text.")
        if self.full_training_data_mode not in {
            "streaming_sharded",
            "lazy_dataloader",
            "unsafe_eager",
            "partial_streaming",
            "unknown_pending_audit",
        }:
            raise ModelStagePlanError("full_training_data_mode is unsupported.")
        if not isinstance(self.verified, bool):
            raise ModelStagePlanError("verified must be a boolean.")
        if not isinstance(self.verification_evidence, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.verification_evidence
        ):
            raise ModelStagePlanError("verification_evidence must be strings.")
        if self.verified and not self.verification_evidence:
            raise ModelStagePlanError("verified capabilities require verification evidence.")
        if not isinstance(self.covered_stages, tuple) or any(
            not isinstance(stage, str) or not stage.strip()
            for stage in self.covered_stages
        ):
            raise ModelStagePlanError("covered_stages must be non-empty strings.")
        if not isinstance(self.limitations, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.limitations
        ):
            raise ModelStagePlanError("limitations must be strings.")
        if self.verified and self.limitations:
            raise ModelStagePlanError(
                "verified full-data capabilities cannot declare limitations."
            )

    @property
    def is_full_safe(self) -> bool:
        return (
            self.verified
            and self.full_training_data_mode in {
                "streaming_sharded",
                "lazy_dataloader",
            }
            and not self.limitations
        )

    @property
    def verification(self) -> tuple[str, ...]:
        """Backward-compatible alias for older tests and reports."""

        return self.verification_evidence


@dataclass(frozen=True, slots=True)
class ProviderFullDataPipelineDeclaration:
    provider_key: str
    mode: Literal["streaming_sharded", "lazy_dataloader", "unsupported"]
    covered_stages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeVerifiedFullDataPipeline:
    provider_key: str
    verified: bool
    evidence_refs: tuple["VerificationEvidenceRef", ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerificationEvidenceRef:
    kind: str
    path: Path
    sha256: str
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ModelStagePlanError("verification evidence kind must be non-empty text.")
        path = Path(self.path)
        if not path.is_file():
            raise ModelStagePlanError("verification evidence artifact path must exist.")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ModelStagePlanError("verification evidence sha256 must be a hex digest.")
        if _sha256_file(path) != self.sha256:
            raise ModelStagePlanError("verification evidence artifact sha256 mismatch.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "summary", _immutable_string_keyed_mapping(self.summary, "summary"))


@dataclass(frozen=True, slots=True)
class VerifiedProviderFullDataPipelineCapability:
    """Runtime full-data verification assembled from artifacts."""

    provider_key: str
    model_key: str
    run_name: str | None
    mode: Literal["streaming_sharded", "lazy_dataloader"]
    verified: bool
    verification_artifact_paths: tuple[Path, ...]
    calibration_artifact_path: Path | None
    selected_overrides_path: Path | None
    behavior_evidence: tuple[str, ...]
    limitations: tuple[str, ...]
    verification_evidence_refs: tuple[VerificationEvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provider_key, str) or not self.provider_key.strip():
            raise ModelStagePlanError("provider_key must be non-empty text.")
        if not isinstance(self.model_key, str) or not self.model_key.strip():
            raise ModelStagePlanError("model_key must be non-empty text.")
        if self.mode not in {"streaming_sharded", "lazy_dataloader"}:
            raise ModelStagePlanError("runtime verified mode must be streaming_sharded or lazy_dataloader.")
        if not isinstance(self.verified, bool):
            raise ModelStagePlanError("runtime verified must be a boolean.")
        artifacts = tuple(Path(path) for path in self.verification_artifact_paths)
        if self.verified and not artifacts:
            raise ModelStagePlanError("runtime verified capabilities require artifact paths.")
        refs = tuple(self.verification_evidence_refs)
        if self.verified and not refs:
            raise ModelStagePlanError("runtime verified capabilities require artifact refs.")
        if any(not isinstance(ref, VerificationEvidenceRef) for ref in refs):
            raise ModelStagePlanError("runtime verification_evidence_refs must be artifact refs.")
        evidence = tuple(self.behavior_evidence)
        if any(not isinstance(item, str) or not item.strip() for item in evidence):
            raise ModelStagePlanError("runtime behavior_evidence must be non-empty strings.")
        limitations = tuple(self.limitations)
        if any(not isinstance(item, str) or not item.strip() for item in limitations):
            raise ModelStagePlanError("runtime limitations must be non-empty strings.")
        if self.verified and limitations:
            raise ModelStagePlanError("runtime verified capabilities cannot declare limitations.")
        object.__setattr__(self, "verification_artifact_paths", artifacts)
        object.__setattr__(
            self,
            "calibration_artifact_path",
            None if self.calibration_artifact_path is None else Path(self.calibration_artifact_path),
        )
        object.__setattr__(
            self,
            "selected_overrides_path",
            None if self.selected_overrides_path is None else Path(self.selected_overrides_path),
        )
        if self.verified and self.selected_overrides_path is None:
            raise ModelStagePlanError(
                "runtime verified capabilities require selected_overrides artifact."
            )
        if self.verified:
            kinds = {ref.kind for ref in refs}
            missing_kinds = {
                "provider_real_calibration",
                "selected_overrides",
                "calibrated_effective_config",
                "representative_surface",
                "progress_totals",
                "bounded_writers",
            } - kinds
            if missing_kinds:
                raise ModelStagePlanError(
                    "runtime verified capabilities require artifact-backed calibration evidence refs."
                )
            _validate_runtime_capability_artifacts(
                provider_key=self.provider_key,
                model_key=self.model_key,
                run_name=self.run_name,
                calibration_artifact_path=self.calibration_artifact_path,
                selected_overrides_path=self.selected_overrides_path,
            )
        object.__setattr__(self, "behavior_evidence", evidence)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "verification_evidence_refs", refs)

    @property
    def is_full_safe(self) -> bool:
        return (
            self.verified
            and not self.limitations
            and self.calibration_artifact_path is not None
            and self.selected_overrides_path is not None
            and bool(self.verification_evidence_refs)
        )


@dataclass(frozen=True, slots=True)
class ModelStageExecutionContext:
    """Inputs visible to one provider stage execution."""

    request: ModelRunRequest
    loaded_config: ModelProviderLoadedConfig
    topology: ArtifactTopology
    working_dir: Path
    stage_results: tuple[ModelStageResult, ...] = ()
    progress_session: ProgressSession | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, ModelRunRequest):
            raise ModelStageExecutionError("request must be a ModelRunRequest.")
        if not isinstance(self.loaded_config, ModelProviderLoadedConfig):
            raise ModelStageExecutionError(
                "loaded_config must be a ModelProviderLoadedConfig."
            )
        if not isinstance(self.topology, ArtifactTopology):
            raise ModelStageExecutionError("topology must be an ArtifactTopology.")
        try:
            object.__setattr__(self, "working_dir", Path(self.working_dir))
        except TypeError as exc:
            raise ModelStageExecutionError("working_dir must be coercible to Path.") from exc
        if not isinstance(self.stage_results, tuple) or any(
            not isinstance(result, ModelStageResult) for result in self.stage_results
        ):
            raise ModelStageExecutionError(
                "stage_results must be a tuple of ModelStageResult values."
            )
        if self.progress_session is not None and not isinstance(
            self.progress_session,
            ProgressSession,
        ):
            raise ModelStageExecutionError("progress_session must be a ProgressSession or None.")

    def provider_progress(self, stage: PlannedModelStage):
        """Return the workflow-owned progress adapter for one provider stage."""

        if not isinstance(stage, PlannedModelStage):
            raise ModelStageExecutionError("stage must be a PlannedModelStage.")
        from text_to_sign_production.workflows.model.processing.provider_progress import (
            ModelProviderProgress,
        )

        return ModelProviderProgress(
            progress_session=self.progress_session,
            provider_key=self.request.model_key.value,
            provider_stage_id=stage.spec.kind.value,
            run_mode=self.request.run_mode.value,
        )


@dataclass(frozen=True, slots=True)
class ModelSingleSampleInferenceContext:
    """Inputs visible to provider-owned single-sample inference."""

    request: ModelRunRequest
    loaded_config: ModelProviderLoadedConfig
    topology: ArtifactTopology
    checkpoint_path: Path
    sample: ModelingManifestSample
    output_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.request, ModelRunRequest):
            raise ModelStageExecutionError("request must be a ModelRunRequest.")
        if not isinstance(self.loaded_config, ModelProviderLoadedConfig):
            raise ModelStageExecutionError(
                "loaded_config must be a ModelProviderLoadedConfig."
            )
        if not isinstance(self.topology, ArtifactTopology):
            raise ModelStageExecutionError("topology must be an ArtifactTopology.")
        if not isinstance(self.sample, ModelingManifestSample):
            raise ModelStageExecutionError("sample must be a ModelingManifestSample.")
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "output_root", Path(self.output_root))


class ModelProvider(Protocol):
    """Provider contract implemented by future concrete model candidates."""

    @property
    def spec(self) -> ModelSpec: ...

    @property
    def full_data_pipeline_capability(self) -> ProviderFullDataPipelineCapability: ...

    def full_data_pipeline_capability_for_config(
        self,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ProviderFullDataPipelineCapability:
        return self.full_data_pipeline_capability

    @property
    def full_data_pipeline_declaration(self) -> ProviderFullDataPipelineDeclaration: ...

    def load_config(self, request: ModelRunRequest) -> ModelProviderLoadedConfig: ...

    def calibration_policy(
        self,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ProviderCalibrationPolicy: ...

    def build_representative_calibration_surface(
        self,
        *,
        context: object,
        candidate_key: str,
        policy: ProviderCalibrationPolicy,
        progress_session: ProgressSession | None,
    ) -> RepresentativeCalibrationSurfaceResult: ...

    def plan_stages(
        self,
        request: ModelRunRequest,
        loaded_config: ModelProviderLoadedConfig,
    ) -> ModelStagePlan: ...

    def execute_stage(
        self,
        stage: PlannedModelStage,
        context: ModelStageExecutionContext,
    ) -> ModelStageResult: ...

    def write_model_reports(
        self,
        context: ModelStageExecutionContext,
        results: ModelExecutionResult,
    ) -> tuple[ModelStageArtifactRef, ...]: ...

    def runtime_support_artifacts(
        self,
        *,
        request: ModelRunRequest,
        loaded_config: ModelProviderLoadedConfig,
        execution: ModelExecutionResult,
        topology: ArtifactTopology,
    ) -> tuple[ModelRuntimeSupportArtifact, ...]: ...

    def infer_single_sample(
        self,
        context: ModelSingleSampleInferenceContext,
    ) -> ModelSingleSampleInferenceResult: ...


@dataclass(frozen=True, slots=True)
class ModelStagePlan:
    """Provider-independent, ordered model execution plan."""

    model: ModelSpec
    request: ModelRunRequest
    objective_attachments: tuple[ObjectiveAttachmentPlan, ...]
    stages: tuple[PlannedModelStage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelSpec):
            raise ModelStagePlanError("model must be a ModelSpec.")
        if not isinstance(self.request, ModelRunRequest):
            raise ModelStagePlanError("request must be a ModelRunRequest.")
        if self.model.key is not self.request.model_key:
            raise ModelStagePlanError("model key must match request model_key.")
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ModelStagePlanError("stages must be a non-empty tuple.")
        if any(not isinstance(stage, PlannedModelStage) for stage in self.stages):
            raise ModelStagePlanError("stages must contain PlannedModelStage values.")
        indices = tuple(stage.index for stage in self.stages)
        if indices != tuple(range(len(self.stages))):
            raise ModelStagePlanError("stage indices must be contiguous starting at 0.")
        validate_stage_sequence_for_spec(
            self.model,
            tuple(stage.spec.kind for stage in self.stages),
        )
        if not isinstance(self.objective_attachments, tuple) or any(
            not isinstance(attachment, ObjectiveAttachmentPlan)
            for attachment in self.objective_attachments
        ):
            raise ModelStagePlanError(
                "objective_attachments must contain ObjectiveAttachmentPlan values."
            )
        attachment_keys = tuple(
            attachment.objective.key for attachment in self.objective_attachments
        )
        if attachment_keys != self.request.auxiliary_objectives:
            raise ModelStagePlanError(
                "objective attachments must match requested auxiliary objectives."
            )
        if any(attachment.model.key is not self.model.key for attachment in self.objective_attachments):
            raise ModelStagePlanError("objective attachment models must match plan model.")


def validate_model_provider(provider: ModelProvider) -> None:
    """Validate that a provider declares a registered model contract."""

    spec = getattr(provider, "spec", None)
    if not isinstance(spec, ModelSpec):
        raise ModelProviderRegistrationError("provider spec must be a ModelSpec.")
    try:
        registered_spec = require_model_spec(spec.key)
    except ModelingRegistryError as exc:
        raise ModelProviderRegistrationError(
            f"provider model key is not registered: {spec.key!r}."
        ) from exc
    if spec.canonical_id != registered_spec.canonical_id:
        raise ModelProviderRegistrationError(
            "provider canonical_id must match the registered model spec."
        )
    if spec.generated_pose_required is not True:
        raise ModelProviderRegistrationError("provider models must require generated pose.")
    capability = getattr(provider, "full_data_pipeline_capability", None)
    if not isinstance(capability, ProviderFullDataPipelineCapability):
        raise ModelProviderRegistrationError(
            "provider full_data_pipeline_capability must be declared."
        )
    if capability.provider_key != spec.key.value:
        raise ModelProviderRegistrationError(
            "provider full_data_pipeline_capability provider_key must match spec key."
        )
    if not callable(getattr(provider, "calibration_policy", None)):
        raise ModelProviderRegistrationError(
            "provider must implement callable calibration_policy()."
        )
    try:
        validate_stage_sequence_for_spec(spec, spec.default_stage_sequence)
    except ModelStagePlanError as exc:
        raise ModelProviderRegistrationError(
            "provider model default stage sequence is invalid."
        ) from exc
    for method_name in (
        "load_config",
        "calibration_policy",
        "build_representative_calibration_surface",
        "plan_stages",
        "execute_stage",
        "write_model_reports",
        "runtime_support_artifacts",
    ):
        if not callable(getattr(provider, method_name, None)):
            raise ModelProviderRegistrationError(
                f"provider must implement callable {method_name}()."
            )


def default_stage_plan_for_request(request: ModelRunRequest) -> ModelStagePlan:
    """Build the generic default stage plan declared by a research model spec."""

    if not isinstance(request, ModelRunRequest):
        raise ModelStagePlanError("request must be a ModelRunRequest.")
    model = require_model_spec(request.model_key)
    validate_objective_attachments(model, request.auxiliary_objectives)
    objective_attachments = build_objective_attachment_plans(
        model,
        request.auxiliary_objectives,
    )
    stage_kinds = stage_sequence_from_values(model.default_stage_sequence)
    stages = tuple(
        PlannedModelStage(
            index=index,
            spec=MODEL_STAGE_SPECS[stage_kind],
            provider_stage_id=f"{model.key.value}.{stage_kind.value}",
        )
        for index, stage_kind in enumerate(stage_kinds)
    )
    return ModelStagePlan(
        model=model,
        request=request,
        objective_attachments=objective_attachments,
        stages=stages,
    )


def execute_stage_plan(
    provider: ModelProvider,
    plan: ModelStagePlan,
    *,
    loaded_config: ModelProviderLoadedConfig,
    topology: ArtifactTopology,
    working_dir: Path,
    progress_session: ProgressSession | None = None,
) -> ModelExecutionResult:
    """Execute planned provider stages sequentially and stop on first failure."""

    from text_to_sign_production.workflows.model.processing.performance import (
        begin_stage_performance,
        finish_stage_performance,
        stage_performance_to_metadata,
    )

    validate_model_provider(provider)
    if not isinstance(plan, ModelStagePlan):
        raise ModelStageExecutionError("plan must be a ModelStagePlan.")
    if provider.spec.key is not plan.model.key:
        raise ModelStageExecutionError("provider model key must match plan model key.")
    if not isinstance(loaded_config, ModelProviderLoadedConfig):
        raise ModelStageExecutionError(
            "loaded_config must be a ModelProviderLoadedConfig."
        )
    if loaded_config.model_key is not plan.model.key:
        raise ModelStageExecutionError("loaded config model key must match plan model key.")
    stage_results: list[ModelStageResult] = []
    for stage in plan.stages:
        context = ModelStageExecutionContext(
            request=plan.request,
            loaded_config=loaded_config,
            topology=topology,
            working_dir=working_dir,
            stage_results=tuple(stage_results),
            progress_session=progress_session,
        )
        if progress_session is not None:
            progress_session.status(
                "model stage start",
                model_key=plan.model.key.value,
                stage=stage.spec.kind.value,
                index=stage.index + 1,
                total=len(plan.stages),
                provider_stage_id=stage.provider_stage_id,
            )
        try:
            performance_timer = begin_stage_performance(
                provider_key=plan.model.key.value,
                provider_stage_id=stage.provider_stage_id,
                stage_kind=stage.spec.kind.value,
            )
            result = provider.execute_stage(stage, context)
        except Exception as exc:
            if progress_session is not None:
                progress_session.status(
                    "model stage failed",
                    model_key=plan.model.key.value,
                    stage=stage.spec.kind.value,
                    index=stage.index + 1,
                    total=len(plan.stages),
                    error=type(exc).__name__,
                )
            raise ModelStageExecutionError(
                f"provider failed while executing stage {stage.provider_stage_id!r}."
            ) from exc
        if not isinstance(result, ModelStageResult):
            raise ModelStageExecutionError(
                f"provider stage {stage.provider_stage_id!r} did not return ModelStageResult."
            )
        if result.stage != stage:
            raise ModelStageExecutionError(
                f"provider result does not correspond to stage {stage.provider_stage_id!r}."
            )
        if result.status is ModelStageStatus.FAILED:
            raise ModelStageExecutionError(
                f"provider reported failed stage {stage.provider_stage_id!r}."
            )
        performance = stage_performance_to_metadata(
            finish_stage_performance(performance_timer, metadata=result.metadata)
        )
        result = replace(
            result,
            metadata={**dict(result.metadata), "performance": performance},
        )
        if progress_session is not None:
            progress_session.status(
                "model stage done",
                model_key=plan.model.key.value,
                stage=stage.spec.kind.value,
                index=stage.index + 1,
                total=len(plan.stages),
                status=result.status.value,
            )
        stage_results.append(result)
    return ModelExecutionResult(
        model_key=plan.model.key,
        run_name=plan.request.run_name,
        stages=tuple(stage_results),
    )


def _immutable_string_keyed_mapping(
    value: Mapping[str, object],
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelStagePlanError(f"{field_name} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ModelStagePlanError(f"{field_name} keys must be strings.")
    return MappingProxyType(dict(value))


def _validate_runtime_capability_artifacts(
    *,
    provider_key: str,
    model_key: str,
    run_name: str | None,
    calibration_artifact_path: Path | None,
    selected_overrides_path: Path | None,
) -> None:
    if calibration_artifact_path is None or not calibration_artifact_path.is_file():
        raise ModelStagePlanError("runtime verified capabilities require provider-real calibration artifact.")
    if selected_overrides_path is None or not selected_overrides_path.is_file():
        raise ModelStagePlanError("runtime verified capabilities require selected_overrides artifact.")
    try:
        calibration = json.loads(calibration_artifact_path.read_text(encoding="utf-8"))
        selected = json.loads(selected_overrides_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelStagePlanError("runtime capability artifacts must be readable JSON.") from exc
    if not isinstance(calibration, Mapping) or not isinstance(selected, Mapping):
        raise ModelStagePlanError("runtime capability artifacts must be JSON objects.")
    if calibration.get("provider_key") != provider_key:
        raise ModelStagePlanError("runtime calibration artifact provider_key mismatch.")
    if calibration.get("model_key") not in {None, model_key}:
        raise ModelStagePlanError("runtime calibration artifact model_key mismatch.")
    if run_name is not None and calibration.get("run_name") not in {None, run_name}:
        raise ModelStagePlanError("runtime calibration artifact run_name mismatch.")
    rows = calibration.get("measurements")
    if not isinstance(rows, tuple | list):
        raise ModelStagePlanError("runtime calibration artifact missing measurements.")
    authoritative_trace_ids = {
        row.get("trace_id")
        for row in rows
        if isinstance(row, Mapping)
        and row.get("benchmark_type") == "provider_real"
        and row.get("trace_validation_passed") is True
        and isinstance(row.get("trace_id"), str)
        and row.get("provider_real_trace") is not None
    }
    if not authoritative_trace_ids:
        raise ModelStagePlanError("runtime calibration artifact must contain validated provider-real traces.")
    effective_hash = calibration.get("effective_config_hash")
    build_hash = calibration.get("model_build_spec_hash")
    if not isinstance(effective_hash, str) or len(effective_hash) != 64:
        raise ModelStagePlanError("runtime calibration artifact missing effective_config_hash.")
    if not isinstance(build_hash, str) or len(build_hash) != 64:
        raise ModelStagePlanError("runtime calibration artifact missing model_build_spec_hash.")
    for row in rows:
        if not isinstance(row, Mapping) or row.get("benchmark_type") != "provider_real":
            continue
        if row.get("effective_config_hash") != effective_hash:
            raise ModelStagePlanError("runtime calibration row effective_config_hash mismatch.")
        if row.get("model_build_spec_hash") != build_hash:
            raise ModelStagePlanError("runtime calibration row model_build_spec_hash mismatch.")
        if row.get("surface_provider_config_sha256") != effective_hash:
            raise ModelStagePlanError("runtime calibration row surface provider config hash mismatch.")
    if selected.get("source") != "provider_real_calibration":
        raise ModelStagePlanError("selected_overrides artifact source must be provider_real_calibration.")
    if selected.get("provider_key") != provider_key or selected.get("model_key") != model_key:
        raise ModelStagePlanError("selected_overrides artifact provider/model mismatch.")
    if run_name is not None and selected.get("run_name") != run_name:
        raise ModelStagePlanError("selected_overrides artifact run_name mismatch.")
    if selected.get("effective_config_hash") != effective_hash:
        raise ModelStagePlanError("selected_overrides artifact effective_config_hash mismatch.")
    if selected.get("model_build_spec_hash") != build_hash:
        raise ModelStagePlanError("selected_overrides artifact model_build_spec_hash mismatch.")
    selected_trace_ids = selected.get("selected_candidate_trace_ids")
    if not isinstance(selected_trace_ids, tuple | list) or not selected_trace_ids:
        raise ModelStagePlanError("selected_overrides artifact missing provider-real trace ids.")
    if any(trace_id not in authoritative_trace_ids for trace_id in selected_trace_ids):
        raise ModelStagePlanError("selected_overrides artifact trace id mismatch.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




__all__ = [
    "ModelProvider",
    "ModelProviderLoadedConfig",
    "ProviderCalibrationPolicy",
    "ProviderFullDataPipelineCapability",
    "ProviderFullDataPipelineDeclaration",
    "RuntimeVerifiedFullDataPipeline",
    "VerificationEvidenceRef",
    "VerifiedProviderFullDataPipelineCapability",
    "ModelSingleSampleInferenceContext",
    "ModelStageExecutionContext",
    "ModelStagePlan",
    "default_stage_plan_for_request",
    "execute_stage_plan",
    "validate_model_provider",
]
