"""Public contracts for model candidate providers and stage orchestration."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates.configs import (
    ModelRunMode,
    ModelRunRequest,
    model_run_request_to_dict,
    validate_objective_attachments,
)
from text_to_sign_production.modeling.candidates.run_modes import (
    ModelRunModePolicy,
    resolve_model_run_mode_policy,
)
from text_to_sign_production.modeling.candidates.errors import (
    ModelCandidateError,
    ModelProviderLookupError,
    ModelProviderRegistrationError,
    ModelStageExecutionError,
    ModelStagePlanError,
    ObjectiveAttachmentError,
)
from text_to_sign_production.modeling.candidates.objectives import (
    ObjectiveAttachmentPlan,
    build_objective_attachment_plans,
)
from text_to_sign_production.modeling.candidates.provider import (
    ModelProvider,
    ModelProviderLoadedConfig,
    ProviderCalibrationPolicy,
    ProviderFullDataPipelineDeclaration,
    ProviderFullDataPipelineCapability,
    RuntimeVerifiedFullDataPipeline,
    VerificationEvidenceRef,
    VerifiedProviderFullDataPipelineCapability,
    ModelSingleSampleInferenceContext,
    ModelStageExecutionContext,
    ModelStagePlan,
    default_stage_plan_for_request,
    execute_stage_plan,
    validate_model_provider,
)
from text_to_sign_production.modeling.candidates.calibration_surfaces import (
    RepresentativeCalibrationSurfaceResult,
)
from text_to_sign_production.modeling.candidates.registry import (
    DEFAULT_MODEL_PROVIDER_REGISTRY,
    ModelProviderRegistry,
    get_model_provider,
    list_model_providers,
    register_model_provider,
    require_model_provider,
)
from text_to_sign_production.modeling.candidates.results import (
    ModelExecutionResult,
    ModelSingleSampleInferenceResult,
    ModelStageArtifactRef,
    ModelStageResult,
    ModelStageStatus,
    is_generated_pose_manifest_role,
    provider_stage_artifact_role_from_value,
)
from text_to_sign_production.modeling.candidates.runtime_support import (
    MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION,
    ModelRuntimeSupportArtifact,
    ModelRuntimeSupportManifest,
    read_runtime_support_manifest,
    support_artifact_from_model_run_file,
    write_runtime_support_manifest,
)
from text_to_sign_production.modeling.candidates.stages import (
    MODEL_STAGE_SPECS,
    ModelStageCategory,
    ModelStageKind,
    ModelStageSpec,
    PlannedModelStage,
    model_stage_kind_from_value,
    stage_sequence_from_values,
    validate_stage_sequence_for_spec,
)

__all__ = [
    "ModelCandidateError",
    "ModelProviderRegistrationError",
    "ModelProviderLookupError",
    "ModelStagePlanError",
    "ModelStageExecutionError",
    "ObjectiveAttachmentError",
    "ModelStageKind",
    "ModelStageCategory",
    "ModelStageSpec",
    "PlannedModelStage",
    "MODEL_STAGE_SPECS",
    "model_stage_kind_from_value",
    "stage_sequence_from_values",
    "validate_stage_sequence_for_spec",
    "ModelRunMode",
    "ModelRunRequest",
    "ModelRunModePolicy",
    "resolve_model_run_mode_policy",
    "model_run_request_to_dict",
    "validate_objective_attachments",
    "ObjectiveAttachmentPlan",
    "build_objective_attachment_plans",
    "ModelStageArtifactRef",
    "ModelStageStatus",
    "ModelStageResult",
    "ModelExecutionResult",
    "ModelSingleSampleInferenceResult",
    "MODEL_RUNTIME_SUPPORT_SCHEMA_VERSION",
    "ModelRuntimeSupportArtifact",
    "ModelRuntimeSupportManifest",
    "read_runtime_support_manifest",
    "support_artifact_from_model_run_file",
    "write_runtime_support_manifest",
    "provider_stage_artifact_role_from_value",
    "is_generated_pose_manifest_role",
    "ModelProviderLoadedConfig",
    "ProviderCalibrationPolicy",
    "RepresentativeCalibrationSurfaceResult",
    "ProviderFullDataPipelineCapability",
    "ProviderFullDataPipelineDeclaration",
    "RuntimeVerifiedFullDataPipeline",
    "VerificationEvidenceRef",
    "VerifiedProviderFullDataPipelineCapability",
    "ModelSingleSampleInferenceContext",
    "ModelStageExecutionContext",
    "ModelStagePlan",
    "ModelProvider",
    "validate_model_provider",
    "default_stage_plan_for_request",
    "execute_stage_plan",
    "ModelProviderRegistry",
    "DEFAULT_MODEL_PROVIDER_REGISTRY",
    "register_model_provider",
    "get_model_provider",
    "require_model_provider",
    "list_model_providers",
]
