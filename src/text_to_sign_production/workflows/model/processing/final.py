"""Build the final notebook-facing model workflow result."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates import ModelRunRequest
from text_to_sign_production.workflows.model.contracts import (
    ModelPublishResult,
    ModelReportArtifacts,
    ModelObjectiveArtifactResult,
    ModelResearchResolution,
    ModelRunMetadataArtifacts,
    ModelValidationArtifactResult,
    ModelRuntimeVerification,
    ModelStageExecutionWorkflowResult,
    ModelWorkflowConfig,
    ModelWorkflowFinalResult,
)


def build_model_final_result(
    *,
    config: ModelWorkflowConfig,
    request: ModelRunRequest,
    research: ModelResearchResolution | None = None,
    provider_available: bool = False,
    runtime_verification: ModelRuntimeVerification | None = None,
    stage_execution: ModelStageExecutionWorkflowResult | None = None,
    metadata_artifacts: ModelRunMetadataArtifacts | None = None,
    validation_artifacts: ModelValidationArtifactResult | None = None,
    objective_artifacts: tuple[ModelObjectiveArtifactResult, ...] = (),
    report_artifacts: ModelReportArtifacts | None = None,
    publish_result: ModelPublishResult | None = None,
) -> ModelWorkflowFinalResult:
    """Collect completed and intentionally unavailable workflow surfaces."""

    return ModelWorkflowFinalResult(
        config=config,
        request=request,
        research=research,
        provider_available=provider_available,
        runtime_verification=runtime_verification,
        stage_execution=stage_execution,
        metadata_artifacts=metadata_artifacts,
        validation_artifacts=validation_artifacts,
        objective_artifacts=objective_artifacts,
        report_artifacts=report_artifacts,
        publish_result=publish_result,
    )


__all__ = ["build_model_final_result"]
