"""Read-only smoke execution protocol construction for model notebooks."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.research import ObjectiveKey
from text_to_sign_production.workflows.model.contracts import (
    ModelPreflightResult,
    ModelSmokeExecutionProtocol,
    ModelSmokeHandoff,
    ModelSmokeProtocolStep,
    ModelWorkflowConfig,
)


def build_model_smoke_execution_protocol(
    *,
    config: ModelWorkflowConfig,
    preflight: ModelPreflightResult,
) -> ModelSmokeExecutionProtocol:
    """Build the operator-visible model smoke protocol without executing it."""

    smoke_mode = config.run_mode is ModelRunMode.SMOKE
    ready = preflight.ready and smoke_mode
    warnings = tuple(
        item
        for item in (
            None if preflight.ready else _preflight_warning(preflight),
            None
            if smoke_mode
            else (
                "Model smoke execution is disabled because MODEL_RUN_MODE is "
                f"{config.run_mode.value!r}; set it to 'smoke' before Aşama 15."
            ),
        )
        if item is not None
    )
    return ModelSmokeExecutionProtocol(
        run_name=config.run_name,
        smoke_mode=smoke_mode,
        ready_for_smoke_execution=ready,
        steps=_model_smoke_steps(),
        handoff=ModelSmokeHandoff(
            model_run_name=config.run_name,
            model_key=config.model_key,
            manifest_family=config.manifest_family,
            run_mode=config.run_mode,
            auxiliary_objectives=config.auxiliary_objectives,
            recommended_checkpoint_policy="best",
            test_manifest_path=_test_manifest_path(preflight),
            target_sentence_guidance=(
                "Choose TARGET_SENTENCE_NAME from an exact source_sentence_name row "
                "in the Drive test split manifest."
            ),
            paste_into_test_model={
                "MODEL_RUN_NAME": config.run_name,
                "CHECKPOINT_POLICY": "best",
                "TARGET_SENTENCE_NAME": "<choose-source_sentence_name-from-test-split>",
            },
        ),
        warnings=warnings,
        limitations=_model_smoke_limitations(config),
    )


def _test_manifest_path(preflight: ModelPreflightResult):
    for check in preflight.checks:
        if check.scope == "test_metadata" and check.path is not None:
            return check.path
    return None


def _preflight_warning(preflight: ModelPreflightResult) -> str:
    first_issue = preflight.blocking_issues[0] if preflight.blocking_issues else "unknown"
    return (
        "Model smoke execution is blocked because preflight is not ready; "
        f"blocking issue count={len(preflight.blocking_issues)}, first issue: {first_issue}"
    )


def _model_smoke_steps() -> tuple[ModelSmokeProtocolStep, ...]:
    data = (
        (
            "Review operator configuration",
            "Confirm notebook parameters, generated run name, manifest family, run mode, and objectives.",
            "Notebook parameters are defined.",
            "Any required operator value is missing, placeholder-like, or inconsistent.",
            "Configuration review matches the intended smoke run.",
        ),
        (
            "Build and validate runtime plan",
            "Construct and validate the read-only restore plan before any runtime mutation.",
            "Operator configuration review is accepted.",
            "Runtime plan validation reports layout or restore-operation errors.",
            "Runtime plan review and validation both succeed.",
        ),
        (
            "Run preflight",
            "Check config, Drive inputs, test metadata readiness, provider/objective readiness, and publish target.",
            "Runtime plan validation succeeds.",
            "Any preflight check fails.",
            "Preflight ready is true with zero blocking issues.",
        ),
        (
            "Restore runtime",
            "Copy or extract only planned restore inputs into the runtime workspace.",
            "Preflight ready is true.",
            "Any restore operation fails.",
            "Runtime restore reports success for all required operations.",
        ),
        (
            "Verify runtime",
            "Verify restored runtime assets before provider resolution or execution.",
            "Runtime restore succeeds.",
            "Runtime verification reports required missing or invalid assets.",
            "Runtime verification readiness succeeds.",
        ),
        (
            "Resolve research/provider",
            "Resolve registry-backed model and objective specs plus provider availability.",
            "Runtime verification succeeds.",
            "Model/objective/provider registry resolution fails.",
            "Research and provider reviews identify the selected implementation.",
        ),
        (
            "Load provider config",
            "Load the provider config snapshot selected by the workflow request.",
            "Provider resolution succeeds.",
            "Provider config cannot be loaded or validated.",
            "Provider config review shows the effective config.",
        ),
        (
            "Plan provider stages",
            "Ask the provider for the ordered smoke stage plan.",
            "Provider config is loaded.",
            "Provider stage planning fails or required objectives cannot attach.",
            "Stage plan review lists required provider stages.",
        ),
        (
            "Execute provider stages",
            "Run the provider-owned smoke stages.",
            "Stage plan is accepted.",
            "Any required provider stage fails.",
            "Stage execution review reports completed smoke stages.",
        ),
        (
            "Write validation artifacts",
            "Materialize validation split diagnostic artifacts from generated outputs.",
            "Provider stage execution completes.",
            "Validation artifact pairing or metric writing fails.",
            "Validation artifact review shows written diagnostic outputs.",
        ),
        (
            "Write semantic objective artifacts if requested",
            "If attached, write proxy-only post-generation semantic objective artifacts.",
            "Validation artifacts are available and semantic_consistency was requested.",
            "Semantic config, generated-pose payload, pairing, or ablation-readiness processing fails.",
            "Semantic objective review is explicit about proxy-only interpretation and ablation readiness.",
        ),
        (
            "Write reports and metadata",
            "Write workflow reports and model run metadata from completed smoke outputs.",
            "Required validation and objective artifacts are available.",
            "Report or metadata writing fails.",
            "Reports and model metadata receipts are available.",
        ),
        (
            "Plan publish",
            "Build the publish plan for completed runtime artifacts.",
            "Reports and metadata have been written.",
            "Publish source bundle is incomplete or invalid.",
            "Publish plan review lists all required sources and targets.",
        ),
        (
            "Execute publish",
            "Copy planned runtime artifacts to the Drive publish surface.",
            "Publish plan is accepted.",
            "Any required publish copy fails.",
            "Publish execution review reports success.",
        ),
        (
            "Verify publish",
            "Verify the published Drive outputs against the publish execution plan.",
            "Publish execution succeeds.",
            "Any required published artifact fails verification.",
            "Publish verification succeeds.",
        ),
        (
            "Review final summary",
            "Review the final model workflow summary without upgrading smoke results to quality claims.",
            "Publish verification succeeds.",
            "Final result reports workflow failure or missing required evidence.",
            "Final summary reports a completed smoke workflow and its limitations.",
        ),
        (
            "Copy handoff values to test_model.ipynb",
            "Copy the generated run name, checkpoint policy, and chosen test sentence into test_model.ipynb.",
            "Final summary has been reviewed.",
            "Any handoff value is missing or copied from an artifact path instead of the displayed request values.",
            "test_model.ipynb request fields are populated from the handoff.",
        ),
    )
    return tuple(
        ModelSmokeProtocolStep(
            order=index,
            name=name,
            purpose=purpose,
            starts_after=starts_after,
            stop_condition=stop_condition,
            success_signal=success_signal,
        )
        for index, (name, purpose, starts_after, stop_condition, success_signal) in enumerate(
            data,
            start=1,
        )
    )


def _model_smoke_limitations(config: ModelWorkflowConfig) -> tuple[str, ...]:
    semantic = ObjectiveKey.SEMANTIC_CONSISTENCY in config.auxiliary_objectives
    semantic_limitations = (
        (
            "If semantic_consistency is attached, it is proxy-only post-generation evaluation.",
            "semantic_consistency is not differentiable semantic training optimization.",
            "semantic ablation is not complete.",
        )
        if semantic
        else (
            "semantic_consistency is not attached in this request; no semantic objective outputs are expected.",
            "semantic ablation is not complete.",
        )
    )
    return (
        "Smoke run is not full training.",
        "Smoke run does not prove model quality.",
        "Validation metrics do not prove sign intelligibility.",
        "Test split is not used for training/validation.",
        "During controlled execution, preserve notebook cell outputs/logs for review.",
        "A failed preflight stops before restore.",
        "A failed provider stage stops before publish.",
        "Do not manually skip failed gates.",
        *semantic_limitations,
    )


__all__ = ["build_model_smoke_execution_protocol"]
