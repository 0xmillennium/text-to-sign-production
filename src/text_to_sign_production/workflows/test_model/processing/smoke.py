"""Read-only smoke execution protocol construction for test_model."""

from __future__ import annotations

from text_to_sign_production.workflows.test_model.contracts import (
    TestModelPreflightResult,
    TestModelRequest,
    TestModelSmokeExecutionProtocol,
    TestModelSmokeProtocolStep,
)

_MODEL_RUN_PLACEHOLDERS = frozenset(
    {
        "paste-generated-model-run-here",
        "paste-model-run-here",
        "replace-with-model-run",
    }
)
_TARGET_PLACEHOLDERS = frozenset(
    {
        "paste-test-source-sentence-name-here",
        "paste-target-sentence-name-here",
        "replace-with-target-sentence",
    }
)


def build_test_model_smoke_execution_protocol(
    *,
    request: TestModelRequest,
    preflight: TestModelPreflightResult,
) -> TestModelSmokeExecutionProtocol:
    """Build the operator-visible test_model smoke protocol without executing it."""

    placeholders_resolved = _placeholders_resolved(request)
    ready = preflight.ready and placeholders_resolved
    warnings = tuple(
        item
        for item in (
            None if preflight.ready else _preflight_warning(preflight),
            None
            if placeholders_resolved
            else (
                "Test-model smoke execution is blocked because MODEL_RUN_NAME or "
                "TARGET_SENTENCE_NAME is still a placeholder."
            ),
        )
        if item is not None
    )
    return TestModelSmokeExecutionProtocol(
        model_run_name=request.model_run_name,
        checkpoint_policy=request.checkpoint_policy,
        target_sentence_name=request.target_sentence_name,
        auxiliary_objectives=preflight.auxiliary_objectives,
        semantic_objective_attached=preflight.semantic_objective_attached,
        semantic_ready_for_comparison=preflight.semantic_ready_for_comparison,
        semantic_required_baseline_missing=preflight.semantic_required_baseline_missing,
        semantic_ablation_status=preflight.semantic_ablation_status,
        ready_for_smoke_execution=ready,
        steps=_test_model_smoke_steps(),
        warnings=warnings,
        limitations=(
            "Single-sample diagnostic is not aggregate model evaluation.",
            "Visual output is for operator review.",
            "Test_model does not prove model quality.",
            "If selected model run has semantic_consistency, test_model does not execute semantic ablation comparison.",
            "No checkpoint fallback is allowed.",
            "During controlled execution, preserve notebook cell outputs/logs for review.",
            "A failed preflight stops before restore.",
            "A failed inference or visualization stage stops before publish.",
            "Do not manually skip failed gates.",
        ),
    )


def _placeholders_resolved(request: TestModelRequest) -> bool:
    return (
        request.model_run_name.strip().lower() not in _MODEL_RUN_PLACEHOLDERS
        and request.target_sentence_name.strip().lower() not in _TARGET_PLACEHOLDERS
    )


def _preflight_warning(preflight: TestModelPreflightResult) -> str:
    first_issue = preflight.blocking_issues[0] if preflight.blocking_issues else "unknown"
    return (
        "Test-model smoke execution is blocked because preflight is not ready; "
        f"blocking issue count={len(preflight.blocking_issues)}, first issue: {first_issue}"
    )


def _test_model_smoke_steps() -> tuple[TestModelSmokeProtocolStep, ...]:
    data = (
        (
            "Review operator configuration",
            "Confirm selected model run, checkpoint policy, and target test sentence.",
            "Notebook parameters are defined.",
            "Any request value is missing, placeholder-like, or inconsistent.",
            "Operator request review matches the intended single-sample diagnostic.",
        ),
        (
            "Build request",
            "Construct the typed test_model request from notebook parameters.",
            "Operator configuration is accepted.",
            "Request validation fails.",
            "Typed request exists with no placeholders.",
        ),
        (
            "Run preflight",
            "Check selected run metadata, checkpoint reference, target sample, Drive inputs, and provider readiness.",
            "Typed request is built.",
            "Any preflight check fails.",
            "Preflight ready is true with zero blocking issues.",
        ),
        (
            "Build restore plan",
            "Create the restore plan for the selected model run and test split support artifacts.",
            "Preflight ready is true.",
            "Restore plan construction cannot resolve required sources.",
            "Restore plan review lists required Drive sources and runtime targets.",
        ),
        (
            "Validate restore plan",
            "Validate restore sources and target runtime paths before copying.",
            "Restore plan is built.",
            "Any required restore source or target alignment check fails.",
            "Restore plan validation is valid.",
        ),
        (
            "Restore runtime",
            "Copy planned model, checkpoint, and test sample support artifacts into runtime.",
            "Restore plan validation succeeds.",
            "Any required restore operation fails.",
            "Runtime restore reports success.",
        ),
        (
            "Verify runtime",
            "Verify restored files before model resolution or inference.",
            "Runtime restore succeeds.",
            "Runtime verification reports required missing or invalid files.",
            "Runtime verification succeeds.",
        ),
        (
            "Resolve model run",
            "Read and validate the restored model run metadata.",
            "Runtime verification succeeds.",
            "Model run metadata cannot be resolved or is not completed.",
            "Model run resolution succeeds.",
        ),
        (
            "Select checkpoint without fallback",
            "Select the requested checkpoint role exactly.",
            "Model run metadata is resolved.",
            "Requested checkpoint is missing; no fallback is allowed.",
            "Checkpoint selection succeeds for the requested policy.",
        ),
        (
            "Resolve target test sample by sentence name",
            "Resolve the exact target source_sentence_name in the test manifest.",
            "Model run and checkpoint are resolved.",
            "Target sentence is missing or ambiguous.",
            "Target resolution returns one test sample.",
        ),
        (
            "Collect source evidence",
            "Collect source video, prepared payload, model, and checkpoint context for review.",
            "Target sample is resolved.",
            "Required source evidence is missing.",
            "Evidence review shows all required diagnostic inputs.",
        ),
        (
            "Run single-sample inference",
            "Run provider inference for exactly one resolved test sample.",
            "Source evidence is available.",
            "Provider inference fails or does not produce the required generated payload.",
            "Inference review shows the generated pose payload and manifest.",
        ),
        (
            "Render skeleton and side-by-side videos",
            "Render operator-review visualizations from source evidence and generated pose.",
            "Single-sample inference succeeds.",
            "Required visualization cannot be rendered.",
            "Visualization review lists the generated videos.",
        ),
        (
            "Write reports",
            "Write diagnostic reports for the single-sample test_model run.",
            "Visualization succeeds.",
            "Report writing fails.",
            "Report review lists written diagnostic files.",
        ),
        (
            "Plan publish",
            "Build the publish plan for test_model diagnostic outputs.",
            "Reports are written.",
            "Publish plan is invalid or missing required sources.",
            "Publish plan review lists runtime and Drive outputs.",
        ),
        (
            "Execute publish",
            "Copy planned diagnostic outputs to Drive.",
            "Publish plan is accepted.",
            "Any required publish operation fails.",
            "Publish execution review reports success.",
        ),
        (
            "Verify publish",
            "Verify the published diagnostic outputs.",
            "Publish execution succeeds.",
            "Any required published file fails verification.",
            "Publish verification succeeds.",
        ),
        (
            "Review final summary",
            "Review final single-sample diagnostic status and limitations.",
            "Publish verification succeeds.",
            "Final result reports failure or missing required evidence.",
            "Final summary reports a completed one-sample diagnostic without quality claims.",
        ),
    )
    return tuple(
        TestModelSmokeProtocolStep(
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


__all__ = ["build_test_model_smoke_execution_protocol"]
