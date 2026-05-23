"""Console review sections for test_model."""

from __future__ import annotations

from text_to_sign_production.workflows.foundation.review import (
    WorkflowReviewSection,
    review_item,
    review_lines_section,
    review_section,
)


def review_request(request) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Operator request",
            (
                review_item(
                    "test_model",
                    (
                        ("MODEL_RUN_NAME", request.model_run_name),
                        ("CHECKPOINT_POLICY", request.checkpoint_policy.value),
                        ("TARGET_SENTENCE_NAME", request.target_sentence_name),
                    ),
                ),
            ),
        ),
    )


def review_preflight(result) -> tuple[WorkflowReviewSection, ...]:
    scopes = tuple(dict.fromkeys(check.scope for check in result.checks))
    return (
        review_section(
            "Preflight summary",
            (
                review_item(
                    result.model_run_name,
                    (
                        ("ready", result.ready),
                        ("checkpoint policy", result.checkpoint_policy.value),
                        ("target sentence name", result.target_sentence_name),
                        (
                            "resolved model key",
                            None if result.resolved_model_key is None else result.resolved_model_key.value,
                        ),
                        (
                            "resolved manifest family",
                            None
                            if result.resolved_manifest_family is None
                            else result.resolved_manifest_family.family_id,
                        ),
                        ("resolved sample id", result.resolved_sample_id),
                        (
                            "auxiliary objectives",
                            tuple(
                                objective.value
                                for objective in result.auxiliary_objectives
                            ),
                        ),
                        (
                            "semantic objective attached",
                            result.semantic_objective_attached,
                        ),
                        (
                            "semantic ready for comparison",
                            result.semantic_ready_for_comparison,
                        ),
                        (
                            "semantic required baseline missing",
                            result.semantic_required_baseline_missing,
                        ),
                        (
                            "semantic ablation status",
                            result.semantic_ablation_status,
                        ),
                        ("check count", len(result.checks)),
                        ("blocking issue count", len(result.blocking_issues)),
                        ("warning count", len(result.warnings)),
                    ),
                ),
            ),
        ),
        review_section(
            "Preflight blocking issues",
            tuple(review_item(issue) for issue in result.blocking_issues),
        ),
        review_section(
            "Preflight warnings",
            tuple(review_item(warning) for warning in result.warnings),
        ),
        review_section(
            "Semantic objective interpretation",
            (
                review_item(
                    "semantic_consistency",
                    (
                        (
                            "proxy-only post-generation evaluation",
                            result.semantic_objective_attached,
                        ),
                        (
                            "semantic ablation may be incomplete",
                            result.semantic_objective_attached,
                        ),
                        (
                            "test_model executes semantic ablation comparison",
                            False,
                        ),
                    ),
                ),
            ),
        ),
        *(
            review_section(
                f"Preflight checks: {scope}",
                tuple(
                    review_item(
                        check.name,
                        (
                            ("status", check.status),
                            ("message", check.message),
                            ("path", check.path),
                        ),
                    )
                    for check in result.checks
                    if check.scope == scope
                ),
            )
            for scope in scopes
        ),
        review_section(
            "Expected restore inputs and support artifacts",
            tuple(
                review_item(
                    item.label,
                    (
                        ("artifact family", item.artifact_family),
                        ("source", item.source_path),
                        ("target", item.target_path),
                        ("required for restore", item.required_for_restore),
                        ("provider key", item.provider_key),
                    ),
                )
                for item in result.expected_inputs
            ),
        ),
        review_section(
            "Expected execution/report/publish outputs",
            tuple(
                review_item(
                    output.label,
                    (
                        ("artifact family", output.artifact_family),
                        ("path", output.path),
                        ("required for success", output.required_for_success),
                    ),
                )
                for output in result.expected_outputs
            ),
        ),
    )


def review_restore_plan(plan) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Restore plan summary",
            (
                review_item(
                    plan.model_run_name,
                    (
                        ("valid", plan.valid),
                        ("operation count", len(plan.operations)),
                        ("blocking errors", len(plan.blocking_errors)),
                        ("warnings", len(plan.warnings)),
                    ),
                ),
            ),
        ),
        review_section(
            "Restore operations",
            tuple(
                review_item(
                    operation.description,
                    (
                        ("group", operation.group),
                        ("required", operation.required),
                        ("source", operation.source),
                        ("target", operation.target),
                    ),
                )
                for operation in plan.operations
            ),
        ),
        _issues_section("Restore plan errors", plan.blocking_errors),
        _issues_section("Restore plan warnings", plan.warnings),
    )


def review_smoke_execution_protocol(protocol) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Test-model smoke protocol summary",
            (
                review_item(
                    protocol.model_run_name,
                    (
                        ("ready for smoke execution", protocol.ready_for_smoke_execution),
                        ("checkpoint policy", protocol.checkpoint_policy.value),
                        ("target sentence name", protocol.target_sentence_name),
                        (
                            "auxiliary objectives",
                            tuple(
                                objective.value
                                for objective in protocol.auxiliary_objectives
                            ),
                        ),
                        (
                            "semantic objective attached",
                            protocol.semantic_objective_attached,
                        ),
                        (
                            "semantic ready for comparison",
                            protocol.semantic_ready_for_comparison,
                        ),
                        (
                            "semantic required baseline missing",
                            protocol.semantic_required_baseline_missing,
                        ),
                        (
                            "semantic ablation status",
                            protocol.semantic_ablation_status,
                        ),
                        ("step count", len(protocol.steps)),
                    ),
                ),
            ),
        ),
        review_section(
            "Ready for smoke execution",
            (
                review_item(
                    "readiness",
                    (
                        ("ready", protocol.ready_for_smoke_execution),
                        ("preflight gate required before restore/inference/publish", True),
                        (
                            "failed preflight stops before restore/inference/publish",
                            True,
                        ),
                    ),
                ),
            ),
        ),
        review_section(
            "Ordered smoke steps",
            tuple(
                review_item(
                    f"{step.order}. {step.name}",
                    (
                        ("purpose", step.purpose),
                        ("starts after", step.starts_after),
                    ),
                )
                for step in protocol.steps
            ),
        ),
        review_section(
            "Smoke stop conditions",
            tuple(
                review_item(
                    f"{step.order}. {step.name}",
                    (("stop condition", step.stop_condition),),
                )
                for step in protocol.steps
            ),
        ),
        review_section(
            "Smoke success signals",
            tuple(
                review_item(
                    f"{step.order}. {step.name}",
                    (("success signal", step.success_signal),),
                )
                for step in protocol.steps
            ),
        ),
        review_section(
            "Test-model smoke warnings",
            tuple(review_item(warning) for warning in protocol.warnings),
        ),
        review_section(
            "Test-model smoke limitations",
            tuple(review_item(limitation) for limitation in protocol.limitations),
        ),
    )


def review_restore_plan_validation(validation) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Restore plan validation",
            (
                review_item(
                    "validation",
                    (
                        ("succeeded", validation.succeeded),
                        ("valid", validation.valid),
                        ("error count", len(validation.blocking_errors)),
                        ("warning count", len(validation.warnings)),
                    ),
                ),
            ),
        ),
        _issues_section("Validation errors", validation.blocking_errors),
        _issues_section("Validation warnings", validation.warnings),
    )


def review_restore_result(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Restore result",
            (
                review_item(
                    "runtime restore",
                    (
                        ("succeeded", result.succeeded),
                        ("operation count", len(result.operation_results)),
                    ),
                ),
            ),
        ),
        review_section(
            "Restore details",
            tuple(
                review_item(
                    item.operation.description,
                    (("status", item.status), ("message", item.message)),
                )
                for item in result.operation_results
            ),
        ),
    )


def review_runtime_verification(verification) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime verification",
            (
                review_item(
                    "readiness",
                    (
                        ("succeeded", verification.succeeded),
                        ("check count", len(verification.checks)),
                        ("error count", len(verification.errors)),
                        ("warning count", len(verification.warnings)),
                    ),
                ),
            ),
        ),
        review_section(
            "Runtime checks",
            tuple(
                review_item(
                    check.label,
                    (
                        ("required", check.required),
                        ("exists", check.exists),
                        ("valid", check.valid),
                        ("path", check.path),
                        ("message", check.message),
                    ),
                )
                for check in verification.checks
            ),
        ),
    )


def review_model_run_resolution(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Model run resolution",
            (
                review_item(
                    result.model_run_name,
                    (
                        ("succeeded", result.succeeded),
                        ("model_key", result.model_key),
                        (
                            "manifest_family",
                            None
                            if result.manifest_family is None
                            else result.manifest_family.family_id,
                        ),
                        ("train_split", result.train_split.value),
                        ("validation_split", result.validation_split.value),
                        ("test_split", result.test_split.value),
                        ("status", result.status),
                        ("metadata", result.run_metadata_path),
                        ("errors", result.errors),
                    ),
                ),
            ),
        ),
    )


def review_checkpoint_selection(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Checkpoint selection",
            (
                review_item(
                    result.policy.value,
                    (
                        ("succeeded", result.succeeded),
                        ("role", result.checkpoint_role),
                        ("path", result.checkpoint_path),
                        ("exists", result.checkpoint_exists),
                        ("errors", result.errors),
                    ),
                ),
            ),
        ),
    )


def review_target_resolution(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Target resolution",
            (
                review_item(
                    result.target_sentence_name,
                    (
                        ("status", result.status),
                        ("resolved sample id", result.resolved_sample_id),
                        ("source sentence name", result.source_sentence_name),
                        (
                            "manifest family",
                            None
                            if result.manifest_family is None
                            else result.manifest_family.family_id,
                        ),
                        ("split", result.split.value),
                        ("manifest", result.manifest_path),
                        ("issues", result.issues),
                        ("warnings", result.warnings),
                    ),
                ),
            ),
        ),
    )


def review_sample_evidence(result) -> tuple[WorkflowReviewSection, ...]:
    model_context = result.model_context or {}
    checkpoint_context = result.checkpoint_context or {}
    return (
        review_section(
            "Sample evidence",
            (
                review_item(
                    result.target.target_sentence_name,
                    (
                        ("source video", result.source_video_path),
                        ("source video exists", result.source_video_exists),
                        ("prepared payload", result.prepared_payload_path),
                        ("prepared payload exists", result.prepared_payload_exists),
                        ("model run name", model_context.get("model_run_name")),
                        ("model key", model_context.get("model_key")),
                        ("manifest family", model_context.get("manifest_family")),
                        ("run mode", model_context.get("run_mode")),
                        ("train split", model_context.get("train_split")),
                        ("validation split", model_context.get("validation_split")),
                        ("test split", model_context.get("test_split")),
                        ("checkpoint policy", checkpoint_context.get("policy")),
                        ("checkpoint role", checkpoint_context.get("checkpoint_role")),
                        ("checkpoint path", checkpoint_context.get("checkpoint_path")),
                        ("checkpoint exists", checkpoint_context.get("checkpoint_exists")),
                    ),
                ),
            ),
        ),
        review_lines_section("Consistency notes", result.consistency_notes),
    )


def review_inference_result(result) -> tuple[WorkflowReviewSection, ...]:
    provider = result.provider_result
    return (
        review_section(
            "Inference result",
            (
                review_item(
                    result.status,
                    (
                        ("succeeded", result.succeeded),
                        ("generated pose", None if provider is None else provider.generated_payload_path),
                        ("manifest", None if provider is None else provider.generated_manifest_path),
                        ("metadata", result.metadata_path),
                        ("errors", result.errors),
                    ),
                ),
            ),
        ),
    )


def review_reference_comparison_result(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Reference comparison result",
            (
                review_item(
                    result.status,
                    (
                        ("status", result.status),
                        ("succeeded", result.succeeded),
                        ("reference_sample_id", result.reference_sample_id),
                        ("generated_sample_id", result.generated_sample_id),
                        ("reference_frame_count", result.reference_frame_count),
                        ("generated_frame_count", result.generated_frame_count),
                        ("aligned_frame_count", result.aligned_frame_count),
                        ("errors", result.errors),
                        ("warnings", result.warnings),
                    ),
                ),
            ),
        ),
        review_section(
            "Reference comparison metrics",
            (
                review_item(
                    "overall",
                    tuple(
                        (metric_key, _metric_value(result, metric_key))
                        for metric_key in (
                            "masked_l1_mean",
                            "masked_l2_mean",
                            "velocity_l1_mean",
                            "velocity_l2_mean",
                            "sequence_length_absolute_error",
                            "valid_joint_coverage",
                        )
                    ),
                ),
            ),
        ),
        review_section(
            "Reference comparison channel metrics",
            tuple(
                review_item(
                    channel,
                    (
                        (
                            "masked_l1_mean",
                            _channel_metric_value(
                                result, channel, "channel_masked_l1_mean"
                            ),
                        ),
                        (
                            "masked_l2_mean",
                            _channel_metric_value(
                                result, channel, "channel_masked_l2_mean"
                            ),
                        ),
                        (
                            "velocity_l1_mean",
                            _channel_metric_value(
                                result, channel, "channel_velocity_l1_mean"
                            ),
                        ),
                        (
                            "velocity_l2_mean",
                            _channel_metric_value(
                                result, channel, "channel_velocity_l2_mean"
                            ),
                        ),
                        (
                            "valid_joint_coverage",
                            _channel_metric_value(
                                result, channel, "channel_valid_joint_coverage"
                            ),
                        ),
                    ),
                )
                for channel in tuple(
                    dict.fromkeys(metric.channel for metric in result.channel_metrics)
                )
            ),
        ),
    )


def review_visualization_result(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Visualization result",
            (
                review_item(
                    result.status,
                    (
                        ("output root", result.output_root),
                        ("artifact count", len(result.artifacts)),
                        ("errors", result.errors),
                        ("warnings", result.warnings),
                    ),
                ),
            ),
        ),
        review_section(
            "Visual artifacts",
            tuple(
                review_item(
                    artifact.label,
                    (
                        ("created", artifact.created),
                        ("path", artifact.path),
                        ("error", artifact.error),
                    ),
                )
                for artifact in result.artifacts
            ),
        ),
    )


def review_report_outputs(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Reports",
            (
                review_item(
                    "test_model reports",
                    (
                        ("succeeded", result.succeeded),
                        ("output root", result.output_root),
                        ("file count", len(result.files)),
                        ("errors", result.errors),
                    ),
                ),
            ),
        ),
        review_lines_section("Report files", tuple(path.as_posix() for path in result.files)),
    )


def review_publish_plan(plan) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish plan",
            (
                review_item(
                    "test_model publish",
                    (
                        ("valid", plan.valid),
                        ("runtime output root", plan.runtime_output_root),
                        ("Drive output root", plan.drive_output_root),
                        ("target count", len(plan.targets)),
                    ),
                ),
            ),
        ),
    )


def review_publish_execution(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish execution",
            (
                review_item(
                    "test_model publish",
                    (
                        ("succeeded", result.succeeded),
                        ("published file count", result.published_file_count),
                        ("error count", len(result.errors)),
                    ),
                ),
            ),
        ),
    )


def review_publish_verification(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish verification",
            (
                review_item(
                    "digest verification",
                    (
                        ("succeeded", result.succeeded),
                        ("check count", len(result.checks)),
                    ),
                ),
            ),
        ),
    )


def review_publish_result(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish result",
            (
                review_item(
                    "test_model publish",
                    (
                        ("execution succeeded", result.execution.succeeded),
                        ("verification succeeded", result.verification.succeeded),
                    ),
                ),
            ),
        ),
    )


def review_final_operator_summary(result) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Final summary",
            (
                review_item(
                    "single-sample model test",
                    (
                        ("succeeded", result.succeeded),
                        (
                            "target sentence name",
                            None if result.target is None else result.target.target_sentence_name,
                        ),
                        (
                            "resolved sample id",
                            None if result.target is None else result.target.resolved_sample_id,
                        ),
                        ("scope", "one unseen test sample diagnostic; not aggregate evaluation"),
                    ),
                ),
            ),
        ),
    )


def _issues_section(title: str, issues) -> WorkflowReviewSection:
    return review_section(
        title,
        tuple(
            review_item(
                issue.code,
                (("message", issue.message), ("path", issue.path), ("label", issue.label)),
            )
            for issue in issues
        ),
    )


def _metric_value(result, metric_key: str):
    for metric in result.metrics:
        if metric.metric_key == metric_key:
            return metric.value
    return None


def _channel_metric_value(result, channel: str, metric_key: str):
    for metric in result.channel_metrics:
        if metric.channel == channel and metric.metric_key == metric_key:
            return metric.value
    return None


__all__ = [name for name in globals() if name.startswith("review_")]
