"""Operator-facing review sections for model workflow orchestration."""

from __future__ import annotations

from collections.abc import Mapping

from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
    operation_kind,
)
from text_to_sign_production.workflows.foundation.review import (
    RenderableValue,
    WorkflowReviewSection,
    review_item,
    review_section,
    to_review_value,
)
from text_to_sign_production.workflows.model.contracts import (
    ModelProviderConfigResult,
    ModelProviderResolution,
    ModelPreflightResult,
    ModelReportArtifacts,
    ModelObjectiveArtifactResult,
    ModelResearchResolution,
    ModelRunMetadataArtifacts,
    ModelSmokeExecutionProtocol,
    ModelValidationArtifactResult,
    ModelRuntimePlan,
    ModelRuntimeRestoreResult,
    ModelRuntimeVerification,
    ModelStageArtifactReceiptResult,
    ModelStageExecutionWorkflowResult,
    ModelStagePlanningResult,
    ModelWorkflowFinalResult,
)


def review_preflight(result: ModelPreflightResult) -> tuple[WorkflowReviewSection, ...]:
    scopes = tuple(dict.fromkeys(check.scope for check in result.checks))
    return (
        review_section(
            "Model preflight summary",
            (
                review_item(
                    result.run_name,
                    (
                        ("ready", result.ready),
                        ("model key", result.model_key.value),
                        ("manifest family", result.manifest_family.family_id),
                        ("run mode", result.run_mode.value),
                        (
                            "auxiliary objectives",
                            tuple(objective.value for objective in result.auxiliary_objectives),
                        ),
                        ("check count", len(result.checks)),
                        ("blocking issue count", len(result.blocking_issues)),
                        ("warning count", len(result.warnings)),
                    ),
                ),
            ),
        ),
        review_section(
            "Model preflight blocking issues",
            tuple(review_item(issue) for issue in result.blocking_issues),
        ),
        review_section(
            "Model preflight warnings",
            tuple(review_item(warning) for warning in result.warnings),
        ),
        *(
            review_section(
                f"Model preflight checks: {scope}",
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
            "Model expected restore inputs",
            tuple(
                review_item(
                    item.label,
                    (
                        ("artifact family", item.artifact_family),
                        ("source", item.source_path),
                        ("target", item.target_path),
                        ("required for restore", item.required_for_restore),
                    ),
                )
                for item in result.expected_inputs
            ),
        ),
        review_section(
            "Model expected execution/report/publish outputs",
            tuple(
                review_item(
                    output.label,
                    (
                        ("artifact family", output.artifact_family),
                        ("path", output.path),
                        ("required for success", output.required_for_success),
                        (
                            "materialized before execution",
                            output.materialized_before_execution,
                        ),
                    ),
                )
                for output in result.expected_outputs
            ),
        ),
    )


def review_runtime_plan(plan: ModelRuntimePlan) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime plan summary",
            (
                review_item(
                    "model",
                    (
                        ("model key", plan.execution_inputs.request.model_key.value),
                        ("run name", plan.execution_inputs.request.run_name),
                        (
                            "manifest family",
                            plan.execution_inputs.request.manifest_family.family_id,
                        ),
                        ("restore operation count", len(plan.restore_operations)),
                        (
                            "required splits",
                            tuple(
                                split_input.split.value
                                for split_input in plan.execution_inputs.split_inputs
                            ),
                        ),
                        ("model config path", plan.execution_inputs.model_config_path),
                        (
                            "semantic objective config snapshot path",
                            plan.execution_inputs.semantic_objective_config_path,
                        ),
                    ),
                ),
                *tuple(_operation_item(operation) for operation in plan.restore_operations),
            ),
        ),
    )


def review_smoke_execution_protocol(
    protocol: ModelSmokeExecutionProtocol,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Model smoke protocol summary",
            (
                review_item(
                    protocol.run_name,
                    (
                        ("ready for smoke execution", protocol.ready_for_smoke_execution),
                        ("smoke mode", protocol.smoke_mode),
                        ("step count", len(protocol.steps)),
                        (
                            "recommended checkpoint policy",
                            protocol.handoff.recommended_checkpoint_policy,
                        ),
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
                        ("preflight gate required before restore/training/inference", True),
                        (
                            "failed preflight stops before restore/training/inference",
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
            "Handoff values for test_model.ipynb",
            (
                *tuple(
                    review_item(key, (("value", to_review_value(value)),))
                    for key, value in protocol.handoff.paste_into_test_model.items()
                ),
                review_item(
                    "TARGET_SENTENCE_NAME source",
                    (
                        ("guidance", protocol.handoff.target_sentence_guidance),
                        ("Drive test manifest", protocol.handoff.test_manifest_path),
                    ),
                ),
            ),
        ),
        review_section(
            "Model smoke warnings",
            tuple(review_item(warning) for warning in protocol.warnings),
        ),
        review_section(
            "Model smoke limitations",
            tuple(review_item(limitation) for limitation in protocol.limitations),
        ),
    )


def review_runtime_restore(
    result: ModelRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime restore summary",
            (
                review_item(
                    "restore",
                    (
                        ("operation count", len(result.execution.results)),
                        ("successful operation count", len(result.execution.successful_results())),
                        ("failed operation count", len(result.execution.failed_results())),
                        ("succeeded", result.execution.succeeded),
                    ),
                ),
            ),
        ),
    )


def review_runtime_verification(
    verification: ModelRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime verification summary",
            (
                review_item(
                    "readiness",
                    (
                        ("level", verification.readiness_level.value),
                        ("checked semantics", verification.checked_semantics),
                        ("limitations", verification.limitations),
                        ("failed check count", len(verification.failed_checks)),
                        ("succeeded", verification.succeeded),
                    ),
                ),
                *(
                    review_item(
                        f"failed: {check.label}",
                        (("path", check.path), ("message", check.message)),
                    )
                    for check in verification.failed_checks
                ),
            ),
        ),
    )


def review_research_resolution(
    resolution: ModelResearchResolution,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Research resolution",
            (
                review_item(
                    resolution.model_spec.display_name,
                    (
                        ("model key", resolution.model_spec.key.value),
                        ("canonical id", resolution.model_spec.canonical_id),
                        (
                            "objectives",
                            tuple(objective.key.value for objective in resolution.objective_specs),
                        ),
                        ("trace issue count", len(resolution.trace_issues)),
                    ),
                ),
            ),
        ),
    )


def review_provider_resolution(
    resolution: ModelProviderResolution,
) -> tuple[WorkflowReviewSection, ...]:
    message = resolution.message
    if not resolution.provider_available:
        message = (
            "No provider is registered for this model key yet. "
            "This is expected before the model candidate implementation stage."
        )
    return (
        review_section(
            "Provider resolution",
            (
                review_item(
                    resolution.provider_key.value,
                    (
                        ("provider available", resolution.provider_available),
                        ("message", message),
                    ),
                ),
            ),
        ),
    )


def review_provider_config(
    result: ModelProviderConfigResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Provider config",
            (
                review_item(
                    result.provider.spec.key.value,
                    (
                        ("source path", result.loaded_config.source_path),
                        ("effective key count", len(result.loaded_config.effective_config)),
                    ),
                ),
            ),
        ),
    )


def review_provider_effective_config(
    result: ModelProviderConfigResult,
) -> tuple[WorkflowReviewSection, ...]:
    effective = result.loaded_config.effective_config
    provider_key = result.provider.spec.key.value
    compute_profile = _first(effective, ("compute_profile",)) or {}
    provider_overrides = (
        _first(compute_profile, ("provider_overrides", provider_key))
        if isinstance(compute_profile, Mapping)
        else None
    )
    active_fields: list[tuple[str, RenderableValue]] = [
        ("model_key", provider_key),
        ("model config path", result.loaded_config.source_path),
        ("run_mode", to_review_value(_first(effective, ("run_mode",), ("run_mode_policy", "run_mode")))),
        (
            "manifest_family",
            to_review_value(_first(effective, ("manifest_family",), ("data", "manifest_family"))),
        ),
        ("train split", to_review_value(_first(effective, ("train_split",), ("data", "train_split")))),
        (
            "validation split",
            to_review_value(_first(effective, ("validation_split",), ("data", "validation_split"))),
        ),
        ("test split", to_review_value(_first(effective, ("test_split",), ("data", "test_split")))),
        ("limit_train_samples", to_review_value(_limit(effective, "limit_train_samples"))),
        (
            "limit_validation_samples",
            to_review_value(_limit(effective, "limit_validation_samples")),
        ),
        (
            "limit_prediction_samples",
            to_review_value(_limit(effective, "limit_prediction_samples")),
        ),
        ("max_epochs", to_review_value(_max_epochs(effective))),
        ("batch_size", to_review_value(_batch_size(effective))),
        *_active_provider_performance_review_fields(provider_key, effective),
        ("frame_batch_size", to_review_value(_frame_batch_size(effective))),
        ("compute_profile", to_review_value(_first(compute_profile, ("name",)))),
        ("precision policy", to_review_value(_first(compute_profile, ("precision", "policy")))),
        ("allow_tf32", to_review_value(_first(compute_profile, ("torch", "allow_tf32")))),
        (
            "float32_matmul_precision",
            to_review_value(_first(compute_profile, ("torch", "float32_matmul_precision"))),
        ),
        ("num_workers", to_review_value(_first(compute_profile, ("dataloader", "num_workers")))),
        ("pin_memory", to_review_value(_first(compute_profile, ("dataloader", "pin_memory")))),
        (
            "persistent_workers",
            to_review_value(_first(compute_profile, ("dataloader", "persistent_workers"))),
        ),
        (
            "prefetch_factor",
            to_review_value(_first(compute_profile, ("dataloader", "prefetch_factor"))),
        ),
        *_sampling_review_fields(effective, provider_overrides),
    ]
    return (
        review_section(
            "Provider effective run config",
            (
                review_item(
                    provider_key,
                    (*active_fields, *_provider_override_review_fields(provider_overrides)),
                ),
                review_item(
                    "provider-specific",
                    tuple(
                        (label, to_review_value(value))
                        for label, value in _provider_specific_items(provider_key, effective)
                    ),
                ),
            ),
        ),
    )


def review_stage_plan(
    result: ModelStagePlanningResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Stage plan",
            (
                review_item(
                    result.stage_plan.model.key.value,
                    (
                        ("stage count", len(result.stage_plan.stages)),
                        (
                            "stages",
                            tuple(stage.spec.kind.value for stage in result.stage_plan.stages),
                        ),
                        (
                            "objective attachments",
                            tuple(
                                attachment.objective.key.value
                                for attachment in result.stage_plan.objective_attachments
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def review_stage_execution(
    result: ModelStageExecutionWorkflowResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Stage execution",
            (
                review_item(
                    result.execution.run_name,
                    (
                        ("stage result count", len(result.execution.stages)),
                        ("completed", result.execution.completed),
                        (
                            "statuses",
                            tuple(stage.status.value for stage in result.execution.stages),
                        ),
                        ("artifact count", len(result.execution.artifact_refs)),
                    ),
                ),
            ),
        ),
    )


def review_stage_artifact_receipts(
    result: ModelStageArtifactReceiptResult,
) -> tuple[WorkflowReviewSection, ...]:
    generated_sample_skips = tuple(
        skipped
        for skipped in result.skipped_artifacts
        if skipped.kind == "generated_pose_sample"
    )
    return (
        review_section(
            "Stage artifact receipts",
            (
                review_item(
                    result.execution_id,
                    (
                        ("receipt count", len(result.receipts)),
                        ("skipped count", len(result.skipped_artifacts)),
                        ("generated sample skip count", len(generated_sample_skips)),
                        ("index path", result.index_path),
                        ("index sha256", result.index_receipt.sha256),
                    ),
                ),
                *(
                    review_item(
                        f"skipped: {skipped.role}",
                        (
                            ("kind", skipped.kind),
                            ("path", skipped.path),
                            ("reason", skipped.reason),
                        ),
                    )
                    for skipped in result.skipped_artifacts[:10]
                ),
            ),
        ),
    )


def review_stage_artifact_receipts_detail(
    result: ModelStageArtifactReceiptResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Stage artifact receipt details",
            tuple(
                review_item(
                    receipt.label,
                    (
                        ("kind", receipt.kind),
                        ("path", receipt.path),
                        ("sha256", receipt.sha256),
                    ),
                )
                for receipt in result.receipts
            ),
        ),
        review_section(
            "Stage artifact skipped details",
            tuple(
                review_item(
                    skipped.role,
                    (
                        ("kind", skipped.kind),
                        ("path", skipped.path),
                        ("reason", skipped.reason),
                    ),
                )
                for skipped in result.skipped_artifacts
            ),
        ),
    )


def review_model_run_metadata_artifacts(
    artifacts: ModelRunMetadataArtifacts,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Model run metadata artifacts",
            tuple(
                review_item(
                    receipt.label,
                    (
                        ("kind", receipt.kind),
                        ("path", receipt.path),
                        ("sha256", receipt.sha256),
                    ),
                )
                for receipt in (
                    artifacts.effective_config,
                    artifacts.research_spec,
                    artifacts.run_metadata,
                    artifacts.runtime_support_manifest,
                )
            ),
        ),
    )


def review_model_reports(
    artifacts: ModelReportArtifacts,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Model report artifacts",
            (
                review_item(
                    "reports",
                    (
                        ("receipt count", len(artifacts.artifacts)),
                        ("provider artifact count", len(artifacts.provider_artifacts)),
                        ("execution id", artifacts.execution_id),
                    ),
                ),
                *(
                    review_item(
                        receipt.label,
                        (("kind", receipt.kind), ("path", receipt.path)),
                    )
                    for receipt in artifacts.artifacts
                ),
            ),
        ),
    )


def review_model_validation_artifacts(
    artifacts: ModelValidationArtifactResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Model validation artifacts",
            (
                review_item(
                    "validation split comparison",
                    (
                        ("paired count", artifacts.paired_count),
                        ("missing generated count", artifacts.missing_generated_count),
                        ("missing reference count", artifacts.missing_reference_count),
                        ("failed generated count", artifacts.failed_generated_count),
                        ("identity mismatch count", artifacts.identity_mismatch_count),
                        ("summary path", artifacts.summary_markdown_path),
                        ("limitations path", artifacts.limitations_path),
                    ),
                ),
            ),
        ),
    )


def review_model_objective_artifacts(
    artifacts: tuple[ModelObjectiveArtifactResult, ...],
) -> tuple[WorkflowReviewSection, ...]:
    if not artifacts:
        return (
            review_section(
                "Model objective artifacts",
                (review_item("auxiliary objectives", (("attached", ()),)),),
            ),
        )
    return (
        review_section(
            "Model objective artifacts",
            tuple(
                review_item(
                    artifact.objective_key.value,
                    (
                        ("objective attached", True),
                        ("post-generation proxy evaluation only", True),
                        ("ablation required", True),
                        ("semantic correctness claimed", False),
                        ("config snapshot", artifact.config_snapshot_path),
                        ("config snapshot sha256", artifact.config_snapshot_sha256),
                        ("final validation generated manifest", artifact.generated_pose_manifest_path),
                        ("candidate policy", artifact.candidate_policy),
                        ("ablation readiness", artifact.ablation_readiness_path),
                        ("ready for comparison", artifact.ready_for_comparison),
                        ("required baseline missing", artifact.required_baseline_missing),
                        ("readiness issues", artifact.readiness_issues),
                        ("completed ablation claimed", False),
                        ("records count", artifact.records_count),
                        ("skipped count", artifact.skipped_count),
                        ("alignment aggregate", artifact.artifact_paths["alignment_aggregate"]),
                    ),
                )
                for artifact in artifacts
            ),
        ),
    )


def review_final_operator_summary(
    result: ModelWorkflowFinalResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Final model workflow summary",
            (
                review_item(
                    result.request.run_name,
                    (
                        ("model key", result.request.model_key.value),
                        ("manifest family", result.request.manifest_family.family_id),
                        ("provider available", result.provider_available),
                        ("runtime verified", result.runtime_verification is not None),
                        ("stage execution available", result.stage_execution is not None),
                        ("metadata written", result.metadata_artifacts is not None),
                        ("validation written", result.validation_artifacts is not None),
                        (
                            "auxiliary objectives",
                            tuple(objective.value for objective in result.request.auxiliary_objectives),
                        ),
                        ("objective artifacts written", bool(result.objective_artifacts)),
                        ("reports written", result.report_artifacts is not None),
                        ("publish available", result.publish_result is not None),
                        ("succeeded", result.succeeded),
                        ("limitations", result.limitations),
                    ),
                ),
            ),
        ),
    )


def _operation_item(operation) -> object:
    if isinstance(operation, FileCopyOperation):
        return review_item(
            operation.label,
            (
                ("operation kind", operation_kind(operation)),
                ("source path", operation.source_path),
                ("target path", operation.target_path),
            ),
        )
    if isinstance(operation, ArchiveExtractOperation):
        return review_item(
            operation.label,
            (
                ("operation kind", operation_kind(operation)),
                ("archive path", operation.archive_path),
                ("extraction root", operation.extraction_root),
            ),
        )
    return review_item(str(operation_kind(operation)))


def _first(mapping, *paths):
    for path in paths:
        current = mapping
        for key in path:
            if not isinstance(current, dict | Mapping):
                current = None
                break
            current = current.get(key)
        if current is not None:
            return current
    return None


def _limit(effective, key: str):
    return _first(effective, ("data", key), ("run_mode_overrides", key), ("run_mode_policy", key), (key,))


def _max_epochs(effective):
    return _first(effective, ("training", "max_epochs"), ("run_mode_overrides", "max_epochs"), ("run_mode_policy", "max_epochs"))


def _batch_size(effective):
    return _first(effective, ("training", "batch_size"), ("run_mode_overrides", "batch_size"), ("run_mode_policy", "batch_size"))


def _frame_batch_size(effective):
    return _first(
        effective,
        ("training", "frame_batch_size"),
        ("run_mode_overrides", "frame_batch_size"),
        ("run_mode_policy", "frame_batch_size"),
    )


def _sampling_review_fields(
    effective,
    provider_overrides: object,
) -> tuple[tuple[str, RenderableValue], ...]:
    configured = _first(
        effective,
        ("diffusion", "sampling_steps"),
        ("generation", "sampling_steps"),
        ("sampling_steps",),
    )
    effective_steps = configured
    source = "config" if configured is not None else None
    run_mode = _first(effective, ("run_mode",), ("run_mode_policy", "run_mode"))
    if isinstance(provider_overrides, Mapping):
        override_key = None
        override_source = None
        if run_mode == "smoke":
            override_key = "smoke_sampling_steps"
            override_source = "compute_profile_smoke_override"
        elif run_mode == "debug":
            override_key = "debug_sampling_steps"
            override_source = "compute_profile_debug_override"
        if override_key is not None:
            override_value = provider_overrides.get(override_key)
            if override_value is not None:
                effective_steps = override_value
                source = override_source
    if configured is None and effective_steps is None and source is None:
        return ()
    return (
        ("configured_sampling_steps", to_review_value(configured)),
        ("effective_sampling_steps", to_review_value(effective_steps)),
        ("sampling_steps_source", to_review_value(source)),
    )


def _provider_override_review_fields(
    provider_overrides: object,
) -> tuple[tuple[str, RenderableValue], ...]:
    if not isinstance(provider_overrides, Mapping) or not provider_overrides:
        return (("provider compute override count", 0),)
    fields: list[tuple[str, RenderableValue]] = [
        ("provider compute override count", len(provider_overrides))
    ]
    active = provider_overrides.get("active")
    candidates = provider_overrides.get("candidates")
    run_modes = ("smoke", "debug", "full")
    if isinstance(active, Mapping):
        for run_mode in run_modes:
            selected = active.get(run_mode)
            if not isinstance(selected, Mapping):
                continue
            for key, value in sorted(selected.items(), key=lambda item: str(item[0])):
                fields.append((f"active {run_mode} {str(key)}", to_review_value(value)))
    if isinstance(candidates, Mapping):
        for key, value in sorted(candidates.items(), key=lambda item: str(item[0])):
            display = "; ".join(str(item) for item in value) if isinstance(value, list | tuple) else value
            fields.append((f"candidate {str(key)}", to_review_value(display)))
    for key, value in sorted(provider_overrides.items(), key=lambda item: str(item[0])):
        if key in {"active", "candidates"}:
            continue
        fields.append((f"provider override {str(key)}", to_review_value(value)))
    return tuple(fields)


def _active_provider_performance_review_fields(
    provider_key: str,
    effective,
) -> tuple[tuple[str, RenderableValue], ...]:
    del provider_key
    if not isinstance(effective, Mapping):
        return ()
    application = effective.get("compute_profile_application")
    if not isinstance(application, Mapping):
        return ()
    fields: list[tuple[str, RenderableValue]] = []
    for section_name, label in (
        ("active_applied", "active"),
        ("dataloader_applied", "dataloader"),
    ):
        section = application.get(section_name)
        if not isinstance(section, Mapping):
            continue
        fields.extend(
            (f"{label}.{str(key)}", to_review_value(value))
            for key, value in sorted(section.items(), key=lambda item: str(item[0]))
        )
    return tuple(fields)


def _provider_specific_items(provider_key: str, effective) -> tuple[tuple[str, object], ...]:
    text_encoder = _first(effective, ("text_encoder", "backend"), ("text_encoder", "encoder_key"))
    if provider_key == "learned_pose_token":
        return (
            ("codebook size", _first(effective, ("codebook", "size"))),
            ("temporal granularity", _first(effective, ("tokenizer", "temporal_granularity"))),
            ("window size", _first(effective, ("tokenizer", "window_size"))),
            ("stride", _first(effective, ("tokenizer", "stride"))),
            ("text encoder kind", text_encoder),
        )
    if provider_key == "latent_diffusion":
        return (
            ("latent target type", _first(effective, ("latent_target", "target_type"))),
            ("candidate_count", _first(effective, ("generation", "candidate_count"))),
            ("denoiser epochs", _max_epochs(effective)),
            ("denoiser batch size", _batch_size(effective)),
            ("temporal latent autoencoder enabled", _first(effective, ("latent_autoencoder",)) is not None),
        )
    if provider_key == "articulator_aware":
        return (
            ("architecture variant", _first(effective, ("structure_variant", "architecture"))),
            ("channel partition policy", _first(effective, ("partition_policy", "source"))),
            ("mask strategy", _first(effective, ("mask_strategy", "missing_channel_policy"))),
            ("loss weighting policy", _first(effective, ("loss_weighting", "policy"))),
            ("text encoder kind", text_encoder),
        )
    return (("provider-specific values", None),)


__all__ = [
    "review_final_operator_summary",
    "review_preflight",
    "review_model_reports",
    "review_model_validation_artifacts",
    "review_model_objective_artifacts",
    "review_model_run_metadata_artifacts",
    "review_provider_config",
    "review_provider_effective_config",
    "review_provider_resolution",
    "review_research_resolution",
    "review_runtime_plan",
    "review_runtime_restore",
    "review_runtime_verification",
    "review_smoke_execution_protocol",
    "review_stage_artifact_receipts",
    "review_stage_artifact_receipts_detail",
    "review_stage_execution",
    "review_stage_plan",
]
