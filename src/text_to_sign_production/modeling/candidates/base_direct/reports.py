"""Provider-owned operator reports for the M0 direct baseline."""

from __future__ import annotations

from text_to_sign_production.core.io import write_markdown
from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelStageArtifactRef,
    ModelStageExecutionContext,
    ModelStageKind,
    resolve_model_run_mode_policy,
)
from text_to_sign_production.modeling.candidates.base_direct.spec import require_base_direct_spec


def write_base_direct_reports(
    *,
    context: ModelStageExecutionContext,
    results: ModelExecutionResult,
) -> tuple[ModelStageArtifactRef, ...]:
    """Write base-direct reports through canonical modeling report topology."""

    topology = context.topology.reports
    run_name = context.request.run_name
    spec = require_base_direct_spec()
    paths = {
        "model_run_summary": topology.model_report_file(
            "base_direct",
            run_name,
            "model_run_summary.md",
        ).path,
        "model_spec": topology.model_spec_report("base_direct", run_name).path,
        "architecture": topology.model_architecture_report("base_direct", run_name).path,
        "training_summary": topology.model_training_summary_report("base_direct", run_name).path,
        "validation_generation_summary": topology.model_report_file(
            "base_direct",
            run_name,
            "validation_generation_summary.md",
        ).path,
        "checkpoint_selection": topology.model_report_file(
            "base_direct",
            run_name,
            "checkpoint_selection.md",
        ).path,
        "limitations": topology.model_report_file(
            "base_direct",
            run_name,
            "limitations.md",
        ).path,
        "generated_pose_summary": topology.model_generated_pose_summary_report(
            "base_direct",
            run_name,
        ).path,
        "baseline_failure_modes": topology.model_report_file(
            "base_direct",
            run_name,
            "baseline_failure_modes.md",
        ).path,
        "risk_controls": topology.model_risk_controls_report("base_direct", run_name).path,
    }
    training_metadata = _metadata_for_kind(results, ModelStageKind.TRAIN)
    export_metadata = _metadata_for_kind(results, ModelStageKind.EXPORT_GENERATED_POSE)
    write_markdown(
        paths["model_run_summary"],
        _model_run_summary(context, training_metadata, export_metadata),
    )
    write_markdown(
        paths["model_spec"],
        (
            "# M0 Direct Text-to-Pose Baseline\n\n"
            f"- Model key: `{spec.key.value}`\n"
            f"- Canonical id: `{spec.canonical_id}`\n"
            f"- Research role: `{spec.research_role.value}`\n"
            f"- Phase: `{spec.phase_number}`\n"
            f"- Output contract: `{spec.output_contract}`\n"
        ),
    )
    write_markdown(
        paths["architecture"],
        (
            "# Architecture\n\n"
            "This baseline directly maps text or transcript input to continuous pose. "
            "A text encoder produces a pooled representation, a temporal decoder expands it "
            "to the reference sequence length, and full BFH regression heads produce body, "
            "left hand, right hand, and face XY coordinates. Export attaches synthetic "
            "validity confidence to create the Stage 3 XYC payload.\n\n"
            "## Absent Mechanisms\n\n"
            "- No learned pose token.\n"
            "- No diffusion.\n"
            "- No articulator-aware mechanism.\n"
            "- No semantic objective.\n"
            "- No retrieval.\n"
            "- No gloss supervision.\n"
            "- No avatar or rendering.\n"
        ),
    )
    write_markdown(
        paths["training_summary"],
        _training_summary(context, training_metadata),
    )
    write_markdown(
        paths["validation_generation_summary"],
        _validation_summary(context, export_metadata),
    )
    write_markdown(
        paths["checkpoint_selection"],
        _checkpoint_selection_summary(training_metadata),
    )
    write_markdown(
        paths["limitations"],
        _limitations_summary(),
    )
    write_markdown(
        paths["generated_pose_summary"],
        _generated_pose_summary(context, export_metadata),
    )
    write_markdown(
        paths["baseline_failure_modes"],
        (
            "# Baseline Failure Modes\n\n"
            "- Direct regression oversmoothing can average distinct motions.\n"
            "- Weak temporal structure can degrade movement coherence.\n"
            "- Hand and face detail may be lost in simple regression.\n"
            "- Semantic drift may occur between transcript and produced motion.\n"
            "- Reference-length dependency does not solve timing prediction.\n"
            "- Synthetic confidence is not uncertainty.\n"
        ),
    )
    write_markdown(
        paths["risk_controls"],
        (
            "# Risk Controls\n\n"
            "- Baseline readiness is not evidence of contribution strength.\n"
            "- Baseline readiness is not evidence of strong task-solving quality.\n"
            "- Automatic metrics are not proof of sign intelligibility.\n"
            "- Visual plausibility is not proof of semantic correctness.\n"
            "- Synthetic confidence is not model uncertainty.\n"
        ),
    )
    return tuple(
        ModelStageArtifactRef(
            role=f"base_direct_{role}_report",
            path=path,
            kind="model_report",
        )
        for role, path in paths.items()
    )


def _metadata_for_kind(
    results: ModelExecutionResult,
    kind: ModelStageKind,
) -> dict[str, object]:
    for result in results.stages:
        if result.stage.spec.kind is kind:
            return dict(result.metadata)
    return {}


def _training_summary(
    context: ModelStageExecutionContext,
    metadata: dict[str, object],
) -> str:
    request = context.request
    return (
        "# Training Summary\n\n"
        f"- Run name: `{request.run_name}`\n"
        f"- Run mode: `{request.run_mode.value}`\n"
        f"- Train split: `{request.train_split.value}`\n"
        f"- Validation split: `{request.validation_split.value}`\n"
        f"- Train samples: `{metadata.get('train_sample_count', 'not recorded')}`\n"
        f"- Validation samples: `{metadata.get('validation_sample_count', 'not recorded')}`\n"
        f"- Completed epoch: `{metadata.get('completed_epoch', 'not recorded')}`\n"
        f"- Last checkpoint: `{metadata.get('last_checkpoint_path', 'not recorded')}`\n"
        f"- Best checkpoint: `{metadata.get('best_checkpoint_path', 'not recorded')}`\n"
        f"- Final train loss: `{metadata.get('final_train_loss', 'not recorded')}`\n"
        f"- Final validation loss: `{metadata.get('final_validation_loss', 'not recorded')}`\n"
        f"- Best metric value: `{metadata.get('best_metric_value', 'not recorded')}`\n"
    )


def _model_run_summary(
    context: ModelStageExecutionContext,
    training_metadata: dict[str, object],
    export_metadata: dict[str, object],
) -> str:
    request = context.request
    return (
        "# Model Run Summary\n\n"
        f"- Model key: `{request.model_key.value}`\n"
        f"- Run name: `{request.run_name}`\n"
        f"- Manifest family: `{request.manifest_family.family_id}`\n"
        f"- Manifest kind: `{request.manifest_family.kind}`\n"
        f"- Manifest tier: `{request.manifest_family.tier}`\n"
        f"- Manifest membership: `{request.manifest_family.membership}`\n"
        f"- Run mode: `{request.run_mode.value}`\n"
        f"- Run-mode quality claim: `{resolve_model_run_mode_policy(request.run_mode).quality_claim}`\n"
        f"- Train split: `{request.train_split.value}`\n"
        f"- Validation split: `{request.validation_split.value}`\n"
        "- Test split: `test` (reserved for `test_model.ipynb`)\n"
        f"- Prediction splits: `{[split.value for split in request.prediction_splits]}`\n"
        f"- Last checkpoint: `{training_metadata.get('last_checkpoint_path', 'not recorded')}`\n"
        f"- Best checkpoint: `{training_metadata.get('best_checkpoint_path', 'not recorded')}`\n"
        f"- Validation generated-pose manifests: `{export_metadata.get('manifest_paths', {})}`\n\n"
        "This run trains on the train split and writes validation split outputs for "
        "model development and checkpoint selection. The direct baseline is a comparison "
        "floor. It does not process the test split.\n"
    )


def _generated_pose_summary(
    context: ModelStageExecutionContext,
    metadata: dict[str, object],
) -> str:
    split_counts = metadata.get("split_sample_counts", {})
    manifests = metadata.get("manifest_paths", {})
    archives = metadata.get("generated_pose_archives", {})
    return (
        "# Generated Pose Summary\n\n"
        f"- Prediction splits: `{[split.value for split in context.request.prediction_splits]}`\n"
        f"- Generated sample counts: `{split_counts}`\n"
        f"- Manifest paths: `{manifests}`\n"
        f"- Sample archives: `{archives}`\n"
        "- Individual sample publish: `False` when a split archive is present\n"
        "- Channel policy: `full_bfh`\n"
        "- Length policy: `reference_length`\n"
        "- Confidence policy: `synthetic_validity`\n\n"
        "Synthetic validity confidence marks usable reference frames; it is not a measure "
        "of model uncertainty or semantic correctness.\n"
    )


def _validation_summary(
    context: ModelStageExecutionContext,
    metadata: dict[str, object],
) -> str:
    split_counts = metadata.get("split_sample_counts", {})
    manifests = metadata.get("manifest_paths", {})
    archives = metadata.get("generated_pose_archives", {})
    return (
        "# Validation Generation Summary\n\n"
        f"- Validation split: `{context.request.validation_split.value}`\n"
        f"- Prediction splits: `{[split.value for split in context.request.prediction_splits]}`\n"
        f"- Generated validation sample counts: `{split_counts}`\n"
        f"- Validation manifest paths: `{manifests}`\n"
        f"- Validation sample archives: `{archives}`\n\n"
        "These outputs are validation split diagnostics for model development and "
        "checkpoint selection. Workflow-owned validation metrics do not imply contribution "
        "strength, and the validation split is not the test split.\n"
    )


def _checkpoint_selection_summary(metadata: dict[str, object]) -> str:
    return (
        "# Checkpoint Selection\n\n"
        f"- Selection metric: `{metadata.get('best_metric_name', 'not recorded')}`\n"
        f"- Selection metric value: `{metadata.get('best_metric_value', 'not recorded')}`\n"
        f"- Best checkpoint: `{metadata.get('best_checkpoint_path', 'not recorded')}`\n"
        f"- Last checkpoint: `{metadata.get('last_checkpoint_path', 'not recorded')}`\n\n"
        "The best checkpoint is selected from validation-stage training feedback when "
        "available. There is no silent fallback in `test_model`; operators choose best or last.\n"
    )


def _limitations_summary() -> str:
    return (
        "# Limitations\n\n"
        "- Validation outputs are not full test split evaluation.\n"
        "- The direct baseline is a comparison floor, not evidence of contribution strength.\n"
        "- Validation metrics do not imply contribution strength.\n"
        "- The validation split is not the test split.\n"
        "- Validation outputs are not aggregate model performance.\n"
        "- Generated pose quality requires separate sample-level review and later evaluation.\n"
        "- Synthetic validity confidence is not model uncertainty.\n"
        "- This baseline does not implement retrieval, semantic objectives, diffusion, "
        "learned tokens, or articulator-specific mechanisms.\n"
    )


__all__ = ["write_base_direct_reports"]
