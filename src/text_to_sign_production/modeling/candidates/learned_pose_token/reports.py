"""Honest report builders for the learned pose-token foundation."""

from __future__ import annotations

from collections.abc import Sequence

from text_to_sign_production.core.io import write_markdown
from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelStageArtifactRef,
    ModelStageExecutionContext,
    ModelStageKind,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.codebook import (
    PoseCodebookStabilityReport,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    LearnedPoseTokenConfig,
    learned_pose_token_config_from_effective_dict,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.tokens import (
    PoseTokenManifestEntry,
)

_BASE_LIMITATIONS = (
    "Token reconstruction quality is not semantic adequacy.",
    "Codebook usage stability does not prove sign intelligibility.",
    "No diffusion mechanism is included.",
    "Semantic consistency, when attached, is post-generation proxy evaluation only.",
    "No retrieval/stitching is included.",
    "No gloss supervision is assumed.",
)


def build_tokenizer_design_report(config: LearnedPoseTokenConfig) -> dict[str, object]:
    """Build a JSON-compatible tokenizer design report."""

    base_frame_feature_dim = 0
    token_unit_feature_dim = 0
    try:
        from text_to_sign_production.modeling.backbones.bfh_vectorization import (
            default_bfh_tensor_layout,
        )

        base_frame_feature_dim = default_bfh_tensor_layout().total_feature_dim
        token_unit_feature_dim = base_frame_feature_dim * config.tokenizer.window_size
    except Exception:
        pass
    return {
        "schema_version": "t2sp-learned-pose-tokenizer-design-report-v1",
        "model_key": config.identity.model_key,
        "phase_number": config.identity.phase_number,
        "research_role": config.identity.research_role,
        "temporal_granularity": config.tokenizer.temporal_granularity,
        "window_size": config.tokenizer.window_size,
        "stride": config.tokenizer.stride,
        "base_frame_feature_dim": base_frame_feature_dim,
        "token_unit_feature_dim": token_unit_feature_dim,
        "latent_dim": config.tokenizer.latent_dim,
        "codebook_size": config.codebook.size,
        "boundaries": {
            "diffusion": False,
            "semantic_auxiliary_objective": False,
            "retrieval_or_stitching": False,
            "avatar_or_articulator_aware_modeling": False,
            "gloss_supervision": False,
            "text_to_token_predictor": False,
            "provider_registration": False,
        },
        "limitations": list(_limitations_for_config(config)),
    }


def build_codebook_stability_report_dict(
    report: PoseCodebookStabilityReport,
) -> dict[str, object]:
    """Return codebook stability report JSON with explicit limitation language."""

    return report.to_dict() | {
        "interpretation": (
            "Codebook stability summarizes token usage only; it is not evidence of sign "
            "intelligibility or semantic adequacy."
        )
    }


def build_reconstruction_report_markdown(
    *,
    run_name: str,
    split: str,
    sample_count: int,
    mean_reconstruction_loss: float | None,
    temporal_granularity: str = "frame",
    window_size: int = 1,
    stride: int = 1,
) -> str:
    """Build a reconstruction report that avoids semantic quality claims."""

    loss_text = "not recorded" if mean_reconstruction_loss is None else str(mean_reconstruction_loss)
    limitations = "\n".join(
        f"- {line}"
        for line in _limitations_for_mode(temporal_granularity, window_size, stride)
    )
    return (
        "# Learned Pose-Token Reconstruction Report\n\n"
        f"- Run name: `{run_name}`\n"
        f"- Split: `{split}`\n"
        f"- Reconstruction samples: `{sample_count}`\n"
        f"- Mean reconstruction loss: `{loss_text}`\n\n"
        f"- Temporal granularity: `{temporal_granularity}`\n"
        f"- Window size: `{window_size}`\n"
        f"- Stride: `{stride}`\n\n"
        "This report covers pose -> learned token/code -> reconstructed pose only. "
        "It does not evaluate text-to-pose generation.\n\n"
        "## Limitations\n\n"
        f"{limitations}\n"
    )


def build_token_manifest_summary(
    entries: Sequence[PoseTokenManifestEntry],
) -> dict[str, object]:
    """Summarize token manifests without claiming downstream readiness."""

    materialized = tuple(entries)
    split_counts: dict[str, int] = {}
    token_total = 0
    frame_total = 0
    temporal_modes = set()
    window_sizes = set()
    strides = set()
    for entry in materialized:
        split_counts[entry.split.value] = split_counts.get(entry.split.value, 0) + 1
        token_total += entry.token_count
        frame_total += entry.frame_count
        temporal_modes.add(entry.temporal_granularity)
        window_sizes.add(entry.window_size)
        strides.add(entry.stride)
    return {
        "schema_version": "t2sp-learned-pose-token-manifest-summary-v1",
        "sequence_count": len(materialized),
        "split_counts": split_counts,
        "token_count": token_total,
        "source_frame_count": frame_total,
        "temporal_granularity": sorted(temporal_modes),
        "window_size": sorted(window_sizes),
        "stride": sorted(strides),
        "limitations": list(_limitations_for_mode(
            next(iter(temporal_modes), "frame"),
            next(iter(window_sizes), 1),
            next(iter(strides), 1),
        )),
    }


def write_learned_pose_token_reports(
    *,
    context: ModelStageExecutionContext,
    results: ModelExecutionResult,
) -> tuple[ModelStageArtifactRef, ...]:
    """Write provider-owned Phase 6 reports."""

    report_root = context.topology.reports.model_report_file(
        "learned_pose_token",
        context.request.run_name,
        "placeholder.md",
    ).path.parent
    report_root.mkdir(parents=True, exist_ok=True)
    generation_metadata = _metadata_for_kind(results, ModelStageKind.EXPORT_GENERATED_POSE)
    text_metadata = _metadata_for_kind(results, ModelStageKind.TRAIN_TEXT_TO_TOKEN)
    representation_metadata = _metadata_for_kind(results, ModelStageKind.FIT_REPRESENTATION)
    semantic_lines = _semantic_objective_lines(context)
    config = learned_pose_token_config_from_effective_dict(
        context.loaded_config.effective_config,
        source_path=context.loaded_config.source_path,
    )
    paths = {
        "learned_pose_token_run_summary": report_root / "learned_pose_token_run_summary.md",
        "representation_report": report_root / "representation_report.md",
        "reconstruction_report": report_root / "reconstruction_report.md",
        "text_to_token_report": report_root / "text_to_token_report.md",
        "generation_report": report_root / "generation_report.md",
        "limitations": report_root / "limitations.md",
    }
    write_markdown(
        paths["learned_pose_token_run_summary"],
        (
            "# Learned Pose-Token Run Summary\n\n"
            f"- Run name: `{context.request.run_name}`\n"
            f"- Manifest family: `{context.request.manifest_family.family_id}`\n"
            f"- Run mode: `{context.request.run_mode.value}`\n"
            f"- Temporal granularity: `{config.tokenizer.temporal_granularity}`\n"
            f"- Window size: `{config.tokenizer.window_size}`\n"
            f"- Stride: `{config.tokenizer.stride}`\n"
            f"- Best checkpoint: `{text_metadata.get('best_checkpoint_path', 'not recorded')}`\n"
            f"- Generated validation manifests: `{generation_metadata.get('manifest_paths', {})}`\n\n"
            f"- Standardization missing-observation policy: "
            f"`{representation_metadata.get('standardization_missing_observation_policy', 'not recorded')}`\n"
            f"- Zero-observation coordinates: "
            f"`{representation_metadata.get('standardization_zero_observation_coordinate_count', 'not recorded')}`\n"
            f"- Channel fallback coordinates: "
            f"`{representation_metadata.get('standardization_channel_fallback_coordinate_count', 'not recorded')}`\n\n"
            "Standardization fallback diagnostics indicate sparse observations; "
            "smoke fallback is not a model quality claim.\n\n"
            "The learned pose-token bottleneck is the provider generation mechanism.\n"
            + "\n".join(f"- {line}" for line in semantic_lines)
            + "\n"
        ),
    )
    write_markdown(
        paths["representation_report"],
        (
            "# Representation Report\n\n"
            "This stage trains a pose -> learned token/code -> reconstructed pose representation. "
            "Codebook stability does not prove sign intelligibility.\n"
            f"- Temporal granularity: `{config.tokenizer.temporal_granularity}`\n"
            f"- Window size: `{config.tokenizer.window_size}`\n"
            f"- Stride: `{config.tokenizer.stride}`\n"
            f"- Standardization missing-observation policy: "
            f"`{representation_metadata.get('standardization_missing_observation_policy', 'not recorded')}`\n"
            f"- Fallback summary by channel: "
            f"`{representation_metadata.get('standardization_fallback_summary_by_channel', {})}`\n"
            "Sparse fallback diagnostics are for smoke/debug robustness, not model quality.\n"
        ),
    )
    write_markdown(
        paths["reconstruction_report"],
        (
            "# Reconstruction Report\n\n"
            "Reconstruction quality is not semantic adequacy. Reconstruction artifacts use "
            "`generation_mode=reconstruction` and are separated from deterministic text-to-pose outputs.\n"
        ),
    )
    write_markdown(
        paths["text_to_token_report"],
        (
            "# Text-To-Token Report\n\n"
            "Text-to-token generation quality must be judged through generated-pose validation metrics. "
            f"{_temporal_limitation(config.tokenizer.temporal_granularity, config.tokenizer.window_size, config.tokenizer.stride)}\n"
        ),
    )
    write_markdown(
        paths["generation_report"],
        (
            "# Generation Report\n\n"
            f"- Generation mode: `{generation_metadata.get('generation_mode', 'deterministic')}`\n"
            f"- Length policy: `{generation_metadata.get('length_policy', 'predicted_length')}`\n"
            f"- Confidence policy: `{generation_metadata.get('confidence_policy', 'synthetic_validity')}`\n"
            f"- Token selection: `{generation_metadata.get('token_selection', 'argmax')}`\n"
            f"- Manifest paths: `{generation_metadata.get('manifest_paths', {})}`\n"
        ),
    )
    write_markdown(paths["limitations"], _phase6_limitations_markdown(semantic_lines, config))
    return tuple(
        ModelStageArtifactRef(
            role=f"learned_pose_token_{role}_report",
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


def _semantic_objective_lines(context: ModelStageExecutionContext) -> tuple[str, ...]:
    if not any(
        objective.value == "semantic_consistency"
        for objective in context.request.auxiliary_objectives
    ):
        return ("No semantic objective is attached.",)
    return (
        "Semantic consistency objective artifacts are attached as post-generation proxy evaluation in this stage.",
        "Semantic consistency is not differentiable training optimization yet.",
        "An executed with/without ablation baseline is required before contribution claims.",
    )


def _phase6_limitations_markdown(
    semantic_lines: tuple[str, ...],
    config: LearnedPoseTokenConfig,
) -> str:
    return (
        "# Limitations\n\n"
        "- Learned pose-token bottleneck is the isolated mechanism.\n"
        "- Reconstruction quality is not semantic adequacy.\n"
        "- Text-to-token generation quality must be judged through generated-pose validation metrics.\n"
        "- Codebook stability does not prove sign intelligibility.\n"
        f"- {_temporal_limitation(config.tokenizer.temporal_granularity, config.tokenizer.window_size, config.tokenizer.stride)}\n"
        + "".join(f"- {line}\n" for line in semantic_lines)
        + "- No retrieval comparator is active.\n"
        "- No diffusion mechanism is active.\n"
        "- No articulator-specific mechanism is active.\n"
    )


def _limitations_for_config(config: LearnedPoseTokenConfig) -> tuple[str, ...]:
    return _limitations_for_mode(
        config.tokenizer.temporal_granularity,
        config.tokenizer.window_size,
        config.tokenizer.stride,
    )


def _limitations_for_mode(
    temporal_granularity: str,
    window_size: int,
    stride: int,
) -> tuple[str, ...]:
    return (
        *_BASE_LIMITATIONS,
        _temporal_limitation(temporal_granularity, window_size, stride),
    )


def _temporal_limitation(
    temporal_granularity: str,
    window_size: int,
    stride: int,
) -> str:
    if temporal_granularity == "window" and window_size > 1:
        return (
            "Window-level tokens model local motion chunks but do not prove sign-level "
            "segmentation, duration adequacy, or semantic correctness."
        )
    return "Frame-level tokenization does not explicitly model duration."


__all__ = [
    "build_codebook_stability_report_dict",
    "build_reconstruction_report_markdown",
    "build_token_manifest_summary",
    "build_tokenizer_design_report",
    "write_learned_pose_token_reports",
]
