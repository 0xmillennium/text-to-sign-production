"""Structure-aware report builders for the articulator-aware foundation."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.io import write_markdown
from text_to_sign_production.modeling.candidates.articulator_aware.diagnostics import (
    ChannelDiagnosticSummary,
)
from text_to_sign_production.modeling.candidates.articulator_aware.losses import (
    ChannelLossWeightingPolicy,
)
from text_to_sign_production.modeling.candidates.articulator_aware.masks import (
    ChannelMaskSummary,
)
from text_to_sign_production.modeling.candidates.articulator_aware.partitions import (
    ArticulatorChannelPartitionPolicy,
)
from text_to_sign_production.modeling.candidates.provider import ModelStageExecutionContext
from text_to_sign_production.modeling.candidates.results import (
    ModelExecutionResult,
    ModelStageArtifactRef,
)

_REQUIRED_CAVEATS = (
    "Articulator-aware structure is the isolated mechanism.",
    "Partitions are derived from canonical BFH channels.",
    "channel_fusion_mlp uses channel-specific heads but not channel-specific temporal branches.",
    "articulator_factorized_temporal uses per-channel temporal branches and gated cross-channel fusion when active.",
    "Face channel is only a proxy for non-manual signal, not linguistic non-manual annotation.",
    "Hand coordination loss is a motion prior, not a linguistic symmetry assumption.",
    "Automatic channel metrics do not prove semantic adequacy.",
    "This model does not prove sign intelligibility.",
    "Channel-level improvements do not prove sign-level adequacy.",
    "No gloss supervision is used.",
    "No learned pose-token bottleneck is active.",
    "No latent diffusion mechanism is active.",
    "No retrieval comparator is active.",
    "Semantic consistency, when attached, is post-generation proxy evaluation only.",
    "Semantic consistency is not differentiable training optimization in this stage.",
    "An executed with/without ablation baseline is required before semantic contribution claims.",
    "No 3D/avatar/rendering is active.",
)


def write_articulator_markdown_report(
    path: Path,
    *,
    title: str,
    lines: tuple[str, ...] | list[str],
) -> None:
    """Write one report with the provider's required interpretation limits."""

    body = "\n".join(f"- {line}" for line in (*lines, *_REQUIRED_CAVEATS))
    write_markdown(path, f"# {title}\n\n{body}\n")


def partition_policy_report(policy: ArticulatorChannelPartitionPolicy) -> dict[str, object]:
    """Build a deterministic partition policy report."""

    return {
        "schema_version": "t2sp-articulator-partition-report-v1",
        "model_key": "articulator_aware",
        "source": policy.source,
        "primary_channels": [channel.value for channel in policy.primary_channels],
        "partition_joint_counts": {
            partition.channel.value: partition.joint_count for partition in policy.partitions
        },
        "composite_groups": {
            name: [channel.value for channel in policy.composite_groups[name]]
            for name in sorted(policy.composite_groups)
        },
        "caveats": list(_REQUIRED_CAVEATS),
    }


def mask_strategy_report(summaries: tuple[ChannelMaskSummary, ...]) -> dict[str, object]:
    """Build a deterministic mask strategy report from mask summaries."""

    return {
        "schema_version": "t2sp-articulator-mask-report-v1",
        "model_key": "articulator_aware",
        "sample_count": len(summaries),
        "all_invalid_channel_mentions": sum(
            len(summary.all_invalid_channels) for summary in summaries
        ),
        "issues": sorted({issue for summary in summaries for issue in summary.issues}),
        "caveats": list(_REQUIRED_CAVEATS),
    }


def loss_weighting_report(policy: ChannelLossWeightingPolicy) -> dict[str, object]:
    """Build a deterministic loss weighting report."""

    return {
        "schema_version": "t2sp-articulator-loss-weighting-report-v1",
        "model_key": "articulator_aware",
        "policy": policy.policy,
        "normalize_weights": policy.normalize_weights,
        "channel_weights": {
            channel.value: policy.channel_weights[channel] for channel in policy.channel_weights
        },
        "normalized_channel_weights": {
            channel.value: policy.normalized_channel_weights[channel]
            for channel in policy.normalized_channel_weights
        },
        "velocity_weight": policy.velocity_weight,
        "symmetry_weight": policy.symmetry_weight,
        "cross_channel_consistency_weight": policy.cross_channel_consistency_weight,
        "active_auxiliary_losses": {
            "velocity": policy.velocity_weight > 0.0,
            "hand_motion_coordination": policy.symmetry_weight > 0.0,
            "cross_channel_temporal_consistency": policy.cross_channel_consistency_weight > 0.0,
        },
        "caveats": list(_REQUIRED_CAVEATS),
    }


def channel_diagnostics_report(summary: ChannelDiagnosticSummary) -> dict[str, object]:
    """Build a deterministic diagnostics report."""

    return {
        "schema_version": "t2sp-articulator-channel-diagnostics-report-v1",
        "model_key": "articulator_aware",
        "split": summary.split.value,
        "channel_count": summary.channel_count,
        "sample_count": summary.sample_count,
        "records_count": summary.records_count,
        "skipped_records_count": summary.skipped_records_count,
        "per_channel_valid_fraction_mean": dict(summary.per_channel_valid_fraction_mean),
        "per_channel_loss_mean": dict(summary.per_channel_loss_mean),
        "caveats": list(dict.fromkeys((*summary.caveats, *_REQUIRED_CAVEATS))),
    }


def structure_candidate_limitations_report() -> dict[str, object]:
    """Build the required limitations report for this foundation stage."""

    return {
        "schema_version": "t2sp-articulator-limitations-report-v1",
        "model_key": "articulator_aware",
        "active_mechanism": "articulator_aware_structure",
        "inactive_mechanisms": {
            "gloss_supervision": False,
            "learned_pose_token_bottleneck": False,
            "latent_diffusion": False,
            "retrieval_comparator": False,
            "semantic_auxiliary_objective": False,
            "non_manual_linguistic_annotation": False,
            "avatar_or_3d_rendering": False,
        },
        "caveats": list(_REQUIRED_CAVEATS),
    }


def write_articulator_aware_reports(
    *,
    context: ModelStageExecutionContext,
    results: ModelExecutionResult,
) -> tuple[ModelStageArtifactRef, ...]:
    """Write provider reports under the workflow-owned report topology."""

    root = context.topology.reports.model_run_root(
        "articulator_aware", context.request.run_name
    ).path
    stage_lines = [
        f"Stage `{result.stage.spec.kind.value}` completed with status `{result.status.value}`."
        for result in results.stages
    ]
    semantic_attached = any(
        objective.value == "semantic_consistency"
        for objective in context.request.auxiliary_objectives
    )
    semantic_line = (
        "Semantic consistency objective is attached for post-generation proxy evaluation only."
        if semantic_attached
        else "No semantic auxiliary objective is attached."
    )
    report_lines = {
        "articulator_aware_run_summary.md": [
            f"Run name: `{context.request.run_name}`",
            f"Manifest family: `{context.request.manifest_family.family_id}`",
            f"Run mode: `{context.request.run_mode.value}`",
            "Training `batch_size` is the source/sample batch size.",
            "Training `frame_batch_size` is the actual frame tensor batch cap.",
            semantic_line,
            *stage_lines,
        ],
        "partition_policy_report.md": [
            "Partitions are derived from canonical BFH channels.",
        ],
        "mask_strategy_report.md": [
            "Observation masks derive from BFH confidence and frame validity.",
        ],
        "loss_weighting_report.md": [
            "Static channel-weighted reconstruction is active.",
            "Auxiliary losses are architecture-aware; factorized temporal runs may activate velocity and cross-channel motion priors.",
            "Hand coordination is reported as a motion prior, not literal linguistic symmetry.",
        ],
        "training_channel_diagnostics_report.md": [
            "Diagnostics report validation behavior, not final test performance.",
        ],
        "generation_report.md": [
            "Generation mode is deterministic.",
            "Generated length is predicted from text rather than copied from reference pose.",
        ],
        "limitations.md": [],
    }
    artifacts: list[ModelStageArtifactRef] = []
    for filename, lines in report_lines.items():
        path = root / filename
        title = filename.removesuffix(".md").replace("_", " ").title()
        if not path.is_file():
            write_articulator_markdown_report(path, title=title, lines=lines)
        artifacts.append(
            ModelStageArtifactRef(
                role=filename.removesuffix(".md"),
                path=path,
                kind="model_articulator_report",
                metadata={"producer_stage": "write_model_reports"},
            )
        )
    return tuple(artifacts)


__all__ = [
    "channel_diagnostics_report",
    "loss_weighting_report",
    "mask_strategy_report",
    "partition_policy_report",
    "structure_candidate_limitations_report",
    "write_articulator_markdown_report",
    "write_articulator_aware_reports",
]
