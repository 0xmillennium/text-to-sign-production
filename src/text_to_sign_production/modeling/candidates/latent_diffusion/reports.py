"""Deterministic report builders for latent_diffusion foundation decisions."""

from __future__ import annotations

from text_to_sign_production.core.io import write_markdown
from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    LatentDiffusionConfig,
    latent_diffusion_config_from_mapping,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.latents import (
    LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
    LatentTargetSpec,
)
from text_to_sign_production.modeling.candidates.provider import ModelStageExecutionContext
from text_to_sign_production.modeling.candidates.results import (
    ModelExecutionResult,
    ModelStageArtifactRef,
)

_CAVEATS = (
    "Latent diffusion is the isolated mechanism.",
    "High literature support does not imply low implementation risk.",
    "Stochastic generation is seed-sensitive.",
    "Plausible motion does not prove semantic correctness.",
    "Automatic pose metrics do not prove sign intelligibility.",
    "Validation metrics are not final test performance.",
    "Semantic consistency, when attached, is post-generation proxy evaluation only.",
    "Semantic consistency is not differentiable training optimization in this stage.",
    "An executed with/without ablation baseline is required before semantic contribution claims.",
    "No retrieval comparator is active.",
    "No articulator-aware mechanism is active.",
    "No audio conditioning is active.",
    "No sparse-keyframe / CFM mechanism is active.",
    "No 3D/avatar/rendering output is active.",
    "Learned pose-token bottleneck is not the primary mechanism here.",
)


def latent_target_decision_report(
    *,
    config: LatentDiffusionConfig,
    target_spec: LatentTargetSpec,
) -> dict[str, object]:
    """Return the deterministic latent-target decision report payload."""

    if target_spec.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
        decision = (
            "Use a provider-owned continuous temporal-window autoencoder latent as "
            "the latent diffusion target."
        )
    else:
        decision = "Use standardized full-BFH frame vectors as the foundation latent target."
    return {
        "schema_version": "t2sp-latent-target-decision-report-v1",
        "model_key": config.identity.model_key,
        "phase_number": config.identity.phase_number,
        "target_spec": target_spec.to_dict(),
        "decision": decision,
        "caveats": list(_CAVEATS),
    }


def seed_policy_report(*, config: LatentDiffusionConfig) -> dict[str, object]:
    """Return the deterministic seed policy report payload."""

    return {
        "schema_version": "t2sp-latent-seed-policy-report-v1",
        "model_key": config.identity.model_key,
        "seed_policy": config.generation.seed_policy,
        "training_seed": config.training.seed,
        "candidate_count": config.generation.candidate_count,
        "statements": [
            "Stochastic generation is seed-sensitive.",
            "Sampling must record the explicit seed and candidate index.",
        ],
        "caveats": list(_CAVEATS),
    }


def compute_failure_cost_report(*, config: LatentDiffusionConfig) -> dict[str, object]:
    """Return a deterministic compute/failure-cost report payload."""

    return {
        "schema_version": "t2sp-latent-compute-failure-cost-report-v1",
        "model_key": config.identity.model_key,
        "schedule_timesteps": config.diffusion.timesteps,
        "sampling_steps": config.diffusion.sampling_steps,
        "risk_statements": [
            "High literature support does not imply low implementation risk.",
            "Longer diffusion schedules increase compute cost and failure investigation cost.",
            "Plausible motion does not prove sign intelligibility.",
        ],
        "caveats": list(_CAVEATS),
    }


def denoising_setup_report(*, config: LatentDiffusionConfig) -> dict[str, object]:
    """Return the deterministic denoising setup report payload."""

    return {
        "schema_version": "t2sp-latent-denoising-setup-report-v1",
        "model_key": config.identity.model_key,
        "architecture": config.denoiser.architecture,
        "prediction_type": config.denoiser.prediction_type,
        "loss": config.diffusion.loss,
        "statements": [
            "Text-conditioned masked-MSE epsilon prediction is the active foundation setup.",
            "Semantic auxiliary objective is not active.",
            "Retrieval/stitching is not active.",
        ],
        "caveats": list(_CAVEATS),
    }


def generation_caveat_report(*, config: LatentDiffusionConfig) -> str:
    """Return a deterministic markdown caveat report for generated latent poses."""

    lines = [
        "# Latent Diffusion Generation Caveats",
        "",
        f"- Model key: `{config.identity.model_key}`",
        f"- Generation mode: `{config.generation.generation_mode}`",
        f"- Seed policy: `{config.generation.seed_policy}`",
        "",
        *[f"- {statement}" for statement in _CAVEATS],
        "",
    ]
    return "\n".join(lines)


def write_latent_diffusion_reports(
    *,
    context: ModelStageExecutionContext,
    results: ModelExecutionResult,
) -> tuple[ModelStageArtifactRef, ...]:
    """Write provider-owned Phase 7 reports under the workflow report root."""

    raw = context.loaded_config.effective_config.get("provider_config")
    if not isinstance(raw, dict):
        raise ValueError("latent_diffusion reports require provider_config in effective_config.")
    config = latent_diffusion_config_from_mapping(raw)
    cache_metadata = _metadata_for_stage(results, "cache_latents")
    generate_metadata = _metadata_for_stage(results, "generate")
    semantic_attached = any(
        objective.value == "semantic_consistency"
        for objective in context.request.auxiliary_objectives
    )
    root = context.topology.reports.model_run_root(
        config.identity.model_key,
        context.request.run_name,
    ).path
    root.mkdir(parents=True, exist_ok=True)
    stage_lines = "\n".join(
        f"- `{stage.stage.spec.kind.value}`: `{stage.status.value}`"
        for stage in results.stages
    )
    report_payloads = {
        "latent_diffusion_run_summary.md": _markdown(
            "Latent Diffusion Run Summary",
            [
                f"Model key: `{config.identity.model_key}`",
                f"Run name: `{context.request.run_name}`",
                f"Manifest family: `{context.request.manifest_family.family_id}`",
                f"Run mode: `{context.request.run_mode.value}`",
                "Latent diffusion is the isolated mechanism.",
                (
                    "Standardization missing-observation policy: "
                    f"`{cache_metadata.get('standardization_missing_observation_policy', 'not recorded')}`"
                ),
                (
                    "Zero-observation coordinates: "
                    f"`{cache_metadata.get('standardization_zero_observation_coordinate_count', 'not recorded')}`"
                ),
                (
                    "Channel fallback coordinates: "
                    f"`{cache_metadata.get('standardization_channel_fallback_coordinate_count', 'not recorded')}`"
                ),
                (
                    "Sampling steps: "
                    f"`{generate_metadata.get('effective_sampling_steps', config.diffusion.sampling_steps)}` "
                    f"(source `{generate_metadata.get('sampling_steps_source', 'config')}`)"
                ),
                "Standardization fallback diagnostics indicate sparse observations; smoke fallback is not a model quality claim.",
                "Smoke/debug reduced sampling steps are for wiring/performance, not final quality.",
                (
                    "Semantic consistency objective is attached for post-generation proxy evaluation."
                    if semantic_attached
                    else "No semantic auxiliary objective is attached."
                ),
                "Generated validation outputs are not final test performance.",
                "",
                "## Stages",
                stage_lines,
            ],
        ),
        "latent_target_report.md": _markdown(
            "Latent Target Report",
            _target_report_lines(config),
        ),
        "denoising_training_report.md": _markdown(
            "Denoising Training Report",
            [
                "The active objective is masked epsilon denoising plus predicted-length MSE.",
                (
                    "Semantic consistency is attached only after generation; it is not a denoising training loss."
                    if semantic_attached
                    else "No semantic auxiliary objective is attached."
                ),
                "No retrieval comparator is active.",
                "High literature support does not imply low implementation risk.",
            ],
        ),
        "seed_policy_report.md": _markdown(
            "Seed Policy Report",
            [
                f"Seed policy: `{config.generation.seed_policy}`",
                "Formula: sha256(f'{run_seed}:{seed_policy}:{sample_id}:{candidate_index}') mod 2147483647",
                "Stochastic generation is seed-sensitive.",
            ],
        ),
        "compute_failure_cost_report.md": _markdown(
            "Compute Failure Cost Report",
            [
                f"Diffusion timesteps: `{config.diffusion.timesteps}`",
                f"Sampling steps: `{config.diffusion.sampling_steps}`",
                "Smoke/debug reduced sampling steps, when active, are for wiring/performance only.",
                "High literature support does not imply low implementation risk.",
            ],
        ),
        "generation_report.md": _markdown(
            "Generation Report",
            [
                f"Generation mode: `{config.generation.generation_mode}`",
                f"Candidate count: `{config.generation.candidate_count}`",
                "Plausible motion does not prove semantic correctness.",
                "Automatic pose metrics do not prove sign intelligibility.",
            ],
        ),
        "limitations.md": _markdown("Limitations", list(_CAVEATS)),
    }
    artifacts: list[ModelStageArtifactRef] = []
    for filename, content in report_payloads.items():
        path = root / filename
        write_markdown(path, content)
        artifacts.append(
            ModelStageArtifactRef(
                role=filename.removesuffix(".md"),
                path=path,
                kind="model_latent_provider_report",
            )
        )
    return tuple(artifacts)


def _markdown(title: str, lines: list[str]) -> str:
    body = "\n".join(f"- {line}" if line else "" for line in lines)
    caveats = "\n".join(f"- {statement}" for statement in _CAVEATS)
    return f"# {title}\n\n{body}\n\n## Required Caveats\n\n{caveats}\n"


def _metadata_for_stage(results: ModelExecutionResult, stage_kind: str) -> dict[str, object]:
    for stage in results.stages:
        if stage.stage.spec.kind.value == stage_kind:
            return dict(stage.metadata)
    return {}


def _target_report_lines(config: LatentDiffusionConfig) -> list[str]:
    if config.latent_target.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
        return [
            "target_type=learned_bfh_window_latent",
            "temporal_granularity=window",
            (
                "The latent is a provider-owned continuous temporal window "
                "autoencoder representation."
            ),
            "It is not the Phase 6 learned_pose_token codebook.",
            "It is not a semantic sign representation.",
            "Reconstruction quality constrains generation quality.",
            "Automatic metrics do not prove sign intelligibility.",
        ]
    return [
        "target_type=standardized_bfh_frame",
        "temporal_granularity=frame",
        "This is frame-space pose diffusion over standardized BFH vectors.",
        "Plausible pose-space motion does not prove semantic correctness.",
        "Learned pose-token bottleneck is not the primary mechanism here.",
    ]


__all__ = [
    "compute_failure_cost_report",
    "denoising_setup_report",
    "generation_caveat_report",
    "latent_target_decision_report",
    "seed_policy_report",
    "write_latent_diffusion_reports",
]
