"""Concrete registry entries for research-backed model candidates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.modeling.registry.access import (
    ModelingRegistryError,
    coerce_model_key,
)
from text_to_sign_production.modeling.research import (
    MechanismTag,
    ModelKey,
    ModelSpec,
    ObjectiveKey,
    RequiredReport,
    ResearchPhase,
    ResearchRole,
    ResearchTrace,
    RiskControl,
    ValidationRequirement,
)

_AUDIT_RESULT_PATH = Path("docs/research/contribution-audit/audit-result.md")
_LITERATURE_POSITIONING_PATH = Path("docs/research/literature-positioning.md")
_ROADMAP_PATH = Path("docs/research/roadmap.md")
_CANDIDATE_UNIVERSE_PATH = Path("docs/research/contribution-audit/candidate-universe/index.md")


def _trace(
    *,
    candidate_id: str,
    candidate_name: str,
    candidate_card_path: str,
    scorecard_path: str,
    selection_decision_path: str,
    source_corpus_ids: tuple[str, ...],
) -> ResearchTrace:
    return ResearchTrace(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        candidate_card_path=Path(candidate_card_path),
        candidate_universe_path=_CANDIDATE_UNIVERSE_PATH,
        scorecard_path=Path(scorecard_path),
        selection_decision_path=Path(selection_decision_path),
        audit_result_path=_AUDIT_RESULT_PATH,
        literature_positioning_path=_LITERATURE_POSITIONING_PATH,
        roadmap_path=_ROADMAP_PATH,
        source_corpus_ids=source_corpus_ids,
    )


def _reports(*identifiers: str) -> tuple[RequiredReport, ...]:
    return tuple(
        RequiredReport(
            identifier=identifier,
            title=identifier.replace("_", " ").title(),
        )
        for identifier in identifiers
    )


def _validations(*identifiers: str) -> tuple[ValidationRequirement, ...]:
    return tuple(
        ValidationRequirement(
            identifier=identifier,
            description=identifier.replace("_", " "),
        )
        for identifier in identifiers
    )


def _risks(statements: Mapping[str, str]) -> tuple[RiskControl, ...]:
    return tuple(
        RiskControl(identifier=identifier, statement=statement)
        for identifier, statement in statements.items()
    )


BASE_DIRECT_SPEC = ModelSpec(
    key=ModelKey.BASE_DIRECT,
    canonical_id="m0_direct_text_to_pose",
    display_name="M0 Direct Text-to-Pose Baseline",
    phase=ResearchPhase.PHASE_4,
    phase_number=4,
    research_role=ResearchRole.BASELINE_OR_ABLATION_FLOOR,
    trace=_trace(
        candidate_id="CAND-M0-DIRECT-TEXT-TO-POSE-BASELINE",
        candidate_name="Direct Text-to-Pose Baseline",
        candidate_card_path=(
            "docs/research/contribution-audit/candidate-cards/direct-text-to-pose-baseline.md"
        ),
        scorecard_path=(
            "docs/research/contribution-audit/scorecards/direct-text-to-pose-baseline-scorecard.md"
        ),
        selection_decision_path=(
            "docs/research/contribution-audit/selection-decisions/"
            "direct-text-to-pose-baseline-selection-decision.md"
        ),
        source_corpus_ids=(
            "SRC-PROGTRANS-2020",
            "SRC-MULTICHANNEL-MDN-2021",
            "SRC-HOW2SIGN-2021",
        ),
    ),
    allowed_mechanisms=(MechanismTag.DIRECT_TEXT_TO_POSE,),
    forbidden_mechanisms=(
        MechanismTag.LEARNED_POSE_TOKEN,
        MechanismTag.LATENT_DIFFUSION,
        MechanismTag.ARTICULATOR_AWARE_STRUCTURE,
        MechanismTag.SEMANTIC_CONSISTENCY_OBJECTIVE,
        MechanismTag.RETRIEVAL_REUSE,
        MechanismTag.GLOSS_SUPERVISION,
        MechanismTag.HAMNOSYS_OR_NOTATION,
        MechanismTag.AVATAR_OR_3D,
        MechanismTag.RENDERING,
        MechanismTag.DICTIONARY_STITCHING,
    ),
    input_contract="text_transcript_plus_prepared_pose_reference",
    output_contract="t2sp-generated-pose-v1",
    generated_pose_required=True,
    default_stage_sequence=("train", "generate", "export_generated_pose"),
    compatible_objectives=(),
    targeted_failure_modes=(
        "direct regression oversmoothing",
        "weak temporal structure",
        "hand/face detail loss",
        "semantic drift between transcript and produced motion",
        "baseline underperformance relative to structured candidates",
    ),
    required_reports=_reports(
        "model_spec",
        "architecture_report",
        "training_summary",
        "generated_pose_summary",
        "baseline_failure_modes",
        "risk_controls",
    ),
    validation_requirements=_validations(
        "same_split_policy",
        "same_artifact_schema",
        "generated_pose_contract",
        "evaluation_harness_consumable",
        "reproducible_run_config",
        "baseline_failure_modes_documented",
    ),
    risk_controls=_risks(
        {
            "baseline_not_contribution_strength": (
                "Baseline readiness is not evidence of contribution strength."
            ),
            "baseline_not_task_solving_quality": (
                "Baseline readiness is not evidence of strong task-solving quality."
            ),
            "automatic_metrics_not_intelligibility": (
                "Automatic metrics are not proof of sign intelligibility."
            ),
        }
    ),
)

LEARNED_POSE_TOKEN_SPEC = ModelSpec(
    key=ModelKey.LEARNED_POSE_TOKEN,
    canonical_id="learned_pose_token_bottleneck",
    display_name="Learned Pose-Token Bottleneck",
    phase=ResearchPhase.PHASE_6,
    phase_number=6,
    research_role=ResearchRole.PRIMARY_MODEL_CANDIDATE,
    trace=_trace(
        candidate_id="CAND-LEARNED-POSE-TOKEN-BOTTLENECK",
        candidate_name="Learned Pose-Token Bottleneck",
        candidate_card_path=(
            "docs/research/contribution-audit/candidate-cards/learned-pose-token-bottleneck.md"
        ),
        scorecard_path=(
            "docs/research/contribution-audit/scorecards/learned-pose-token-bottleneck-scorecard.md"
        ),
        selection_decision_path=(
            "docs/research/contribution-audit/selection-decisions/"
            "learned-pose-token-bottleneck-selection-decision.md"
        ),
        source_corpus_ids=(
            "SRC-SIGNVQNET-2024",
            "SRC-DATA-DRIVEN-REP-2024",
            "SRC-T2S-GPT-2024",
            "SRC-UNIGLOR-2024",
            "SRC-HOW2SIGN-2021",
        ),
    ),
    allowed_mechanisms=(
        MechanismTag.LEARNED_POSE_TOKEN,
        MechanismTag.COMPACT_MOTION_CODE,
    ),
    forbidden_mechanisms=(
        MechanismTag.LATENT_DIFFUSION,
        MechanismTag.ARTICULATOR_AWARE_STRUCTURE,
        MechanismTag.RETRIEVAL_REUSE,
        MechanismTag.GLOSS_SUPERVISION,
        MechanismTag.HAMNOSYS_OR_NOTATION,
        MechanismTag.AVATAR_OR_3D,
        MechanismTag.RENDERING,
    ),
    input_contract="text_transcript_plus_prepared_pose_reference",
    output_contract="t2sp-generated-pose-v1",
    generated_pose_required=True,
    default_stage_sequence=(
        "fit_representation",
        "evaluate_reconstruction",
        "train_text_to_token",
        "decode_to_pose",
        "export_generated_pose",
    ),
    compatible_objectives=(ObjectiveKey.SEMANTIC_CONSISTENCY,),
    targeted_failure_modes=(
        "token/codebook collapse",
        "reconstruction-generation mismatch",
        "temporal granularity mismatch",
        "semantic drift after token decoding",
        "overcompression of hand and face detail",
    ),
    required_reports=_reports(
        "model_spec",
        "tokenizer_design",
        "reconstruction_report",
        "codebook_stability",
        "generated_pose_summary",
        "comparison_against_m0",
        "risk_controls",
    ),
    validation_requirements=_validations(
        "same_split_policy",
        "same_artifact_schema",
        "generated_pose_contract",
        "reconstruction_generation_separated",
        "codebook_stability_tracked",
        "evaluation_harness_consumable",
    ),
    risk_controls=_risks(
        {
            "reconstruction_not_semantic_adequacy": (
                "Token reconstruction quality is not semantic adequacy."
            ),
            "semantic_auxiliary_requires_ablation": (
                "Semantic consistency must remain a separately evaluated ablation."
            ),
            "avatar_out_of_scope": "Avatar or rendered output is out of scope.",
        }
    ),
)

LATENT_DIFFUSION_SPEC = ModelSpec(
    key=ModelKey.LATENT_DIFFUSION,
    canonical_id="gloss_free_latent_diffusion",
    display_name="Gloss-Free Latent Diffusion",
    phase=ResearchPhase.PHASE_7,
    phase_number=7,
    research_role=ResearchRole.PRIMARY_MODEL_CANDIDATE,
    trace=_trace(
        candidate_id="CAND-GLOSS-FREE-LATENT-DIFFUSION",
        candidate_name="Gloss-Free Latent Diffusion",
        candidate_card_path=(
            "docs/research/contribution-audit/candidate-cards/gloss-free-latent-diffusion.md"
        ),
        scorecard_path=(
            "docs/research/contribution-audit/scorecards/gloss-free-latent-diffusion-scorecard.md"
        ),
        selection_decision_path=(
            "docs/research/contribution-audit/selection-decisions/"
            "gloss-free-latent-diffusion-selection-decision.md"
        ),
        source_corpus_ids=(
            "SRC-TEXT2SIGNDIFF-2025",
            "SRC-ILRSLP-2025",
            "SRC-NSLPG-2021",
            "SRC-SIGNIDD-2025",
            "SRC-HOW2SIGN-2021",
        ),
    ),
    allowed_mechanisms=(
        MechanismTag.LATENT_DIFFUSION,
        MechanismTag.LATENT_GENERATION,
    ),
    forbidden_mechanisms=(
        MechanismTag.LEARNED_POSE_TOKEN,
        MechanismTag.ARTICULATOR_AWARE_STRUCTURE,
        MechanismTag.RETRIEVAL_REUSE,
        MechanismTag.GLOSS_SUPERVISION,
        MechanismTag.AUDIO_CONDITIONING,
        MechanismTag.CFM_OR_SPARSE_KEYFRAME,
        MechanismTag.AVATAR_OR_3D,
        MechanismTag.RENDERING,
    ),
    input_contract="text_transcript_plus_prepared_pose_reference",
    output_contract="t2sp-generated-pose-v1",
    generated_pose_required=True,
    default_stage_sequence=(
        "define_latent_target",
        "cache_latents",
        "train_denoiser",
        "generate",
        "export_generated_pose",
    ),
    compatible_objectives=(ObjectiveKey.SEMANTIC_CONSISTENCY,),
    targeted_failure_modes=(
        "unstable latent target definition",
        "stochastic generation variance",
        "high compute cost",
        "seed sensitivity",
        "plausible but semantically wrong motion",
        "temporal incoherence",
    ),
    required_reports=_reports(
        "model_spec",
        "latent_target_decision",
        "seed_policy",
        "compute_failure_cost",
        "generation_report",
        "comparison_against_m0",
        "risk_controls",
    ),
    validation_requirements=_validations(
        "same_split_policy",
        "same_artifact_schema",
        "generated_pose_contract",
        "seed_policy_documented",
        "multiple_candidate_generation_supported",
        "compute_failure_cost_documented",
        "evaluation_harness_consumable",
    ),
    risk_controls=_risks(
        {
            "literature_support_not_low_risk": (
                "High literature support is not evidence of low implementation risk."
            ),
            "audio_conditioning_out_of_scope": "Audio-conditioned scope creep is forbidden.",
            "sparse_keyframe_cfm_out_of_scope": ("Sparse-keyframe or CFM expansion is forbidden."),
            "avatar_out_of_scope": "Avatar or 3D rendering scope is forbidden.",
        }
    ),
)

ARTICULATOR_AWARE_SPEC = ModelSpec(
    key=ModelKey.ARTICULATOR_AWARE,
    canonical_id="articulator_aware_structure_candidate",
    display_name="Articulator-Aware Structure Candidate",
    phase=ResearchPhase.PHASE_8,
    phase_number=8,
    research_role=ResearchRole.PRIMARY_MODEL_CANDIDATE,
    trace=_trace(
        candidate_id="CAND-ARTICULATOR-DISENTANGLED-LATENT",
        candidate_name="Articulator-Disentangled Latent Modeling",
        candidate_card_path=(
            "docs/research/contribution-audit/candidate-cards/articulator-disentangled-latent.md"
        ),
        scorecard_path=(
            "docs/research/contribution-audit/scorecards/"
            "articulator-disentangled-latent-scorecard.md"
        ),
        selection_decision_path=(
            "docs/research/contribution-audit/selection-decisions/"
            "articulator-disentangled-latent-selection-decision.md"
        ),
        source_corpus_ids=(
            "SRC-DARSLP-2025",
            "SRC-A2VSLP-2026",
            "SRC-MCST-2024",
            "SRC-MULTICHANNEL-MDN-2021",
            "SRC-HOW2SIGN-2021",
        ),
    ),
    allowed_mechanisms=(MechanismTag.ARTICULATOR_AWARE_STRUCTURE,),
    forbidden_mechanisms=(
        MechanismTag.LEARNED_POSE_TOKEN,
        MechanismTag.LATENT_DIFFUSION,
        MechanismTag.RETRIEVAL_REUSE,
        MechanismTag.NON_MANUAL_LINGUISTIC_ANNOTATION,
        MechanismTag.GLOSS_SUPERVISION,
        MechanismTag.AVATAR_OR_3D,
        MechanismTag.RENDERING,
    ),
    input_contract="text_transcript_plus_prepared_pose_reference",
    output_contract="t2sp-generated-pose-v1",
    generated_pose_required=True,
    default_stage_sequence=(
        "define_channel_partitions",
        "define_mask_strategy",
        "define_loss_weighting",
        "train_structure_aware",
        "export_generated_pose",
    ),
    compatible_objectives=(ObjectiveKey.SEMANTIC_CONSISTENCY,),
    targeted_failure_modes=(
        "hand channel degradation",
        "face/non-manual degradation",
        "cross-channel inconsistency",
        "loss imbalance across articulators",
        "channel-level metric overclaiming",
    ),
    required_reports=_reports(
        "model_spec",
        "partition_policy",
        "mask_policy",
        "loss_weighting_policy",
        "channel_stratified_report",
        "comparison_against_m0",
        "risk_controls",
    ),
    validation_requirements=_validations(
        "same_split_policy",
        "same_artifact_schema",
        "generated_pose_contract",
        "channel_partition_documented",
        "mask_strategy_documented",
        "channel_stratified_metrics_required",
        "evaluation_harness_consumable",
    ),
    risk_controls=_risks(
        {
            "channel_metric_not_sign_level_adequacy": (
                "Channel-level improvement is not sign-level adequacy."
            ),
            "non_manual_linguistic_annotation_out_of_scope": (
                "Full non-manual linguistic annotation is out of scope."
            ),
            "avatar_out_of_scope": "3D parametric avatar expansion is out of scope.",
        }
    ),
)

_MODEL_SPEC_SEQUENCE = (
    BASE_DIRECT_SPEC,
    LEARNED_POSE_TOKEN_SPEC,
    LATENT_DIFFUSION_SPEC,
    ARTICULATOR_AWARE_SPEC,
)

MODEL_SPECS: Mapping[ModelKey, ModelSpec] = MappingProxyType(
    {spec.key: spec for spec in _MODEL_SPEC_SEQUENCE}
)


def list_model_specs() -> tuple[ModelSpec, ...]:
    """Return model specs in phase/key order."""

    return _MODEL_SPEC_SEQUENCE


def get_model_spec(key: ModelKey | str) -> ModelSpec | None:
    """Return a model spec, or None when the key is unknown."""

    try:
        resolved = coerce_model_key(key)
    except ModelingRegistryError:
        return None
    return MODEL_SPECS.get(resolved)


def require_model_spec(key: ModelKey | str) -> ModelSpec:
    """Return a model spec or raise for an unknown key."""

    resolved = coerce_model_key(key)
    try:
        return MODEL_SPECS[resolved]
    except KeyError as exc:
        raise ModelingRegistryError(f"unknown model key {resolved.value!r}") from exc


__all__ = [
    "ARTICULATOR_AWARE_SPEC",
    "BASE_DIRECT_SPEC",
    "LATENT_DIFFUSION_SPEC",
    "LEARNED_POSE_TOKEN_SPEC",
    "MODEL_SPECS",
    "get_model_spec",
    "list_model_specs",
    "require_model_spec",
]
