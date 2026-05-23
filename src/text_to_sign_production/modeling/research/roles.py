"""Research artifact kind and role identifiers."""

from __future__ import annotations

import enum


class ResearchArtifactKind(enum.StrEnum):
    """Top-level kinds for research-backed modeling artifacts."""

    MODEL = "model"
    OBJECTIVE = "objective"
    COMPARATOR = "comparator"
    EVALUATION_PROTOCOL = "evaluation_protocol"


class ResearchRole(enum.StrEnum):
    """Audit-derived role labels for modeling research artifacts."""

    BASELINE_OR_ABLATION_FLOOR = "baseline_or_ablation_floor"
    PRIMARY_MODEL_CANDIDATE = "primary_model_candidate"
    EVALUATION_SUPPORT_SURFACE = "evaluation_support_surface"
    AUXILIARY_ADDITIVE_OBJECTIVE = "auxiliary_additive_objective"
    COUNTER_ALTERNATIVE_COMPARATOR = "counter_alternative_comparator"


__all__ = [
    "ResearchArtifactKind",
    "ResearchRole",
]
