"""Research-backed modeling domain contracts."""

from __future__ import annotations

from text_to_sign_production.modeling.research.identifiers import (
    ComparatorKey,
    EvaluationProtocolKey,
    ModelKey,
    ObjectiveKey,
)
from text_to_sign_production.modeling.research.mechanisms import MechanismTag
from text_to_sign_production.modeling.research.phases import ResearchPhase, phase_number
from text_to_sign_production.modeling.research.reports import RequiredReport
from text_to_sign_production.modeling.research.risks import RiskControl
from text_to_sign_production.modeling.research.roles import ResearchArtifactKind, ResearchRole
from text_to_sign_production.modeling.research.specs import (
    ComparatorSpec,
    EvaluationProtocolSpec,
    ModelSpec,
    ObjectiveSpec,
)
from text_to_sign_production.modeling.research.traceability import (
    ResearchTrace,
    ResearchTraceIssue,
    validate_trace_paths,
)
from text_to_sign_production.modeling.research.validation import (
    ModelingResearchSpecError,
    ValidationRequirement,
)

__all__ = [
    "ComparatorKey",
    "ComparatorSpec",
    "EvaluationProtocolKey",
    "EvaluationProtocolSpec",
    "MechanismTag",
    "ModelKey",
    "ModelSpec",
    "ModelingResearchSpecError",
    "ObjectiveKey",
    "ObjectiveSpec",
    "RequiredReport",
    "ResearchArtifactKind",
    "ResearchPhase",
    "ResearchRole",
    "ResearchTrace",
    "ResearchTraceIssue",
    "RiskControl",
    "ValidationRequirement",
    "phase_number",
    "validate_trace_paths",
]
