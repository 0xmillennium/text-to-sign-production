"""Typed research-backed specifications for modeling artifacts."""

from __future__ import annotations

import enum
from dataclasses import dataclass

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
from text_to_sign_production.modeling.research.traceability import ResearchTrace
from text_to_sign_production.modeling.research.validation import (
    ModelingResearchSpecError,
    ValidationRequirement,
    ensure_disjoint_enum_values,
    require_non_empty,
    require_unique_enum_members,
    require_unique_strings,
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Research-backed model candidate specification."""

    key: ModelKey
    canonical_id: str
    display_name: str
    phase: ResearchPhase
    phase_number: int
    research_role: ResearchRole
    trace: ResearchTrace
    allowed_mechanisms: tuple[MechanismTag, ...]
    forbidden_mechanisms: tuple[MechanismTag, ...]
    input_contract: str
    output_contract: str
    generated_pose_required: bool
    default_stage_sequence: tuple[str, ...]
    compatible_objectives: tuple[ObjectiveKey, ...]
    targeted_failure_modes: tuple[str, ...]
    required_reports: tuple[RequiredReport, ...]
    validation_requirements: tuple[ValidationRequirement, ...]
    risk_controls: tuple[RiskControl, ...]

    @property
    def artifact_kind(self) -> ResearchArtifactKind:
        """Return the research artifact kind represented by this spec."""

        return ResearchArtifactKind.MODEL

    def __post_init__(self) -> None:
        _coerce_common(self, key_type=ModelKey)
        if self.research_role not in (
            ResearchRole.BASELINE_OR_ABLATION_FLOOR,
            ResearchRole.PRIMARY_MODEL_CANDIDATE,
        ):
            raise ModelingResearchSpecError(
                "model research_role must be baseline_or_ablation_floor or primary_model_candidate."
            )
        if self.generated_pose_required is not True:
            raise ModelingResearchSpecError("model generated_pose_required must be True.")
        require_unique_strings(self.default_stage_sequence, field_name="default_stage_sequence")
        require_unique_enum_members(
            self.compatible_objectives,
            enum_type=ObjectiveKey,
            field_name="compatible_objectives",
        )
        require_unique_strings(self.targeted_failure_modes, field_name="targeted_failure_modes")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable model specification."""

        return _base_spec_dict(self) | {
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "generated_pose_required": self.generated_pose_required,
            "default_stage_sequence": list(self.default_stage_sequence),
            "compatible_objectives": _enum_values(self.compatible_objectives),
            "targeted_failure_modes": list(self.targeted_failure_modes),
            "required_reports": _to_dict_list(self.required_reports),
            "validation_requirements": _to_dict_list(self.validation_requirements),
            "risk_controls": _to_dict_list(self.risk_controls),
        }


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Research-backed auxiliary objective specification."""

    key: ObjectiveKey
    canonical_id: str
    display_name: str
    phase: ResearchPhase
    phase_number: int
    research_role: ResearchRole
    trace: ResearchTrace
    standalone_model_allowed: bool
    allowed_attachment_models: tuple[ModelKey, ...]
    requires_ablation: bool
    allowed_mechanisms: tuple[MechanismTag, ...]
    forbidden_mechanisms: tuple[MechanismTag, ...]
    required_reports: tuple[RequiredReport, ...]
    validation_requirements: tuple[ValidationRequirement, ...]
    risk_controls: tuple[RiskControl, ...]

    @property
    def artifact_kind(self) -> ResearchArtifactKind:
        """Return the research artifact kind represented by this spec."""

        return ResearchArtifactKind.OBJECTIVE

    def __post_init__(self) -> None:
        _coerce_common(self, key_type=ObjectiveKey)
        if self.research_role is not ResearchRole.AUXILIARY_ADDITIVE_OBJECTIVE:
            raise ModelingResearchSpecError(
                "objective research_role must be auxiliary_additive_objective."
            )
        if self.standalone_model_allowed is not False:
            raise ModelingResearchSpecError("standalone_model_allowed must be False.")
        if self.requires_ablation is not True:
            raise ModelingResearchSpecError("requires_ablation must be True.")
        require_unique_enum_members(
            self.allowed_attachment_models,
            enum_type=ModelKey,
            field_name="allowed_attachment_models",
        )
        if not self.allowed_attachment_models:
            raise ModelingResearchSpecError("allowed_attachment_models must be non-empty.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable objective specification."""

        return _base_spec_dict(self) | {
            "standalone_model_allowed": self.standalone_model_allowed,
            "allowed_attachment_models": _enum_values(self.allowed_attachment_models),
            "requires_ablation": self.requires_ablation,
            "required_reports": _to_dict_list(self.required_reports),
            "validation_requirements": _to_dict_list(self.validation_requirements),
            "risk_controls": _to_dict_list(self.risk_controls),
        }


@dataclass(frozen=True, slots=True)
class ComparatorSpec:
    """Research-backed comparator specification."""

    key: ComparatorKey
    canonical_id: str
    display_name: str
    phase: ResearchPhase
    phase_number: int
    research_role: ResearchRole
    trace: ResearchTrace
    primary_model_allowed: bool
    generated_pose_required: bool
    requires_leakage_policy: bool
    allowed_mechanisms: tuple[MechanismTag, ...]
    forbidden_mechanisms: tuple[MechanismTag, ...]
    required_reports: tuple[RequiredReport, ...]
    validation_requirements: tuple[ValidationRequirement, ...]
    risk_controls: tuple[RiskControl, ...]

    @property
    def artifact_kind(self) -> ResearchArtifactKind:
        """Return the research artifact kind represented by this spec."""

        return ResearchArtifactKind.COMPARATOR

    def __post_init__(self) -> None:
        _coerce_common(self, key_type=ComparatorKey)
        if self.research_role is not ResearchRole.COUNTER_ALTERNATIVE_COMPARATOR:
            raise ModelingResearchSpecError(
                "comparator research_role must be counter_alternative_comparator."
            )
        if self.primary_model_allowed is not False:
            raise ModelingResearchSpecError("primary_model_allowed must be False.")
        if self.generated_pose_required is not True:
            raise ModelingResearchSpecError("generated_pose_required must be True.")
        if self.requires_leakage_policy is not True:
            raise ModelingResearchSpecError("requires_leakage_policy must be True.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable comparator specification."""

        return _base_spec_dict(self) | {
            "primary_model_allowed": self.primary_model_allowed,
            "generated_pose_required": self.generated_pose_required,
            "requires_leakage_policy": self.requires_leakage_policy,
            "required_reports": _to_dict_list(self.required_reports),
            "validation_requirements": _to_dict_list(self.validation_requirements),
            "risk_controls": _to_dict_list(self.risk_controls),
        }


@dataclass(frozen=True, slots=True)
class EvaluationProtocolSpec:
    """Research-backed evaluation protocol specification."""

    key: EvaluationProtocolKey
    canonical_id: str
    display_name: str
    phase: ResearchPhase
    phase_number: int
    research_role: ResearchRole
    trace: ResearchTrace
    required_input_contracts: tuple[str, ...]
    required_output_contracts: tuple[str, ...]
    required_metric_families: tuple[str, ...]
    optional_metric_families: tuple[str, ...]
    required_qualitative_surfaces: tuple[str, ...]
    required_limitation_statements: tuple[str, ...]
    required_reports: tuple[RequiredReport, ...]
    validation_requirements: tuple[ValidationRequirement, ...]
    risk_controls: tuple[RiskControl, ...]

    @property
    def artifact_kind(self) -> ResearchArtifactKind:
        """Return the research artifact kind represented by this spec."""

        return ResearchArtifactKind.EVALUATION_PROTOCOL

    def __post_init__(self) -> None:
        _coerce_common(self, key_type=EvaluationProtocolKey)
        if self.research_role is not ResearchRole.EVALUATION_SUPPORT_SURFACE:
            raise ModelingResearchSpecError(
                "evaluation protocol research_role must be evaluation_support_surface."
            )
        require_unique_strings(self.required_input_contracts, field_name="required_input_contracts")
        require_unique_strings(
            self.required_output_contracts,
            field_name="required_output_contracts",
        )
        require_unique_strings(self.required_metric_families, field_name="required_metric_families")
        require_unique_strings(self.optional_metric_families, field_name="optional_metric_families")
        require_unique_strings(
            self.required_qualitative_surfaces,
            field_name="required_qualitative_surfaces",
        )
        require_unique_strings(
            self.required_limitation_statements,
            field_name="required_limitation_statements",
        )
        if not self.required_metric_families and not self.required_qualitative_surfaces:
            raise ModelingResearchSpecError(
                "evaluation protocols require metrics or qualitative surfaces."
            )
        if not self.required_limitation_statements:
            raise ModelingResearchSpecError("required_limitation_statements must be non-empty.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable evaluation protocol specification."""

        return _base_spec_dict(self) | {
            "required_input_contracts": list(self.required_input_contracts),
            "required_output_contracts": list(self.required_output_contracts),
            "required_metric_families": list(self.required_metric_families),
            "optional_metric_families": list(self.optional_metric_families),
            "required_qualitative_surfaces": list(self.required_qualitative_surfaces),
            "required_limitation_statements": list(self.required_limitation_statements),
            "required_reports": _to_dict_list(self.required_reports),
            "validation_requirements": _to_dict_list(self.validation_requirements),
            "risk_controls": _to_dict_list(self.risk_controls),
        }


def _coerce_common(instance: object, *, key_type: type[enum.StrEnum]) -> None:
    object.__setattr__(instance, "key", key_type(instance.key))
    object.__setattr__(instance, "phase", ResearchPhase(instance.phase))
    object.__setattr__(
        instance,
        "research_role",
        ResearchRole(instance.research_role),
    )
    if instance.phase_number != phase_number(instance.phase):
        raise ModelingResearchSpecError("phase_number must match phase.")
    for field_name in ("canonical_id", "display_name"):
        require_non_empty(getattr(instance, field_name), field_name=field_name)
    if hasattr(instance, "allowed_mechanisms") and hasattr(instance, "forbidden_mechanisms"):
        require_unique_enum_members(
            instance.allowed_mechanisms,
            enum_type=MechanismTag,
            field_name="allowed_mechanisms",
        )
        require_unique_enum_members(
            instance.forbidden_mechanisms,
            enum_type=MechanismTag,
            field_name="forbidden_mechanisms",
        )
        ensure_disjoint_enum_values(
            instance.allowed_mechanisms,
            instance.forbidden_mechanisms,
            left_name="allowed_mechanisms",
            right_name="forbidden_mechanisms",
        )
    _require_unique_identifiers(instance.required_reports, "required_reports")
    _require_unique_identifiers(
        instance.validation_requirements,
        "validation_requirements",
    )
    _require_unique_identifiers(instance.risk_controls, "risk_controls")
    for field_name in ("input_contract", "output_contract"):
        if hasattr(instance, field_name):
            require_non_empty(getattr(instance, field_name), field_name=field_name)


def _base_spec_dict(instance: object) -> dict[str, object]:
    result: dict[str, object] = {
        "key": instance.key.value,
        "canonical_id": instance.canonical_id,
        "display_name": instance.display_name,
        "phase": instance.phase.value,
        "phase_number": instance.phase_number,
        "research_role": instance.research_role.value,
        "artifact_kind": instance.artifact_kind.value,
        "trace": instance.trace.to_dict(),
    }
    if hasattr(instance, "allowed_mechanisms") and hasattr(instance, "forbidden_mechanisms"):
        result["allowed_mechanisms"] = _enum_values(instance.allowed_mechanisms)
        result["forbidden_mechanisms"] = _enum_values(instance.forbidden_mechanisms)
    return result


def _enum_values(values: tuple[enum.StrEnum, ...]) -> list[str]:
    return [value.value for value in values]


def _require_unique_identifiers(values: tuple[object, ...], field_name: str) -> None:
    if not isinstance(values, tuple):
        raise ModelingResearchSpecError(f"{field_name} must be a tuple.")
    identifiers: list[str] = []
    for value in values:
        identifier = getattr(value, "identifier", None)
        require_non_empty(identifier, field_name=f"{field_name}.identifier")
        identifiers.append(identifier)
    require_unique_strings(tuple(identifiers), field_name=f"{field_name}.identifier")


def _to_dict_list(values: tuple[object, ...]) -> list[dict[str, object]]:
    return [value.to_dict() for value in values]


__all__ = [
    "ComparatorSpec",
    "EvaluationProtocolSpec",
    "ModelSpec",
    "ObjectiveSpec",
]
