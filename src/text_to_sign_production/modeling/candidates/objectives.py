"""Planning contracts for auxiliary objective attachments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from text_to_sign_production.modeling.candidates.configs import validate_objective_attachments
from text_to_sign_production.modeling.candidates.errors import ObjectiveAttachmentError
from text_to_sign_production.modeling.research import ModelSpec, ObjectiveKey, ObjectiveSpec


@dataclass(frozen=True, slots=True)
class ObjectiveAttachmentPlan:
    """A compatible auxiliary objective attached to a model plan."""

    objective: ObjectiveSpec
    model: ModelSpec
    stage_id: str
    required_ablation: bool

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveSpec):
            raise ObjectiveAttachmentError("objective must be an ObjectiveSpec.")
        if not isinstance(self.model, ModelSpec):
            raise ObjectiveAttachmentError("model must be a ModelSpec.")
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ObjectiveAttachmentError("stage_id must be non-empty.")
        if self.model.key not in self.objective.allowed_attachment_models:
            raise ObjectiveAttachmentError(
                f"objective {self.objective.key.value!r} cannot attach to model "
                f"{self.model.key.value!r}."
            )
        if self.objective.key not in self.model.compatible_objectives:
            raise ObjectiveAttachmentError(
                f"model {self.model.key.value!r} does not declare objective "
                f"{self.objective.key.value!r} as compatible."
            )
        if self.objective.requires_ablation and self.required_ablation is not True:
            raise ObjectiveAttachmentError(
                f"objective {self.objective.key.value!r} requires an ablation plan."
            )


def build_objective_attachment_plans(
    model_spec: ModelSpec,
    objectives: Iterable[ObjectiveKey | str],
) -> tuple[ObjectiveAttachmentPlan, ...]:
    """Build deterministic attachment-plan records for compatible objectives."""

    objective_specs = validate_objective_attachments(model_spec, objectives)
    return tuple(
        ObjectiveAttachmentPlan(
            objective=objective,
            model=model_spec,
            stage_id=f"{objective.key.value}.attach",
            required_ablation=objective.requires_ablation,
        )
        for objective in objective_specs
    )


__all__ = [
    "ObjectiveAttachmentPlan",
    "build_objective_attachment_plans",
]
