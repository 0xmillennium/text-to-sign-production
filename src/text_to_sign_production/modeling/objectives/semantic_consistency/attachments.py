"""Attachment policy and required ablation planning for semantic consistency."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticConsistencyObjectiveConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.research import (
    ModelKey,
    ModelSpec,
    ObjectiveKey,
    ObjectiveSpec,
)

SEMANTIC_ATTACHMENT_POLICY_SCHEMA_VERSION = "t2sp-semantic-attachment-policy-v1"
SEMANTIC_ATTACHMENT_DECISION_SCHEMA_VERSION = "t2sp-semantic-attachment-decision-v1"
SEMANTIC_ABLATION_PLAN_SCHEMA_VERSION = "t2sp-semantic-ablation-plan-v2"
_ELIGIBLE_MODELS = frozenset(
    {ModelKey.LEARNED_POSE_TOKEN, ModelKey.LATENT_DIFFUSION, ModelKey.ARTICULATOR_AWARE}
)


@dataclass(frozen=True, slots=True)
class SemanticObjectiveAttachmentPolicy:
    schema_version: str
    objective_key: ObjectiveKey
    allowed_models: tuple[ModelKey, ...]
    forbidden_models: tuple[ModelKey, ...]
    requires_ablation: bool
    standalone_model_allowed: bool
    required_stage_id: str

    def __post_init__(self) -> None:
        _schema(self.schema_version, SEMANTIC_ATTACHMENT_POLICY_SCHEMA_VERSION, "attachment policy")
        object.__setattr__(self, "objective_key", _objective_key(self.objective_key))
        allowed = _models(self.allowed_models, "allowed_models")
        forbidden = _models(self.forbidden_models, "forbidden_models")
        if not allowed or not set(allowed).issubset(_ELIGIBLE_MODELS):
            raise SemanticConsistencyError(
                "semantic attachment policy allowed_models must contain eligible "
                "non-baseline attachment models only."
            )
        if ModelKey.BASE_DIRECT in allowed or ModelKey.BASE_DIRECT not in forbidden:
            raise SemanticConsistencyError(
                "semantic attachment policy must reject base_direct and list it as forbidden."
            )
        if set(allowed) & set(forbidden):
            raise SemanticConsistencyError(
                "semantic attachment policy allowed_models and forbidden_models overlap."
            )
        if self.requires_ablation is not True:
            raise SemanticConsistencyError(
                "semantic attachment policy requires_ablation must be true."
            )
        if self.standalone_model_allowed is not False:
            raise SemanticConsistencyError(
                "semantic attachment policy cannot permit a standalone semantic model."
            )
        _text(self.required_stage_id, "required_stage_id")
        object.__setattr__(self, "allowed_models", allowed)
        object.__setattr__(self, "forbidden_models", forbidden)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "objective_key": self.objective_key.value,
            "allowed_models": [model.value for model in self.allowed_models],
            "forbidden_models": [model.value for model in self.forbidden_models],
            "requires_ablation": self.requires_ablation,
            "standalone_model_allowed": self.standalone_model_allowed,
            "required_stage_id": self.required_stage_id,
        }


@dataclass(frozen=True, slots=True)
class SemanticObjectiveAttachmentDecision:
    schema_version: str
    model_key: ModelKey
    objective_key: ObjectiveKey
    attach_allowed: bool
    requires_ablation: bool
    reasons: tuple[str, ...]
    blocking_issues: tuple[str, ...]

    def __post_init__(self) -> None:
        _schema(
            self.schema_version,
            SEMANTIC_ATTACHMENT_DECISION_SCHEMA_VERSION,
            "attachment decision",
        )
        object.__setattr__(self, "model_key", _model(self.model_key, "model_key"))
        object.__setattr__(self, "objective_key", _objective_key(self.objective_key))
        if not isinstance(self.attach_allowed, bool):
            raise SemanticConsistencyError("attachment decision attach_allowed must be a boolean.")
        if self.requires_ablation is not True:
            raise SemanticConsistencyError(
                "semantic attachment decision requires_ablation must be true."
            )
        reasons = _messages(self.reasons, "reasons")
        issues = _messages(self.blocking_issues, "blocking_issues", allow_empty=True)
        if self.attach_allowed and self.model_key not in _ELIGIBLE_MODELS:
            raise SemanticConsistencyError(
                f"semantic attachment cannot be allowed for model {self.model_key.value!r}; "
                "base_direct is forbidden."
            )
        if self.attach_allowed and issues:
            raise SemanticConsistencyError(
                "allowed semantic attachment decision must not contain blocking_issues."
            )
        if not self.attach_allowed and not issues:
            raise SemanticConsistencyError(
                "rejected semantic attachment decision must contain an actionable blocking issue."
            )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "blocking_issues", issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_key": self.model_key.value,
            "objective_key": self.objective_key.value,
            "attach_allowed": self.attach_allowed,
            "requires_ablation": self.requires_ablation,
            "reasons": list(self.reasons),
            "blocking_issues": list(self.blocking_issues),
        }


@dataclass(frozen=True, slots=True)
class SemanticAblationPlan:
    schema_version: str
    model_key: ModelKey
    objective_key: ObjectiveKey
    baseline_run_name: str | None
    objective_run_name: str | None
    same_manifest_family_required: bool
    same_splits_required: bool
    same_validation_protocol_required: bool
    same_seed_required: bool
    status: str
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        _schema(self.schema_version, SEMANTIC_ABLATION_PLAN_SCHEMA_VERSION, "ablation plan")
        model = _model(self.model_key, "model_key")
        if model is ModelKey.BASE_DIRECT:
            raise SemanticConsistencyError(
                "semantic ablation plan model_key cannot be base_direct; use an eligible "
                "attachment model and compare its with/without-objective runs."
            )
        object.__setattr__(self, "model_key", model)
        object.__setattr__(self, "objective_key", _objective_key(self.objective_key))
        for value, name in (
            (self.baseline_run_name, "baseline_run_name"),
            (self.objective_run_name, "objective_run_name"),
        ):
            if value is not None:
                _text(value, name)
        for name in (
            "same_manifest_family_required",
            "same_splits_required",
            "same_validation_protocol_required",
            "same_seed_required",
        ):
            if not isinstance(getattr(self, name), bool):
                raise SemanticConsistencyError(f"{name} must be a boolean.")
        if not all(
            (
                self.same_manifest_family_required,
                self.same_splits_required,
                self.same_validation_protocol_required,
            )
        ):
            raise SemanticConsistencyError(
                "semantic ablation plan must preserve manifest family, splits, and "
                "validation protocol."
            )
        if self.status not in {"incomplete", "planned"}:
            raise SemanticConsistencyError(
                "semantic ablation plan status must be incomplete or planned; complete "
                "requires a future execution-evidence contract."
            )
        paired = self.baseline_run_name is not None and self.objective_run_name is not None
        if self.status == "planned" and not paired:
            raise SemanticConsistencyError(
                "semantic ablation status cannot be planned without both "
                "baseline_run_name and objective_run_name."
            )
        if self.status == "incomplete" and paired:
            raise SemanticConsistencyError(
                "semantic ablation plan with both run names must be planned until execution "
                "integration can mark it complete."
            )
        issues = _messages(self.issues, "issues", allow_empty=True)
        if self.status == "incomplete" and not issues:
            raise SemanticConsistencyError(
                "incomplete semantic ablation plan must describe the missing run pairing."
            )
        object.__setattr__(self, "issues", issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_key": self.model_key.value,
            "objective_key": self.objective_key.value,
            "baseline_run_name": self.baseline_run_name,
            "objective_run_name": self.objective_run_name,
            "same_manifest_family_required": self.same_manifest_family_required,
            "same_splits_required": self.same_splits_required,
            "same_validation_protocol_required": self.same_validation_protocol_required,
            "same_seed_required": self.same_seed_required,
            "status": self.status,
            "issues": list(self.issues),
        }


def build_semantic_attachment_policy(
    config: SemanticConsistencyObjectiveConfig,
) -> SemanticObjectiveAttachmentPolicy:
    """Build the immutable attachment policy declared by the objective config."""

    _config(config)
    return SemanticObjectiveAttachmentPolicy(
        schema_version=SEMANTIC_ATTACHMENT_POLICY_SCHEMA_VERSION,
        objective_key=config.identity.objective_key,
        allowed_models=config.attachment.allowed_models,
        forbidden_models=config.attachment.forbidden_models,
        requires_ablation=config.identity.requires_ablation,
        standalone_model_allowed=config.identity.standalone_model_allowed,
        required_stage_id=config.attachment.stage_id,
    )


def decide_semantic_attachment(
    *,
    model_spec: ModelSpec,
    objective_spec: ObjectiveSpec,
    policy: SemanticObjectiveAttachmentPolicy,
) -> SemanticObjectiveAttachmentDecision:
    """Record whether registry and foundation policy permit future attachment."""

    if not isinstance(model_spec, ModelSpec):
        raise SemanticConsistencyError("model_spec must be a ModelSpec.")
    if not isinstance(objective_spec, ObjectiveSpec):
        raise SemanticConsistencyError("objective_spec must be an ObjectiveSpec.")
    if not isinstance(policy, SemanticObjectiveAttachmentPolicy):
        raise SemanticConsistencyError("policy must be a SemanticObjectiveAttachmentPolicy.")
    if objective_spec.key is not policy.objective_key:
        raise SemanticConsistencyError(
            "objective_spec does not match semantic attachment policy objective_key."
        )
    blockers: list[str] = []
    if model_spec.key in policy.forbidden_models:
        blockers.append(
            f"model {model_spec.key.value!r} is forbidden by semantic attachment policy; "
            "base_direct remains the non-semantic baseline."
        )
    if model_spec.key not in policy.allowed_models:
        blockers.append(
            f"model {model_spec.key.value!r} is not an allowed semantic_consistency attachment."
        )
    if model_spec.key not in objective_spec.allowed_attachment_models:
        blockers.append(
            f"objective registry does not permit semantic_consistency on {model_spec.key.value!r}."
        )
    if objective_spec.key not in model_spec.compatible_objectives:
        blockers.append(
            f"model registry does not declare semantic_consistency compatible with "
            f"{model_spec.key.value!r}."
        )
    allowed = not blockers
    reasons = (
        (
            f"model {model_spec.key.value!r} is permitted for future semantic_consistency attachment.",
            "With/without ablation remains required before interpreting the auxiliary objective.",
        )
        if allowed
        else ("Semantic consistency attachment was rejected by registry/policy checks.",)
    )
    return SemanticObjectiveAttachmentDecision(
        schema_version=SEMANTIC_ATTACHMENT_DECISION_SCHEMA_VERSION,
        model_key=model_spec.key,
        objective_key=objective_spec.key,
        attach_allowed=allowed,
        requires_ablation=policy.requires_ablation,
        reasons=reasons,
        blocking_issues=tuple(blockers),
    )


def build_semantic_ablation_plan(
    *,
    model_key: ModelKey,
    baseline_run_name: str | None,
    objective_run_name: str | None,
    config: SemanticConsistencyObjectiveConfig,
) -> SemanticAblationPlan:
    """Plan a required with/without-objective pair without executing it."""

    _config(config)
    model = _model(model_key, "model_key")
    if model not in config.attachment.allowed_models:
        raise SemanticConsistencyError(
            f"semantic ablation cannot be planned for model {model.value!r}; choose one of "
            + ", ".join(item.value for item in config.attachment.allowed_models)
            + "."
        )
    issues: list[str] = []
    if baseline_run_name is None:
        issues.append("baseline_run_name is required for the without-objective comparison.")
    if objective_run_name is None:
        issues.append("objective_run_name is required for the with-objective comparison.")
    return SemanticAblationPlan(
        schema_version=SEMANTIC_ABLATION_PLAN_SCHEMA_VERSION,
        model_key=model,
        objective_key=config.identity.objective_key,
        baseline_run_name=baseline_run_name,
        objective_run_name=objective_run_name,
        same_manifest_family_required=config.ablation.same_manifest_family_required,
        same_splits_required=config.attachment.require_same_splits,
        same_validation_protocol_required=config.ablation.same_validation_protocol_required,
        same_seed_required=config.ablation.same_seed_required,
        status="incomplete" if issues else "planned",
        issues=tuple(issues),
    )


def _config(value: object) -> None:
    if not isinstance(value, SemanticConsistencyObjectiveConfig):
        raise SemanticConsistencyError("config must be a SemanticConsistencyObjectiveConfig.")


def _schema(value: str, expected: str, label: str) -> None:
    if value != expected:
        raise SemanticConsistencyError(f"{label} schema_version is unsupported: {value!r}.")


def _objective_key(value: ObjectiveKey | str) -> ObjectiveKey:
    try:
        key = ObjectiveKey(value)
    except (TypeError, ValueError) as exc:
        raise SemanticConsistencyError(f"unknown semantic objective_key: {value!r}.") from exc
    if key is not ObjectiveKey.SEMANTIC_CONSISTENCY:
        raise SemanticConsistencyError("objective_key must be 'semantic_consistency'.")
    return key


def _models(values: tuple[ModelKey, ...], label: str) -> tuple[ModelKey, ...]:
    resolved = tuple(_model(item, label) for item in values)
    if len(resolved) != len(set(resolved)):
        raise SemanticConsistencyError(f"{label} must not contain duplicates.")
    return resolved


def _model(value: ModelKey | str, label: str) -> ModelKey:
    try:
        return ModelKey(value)
    except (TypeError, ValueError) as exc:
        raise SemanticConsistencyError(f"{label} contains an unknown model key: {value!r}.") from exc


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SemanticConsistencyError(f"{label} must be non-empty.")


def _messages(
    values: tuple[str, ...],
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    values = tuple(values)
    if not allow_empty and not values:
        raise SemanticConsistencyError(f"{label} must be non-empty.")
    for value in values:
        _text(value, label)
    return values


__all__ = [
    "SEMANTIC_ABLATION_PLAN_SCHEMA_VERSION",
    "SEMANTIC_ATTACHMENT_DECISION_SCHEMA_VERSION",
    "SEMANTIC_ATTACHMENT_POLICY_SCHEMA_VERSION",
    "SemanticAblationPlan",
    "SemanticObjectiveAttachmentDecision",
    "SemanticObjectiveAttachmentPolicy",
    "build_semantic_ablation_plan",
    "build_semantic_attachment_policy",
    "decide_semantic_attachment",
]
