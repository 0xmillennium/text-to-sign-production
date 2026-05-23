"""Readiness contracts for future semantic-objective ablation comparison."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticConsistencyObjectiveConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey

SEMANTIC_ABLATION_RUN_REFERENCE_SCHEMA_VERSION = "t2sp-semantic-ablation-run-reference-v1"
SEMANTIC_ABLATION_READINESS_SCHEMA_VERSION = "t2sp-semantic-ablation-readiness-v1"
SEMANTIC_VALIDATION_PROTOCOL = "final_validation_generated_pose_proxy_v1"
_CAVEATS = (
    "Semantic alignment metrics are post-generation proxy measurements only.",
    "Readiness does not mean an ablation comparison has been executed or completed.",
    "No semantic objective contribution claim is supported without compatible baseline evidence.",
    "Proxy semantic metrics do not prove sign intelligibility or linguistic correctness.",
)


@dataclass(frozen=True, slots=True)
class SemanticAblationRunReference:
    schema_version: str
    run_name: str
    model_key: ModelKey
    manifest_family: str
    auxiliary_objectives: tuple[ObjectiveKey, ...]
    evaluated_splits: tuple[SampleSplit, ...]
    validation_protocol: str
    run_metadata_path: Path | None

    def __post_init__(self) -> None:
        _schema(self.schema_version, SEMANTIC_ABLATION_RUN_REFERENCE_SCHEMA_VERSION, "run reference")
        _text(self.run_name, "run_name")
        _text(self.manifest_family, "manifest_family")
        _text(self.validation_protocol, "validation_protocol")
        try:
            object.__setattr__(self, "model_key", ModelKey(self.model_key))
            objectives = tuple(ObjectiveKey(value) for value in self.auxiliary_objectives)
            evaluated_splits = tuple(SampleSplit(value) for value in self.evaluated_splits)
        except (TypeError, ValueError) as exc:
            raise SemanticConsistencyError("semantic ablation run reference has an invalid enum value.") from exc
        if len(set(objectives)) != len(objectives):
            raise SemanticConsistencyError("semantic ablation run auxiliary_objectives must be unique.")
        if not evaluated_splits:
            raise SemanticConsistencyError("semantic ablation run evaluated_splits must be non-empty.")
        if len(set(evaluated_splits)) != len(evaluated_splits):
            raise SemanticConsistencyError("semantic ablation run evaluated_splits must be unique.")
        object.__setattr__(self, "auxiliary_objectives", objectives)
        object.__setattr__(self, "evaluated_splits", evaluated_splits)
        if self.run_metadata_path is not None:
            object.__setattr__(self, "run_metadata_path", Path(self.run_metadata_path))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "model_key": self.model_key.value,
            "manifest_family": self.manifest_family,
            "auxiliary_objectives": [objective.value for objective in self.auxiliary_objectives],
            "evaluated_splits": [split.value for split in self.evaluated_splits],
            "validation_protocol": self.validation_protocol,
            "run_metadata_path": (
                None if self.run_metadata_path is None else str(self.run_metadata_path)
            ),
        }


@dataclass(frozen=True, slots=True)
class SemanticAblationReadinessResult:
    schema_version: str
    objective_key: ObjectiveKey
    model_key: ModelKey
    objective_run_name: str
    baseline_run_name: str | None
    ready_for_comparison: bool
    required_baseline_missing: bool
    same_model_required: bool
    same_manifest_family_required: bool
    same_splits_required: bool
    same_validation_protocol_required: bool
    issues: tuple[str, ...]
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        _schema(self.schema_version, SEMANTIC_ABLATION_READINESS_SCHEMA_VERSION, "readiness")
        try:
            object.__setattr__(self, "objective_key", ObjectiveKey(self.objective_key))
            object.__setattr__(self, "model_key", ModelKey(self.model_key))
        except (TypeError, ValueError) as exc:
            raise SemanticConsistencyError("semantic ablation readiness has an invalid enum value.") from exc
        if self.objective_key is not ObjectiveKey.SEMANTIC_CONSISTENCY:
            raise SemanticConsistencyError("semantic ablation readiness objective_key must be semantic_consistency.")
        _text(self.objective_run_name, "objective_run_name")
        if self.baseline_run_name is not None:
            _text(self.baseline_run_name, "baseline_run_name")
        for name in (
            "ready_for_comparison",
            "required_baseline_missing",
            "same_model_required",
            "same_manifest_family_required",
            "same_splits_required",
            "same_validation_protocol_required",
        ):
            if not isinstance(getattr(self, name), bool):
                raise SemanticConsistencyError(f"{name} must be a boolean.")
        issues = _messages(self.issues, "issues", allow_empty=True)
        caveats = _messages(self.caveats, "caveats")
        if self.required_baseline_missing and self.baseline_run_name is not None:
            raise SemanticConsistencyError("required_baseline_missing cannot be true when a baseline is named.")
        if self.ready_for_comparison and (self.required_baseline_missing or issues):
            raise SemanticConsistencyError("ready semantic ablation comparison cannot contain blocking issues.")
        if not self.ready_for_comparison and not issues:
            raise SemanticConsistencyError("not-ready semantic ablation comparison must describe blocking issues.")
        for caveat in _CAVEATS:
            if caveat not in caveats:
                raise SemanticConsistencyError(f"semantic ablation readiness caveats must include: {caveat}")
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "caveats", caveats)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "objective_key": self.objective_key.value,
            "model_key": self.model_key.value,
            "objective_run_name": self.objective_run_name,
            "baseline_run_name": self.baseline_run_name,
            "ready_for_comparison": self.ready_for_comparison,
            "required_baseline_missing": self.required_baseline_missing,
            "same_model_required": self.same_model_required,
            "same_manifest_family_required": self.same_manifest_family_required,
            "same_splits_required": self.same_splits_required,
            "same_validation_protocol_required": self.same_validation_protocol_required,
            "issues": list(self.issues),
            "caveats": list(self.caveats),
        }


def evaluate_semantic_ablation_readiness(
    *,
    objective_run: SemanticAblationRunReference,
    baseline_run: SemanticAblationRunReference | None,
    config: SemanticConsistencyObjectiveConfig,
) -> SemanticAblationReadinessResult:
    """Evaluate whether a compatible without-objective run is available to compare."""

    if not isinstance(objective_run, SemanticAblationRunReference):
        raise SemanticConsistencyError("objective_run must be a SemanticAblationRunReference.")
    if baseline_run is not None and not isinstance(baseline_run, SemanticAblationRunReference):
        raise SemanticConsistencyError("baseline_run must be a SemanticAblationRunReference or None.")
    if not isinstance(config, SemanticConsistencyObjectiveConfig):
        raise SemanticConsistencyError("config must be a SemanticConsistencyObjectiveConfig.")
    issues: list[str] = []
    if ObjectiveKey.SEMANTIC_CONSISTENCY not in objective_run.auxiliary_objectives:
        issues.append("Objective run must include semantic_consistency in auxiliary_objectives.")
    issues.extend(_metadata_issues(reference=objective_run, role="objective"))
    if baseline_run is None:
        issues.append(
            "A baseline run without semantic_consistency is required before semantic ablation comparison."
        )
    else:
        issues.extend(_metadata_issues(reference=baseline_run, role="baseline"))
        if ObjectiveKey.SEMANTIC_CONSISTENCY in baseline_run.auxiliary_objectives:
            issues.append("Baseline run must not include semantic_consistency in auxiliary_objectives.")
        if baseline_run.model_key is not objective_run.model_key:
            issues.append("Baseline and objective runs must use the same model_key.")
        if (
            config.ablation.same_manifest_family_required
            and baseline_run.manifest_family != objective_run.manifest_family
        ):
            issues.append("Baseline and objective runs must use the same manifest_family.")
        if (
            config.attachment.require_same_splits
            and baseline_run.evaluated_splits != objective_run.evaluated_splits
        ):
            issues.append("Baseline and objective runs must use the same evaluated_splits.")
        if (
            config.ablation.same_validation_protocol_required
            and baseline_run.validation_protocol != objective_run.validation_protocol
        ):
            issues.append("Baseline and objective runs must use the same validation_protocol.")
    return SemanticAblationReadinessResult(
        schema_version=SEMANTIC_ABLATION_READINESS_SCHEMA_VERSION,
        objective_key=ObjectiveKey.SEMANTIC_CONSISTENCY,
        model_key=objective_run.model_key,
        objective_run_name=objective_run.run_name,
        baseline_run_name=None if baseline_run is None else baseline_run.run_name,
        ready_for_comparison=not issues,
        required_baseline_missing=baseline_run is None,
        same_model_required=True,
        same_manifest_family_required=config.ablation.same_manifest_family_required,
        same_splits_required=config.attachment.require_same_splits,
        same_validation_protocol_required=config.ablation.same_validation_protocol_required,
        issues=tuple(issues),
        caveats=_CAVEATS,
    )


def _metadata_issues(
    *,
    reference: SemanticAblationRunReference,
    role: str,
) -> tuple[str, ...]:
    if role not in {"baseline", "objective"}:
        raise SemanticConsistencyError("semantic ablation metadata role must be baseline or objective.")
    path = reference.run_metadata_path
    if path is None:
        return (f"{role.capitalize()} run metadata path is required for ablation readiness.",)
    if not path.is_file():
        return (
            f"{role.capitalize()} run metadata is missing or unreadable at {path}; "
            "publish/materialize compatible model run metadata before ablation comparison.",
        )
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (f"{role.capitalize()} run metadata JSON is unreadable at {path}: {exc}",)
    if not isinstance(loaded, dict):
        return (f"{role.capitalize()} run metadata root must be a JSON object: {path}",)
    if not loaded:
        return (
            f"{role.capitalize()} run metadata is empty at {path}; materialized metadata "
            "must include run identity, model_key, manifest_family, auxiliary_objectives, "
            "status, and evaluated split evidence.",
        )
    issues: list[str] = []
    metadata_run_name = loaded.get("model_run_name") or loaded.get("run_name")
    if not isinstance(metadata_run_name, str) or not metadata_run_name.strip():
        issues.append(
            f"{role.capitalize()} run metadata must include non-empty run_name or model_run_name."
        )
    elif metadata_run_name != reference.run_name:
        issues.append(
            f"{role.capitalize()} run metadata run_name {metadata_run_name!r} does not "
            f"match reference {reference.run_name!r}."
        )
    metadata_model_key = loaded.get("model_key")
    if not isinstance(metadata_model_key, str) or not metadata_model_key.strip():
        issues.append(f"{role.capitalize()} run metadata must include non-empty model_key.")
    elif metadata_model_key != reference.model_key.value:
        issues.append(
            f"{role.capitalize()} run metadata model_key {metadata_model_key!r} does not "
            f"match reference {reference.model_key.value!r}."
        )
    metadata_manifest_family = loaded.get("manifest_family")
    if not isinstance(metadata_manifest_family, str) or not metadata_manifest_family.strip():
        issues.append(
            f"{role.capitalize()} run metadata must include non-empty manifest_family."
        )
    elif metadata_manifest_family != reference.manifest_family:
        issues.append(
            f"{role.capitalize()} run metadata manifest_family "
            f"{metadata_manifest_family!r} does not match reference "
            f"{reference.manifest_family!r}."
        )
    metadata_objectives = loaded.get("auxiliary_objectives")
    if not isinstance(metadata_objectives, list) or not all(
        isinstance(item, str) and item.strip() for item in metadata_objectives
    ):
        issues.append(
            f"{role.capitalize()} run metadata must include auxiliary_objectives as "
            "a list of non-empty strings."
        )
    else:
        expected = tuple(objective.value for objective in reference.auxiliary_objectives)
        observed = tuple(metadata_objectives)
        if observed != expected:
            issues.append(
                f"{role.capitalize()} run metadata auxiliary_objectives {observed!r} "
                f"do not match reference {expected!r}."
            )
        has_semantic = ObjectiveKey.SEMANTIC_CONSISTENCY.value in observed
        if role == "objective" and not has_semantic:
            issues.append(
                "Objective run metadata auxiliary_objectives must include semantic_consistency."
            )
        if role == "baseline" and has_semantic:
            issues.append(
                "Baseline run metadata auxiliary_objectives must not include semantic_consistency."
            )
    metadata_splits = _metadata_evaluated_splits(loaded)
    if metadata_splits is None:
        issues.append(
            f"{role.capitalize()} run metadata must include evaluated_splits, "
            "validation_split, or prediction_splits with valid split values."
        )
    elif metadata_splits != reference.evaluated_splits:
        issues.append(
            f"{role.capitalize()} run metadata evaluated splits "
            f"{tuple(split.value for split in metadata_splits)!r} do not match reference "
            f"{tuple(split.value for split in reference.evaluated_splits)!r}."
        )
    status = loaded.get("status")
    if not isinstance(status, str) or not status.strip():
        issues.append(f"{role.capitalize()} run metadata must include non-empty status.")
    elif status != "completed":
        issues.append(
            f"{role.capitalize()} run metadata status must be 'completed'; observed {status!r}."
        )
    return tuple(issues)


def _metadata_evaluated_splits(metadata: dict[str, object]) -> tuple[SampleSplit, ...] | None:
    raw_splits = metadata.get("evaluated_splits")
    if isinstance(raw_splits, list) and all(isinstance(item, str) for item in raw_splits):
        try:
            return tuple(SampleSplit(item) for item in raw_splits)
        except ValueError:
            return None
    raw_splits = metadata.get("prediction_splits")
    if isinstance(raw_splits, list) and all(isinstance(item, str) for item in raw_splits):
        try:
            return tuple(SampleSplit(item) for item in raw_splits)
        except ValueError:
            return None
    validation_split = metadata.get("validation_split")
    if isinstance(validation_split, str):
        try:
            return (SampleSplit(validation_split),)
        except ValueError:
            return None
    return None


def _schema(value: str, expected: str, label: str) -> None:
    if value != expected:
        raise SemanticConsistencyError(
            f"semantic ablation {label} schema_version is unsupported: {value!r}."
        )


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SemanticConsistencyError(f"{label} must be non-empty.")


def _messages(values: tuple[str, ...], label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    resolved = tuple(values)
    if not allow_empty and not resolved:
        raise SemanticConsistencyError(f"{label} must be non-empty.")
    for value in resolved:
        _text(value, label)
    return resolved


__all__ = [
    "SEMANTIC_ABLATION_READINESS_SCHEMA_VERSION",
    "SEMANTIC_ABLATION_RUN_REFERENCE_SCHEMA_VERSION",
    "SEMANTIC_VALIDATION_PROTOCOL",
    "SemanticAblationReadinessResult",
    "SemanticAblationRunReference",
    "evaluate_semantic_ablation_readiness",
]
