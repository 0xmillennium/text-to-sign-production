from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from text_to_sign_production.data.gate.pose import (
    DEFAULT_PERSON_SELECTION_POLICY,
    PersonSelectionPolicy,
)

CANONICAL_GATES_CONFIG_RELPATH = Path("configs/data/gates.yaml")
CANONICAL_FILTERS_CONFIG_RELPATH = Path("configs/data/filters.yaml")
CANONICAL_TIERS_CONFIG_RELPATH = Path("configs/data/tiers.yaml")


class DebugWorkflowConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DebugWorkflowConfig:
    project_root: Path
    drive_project_root: Path
    runtime_root: Path
    gates_config_relpath: Path = CANONICAL_GATES_CONFIG_RELPATH
    filters_config_relpath: Path = CANONICAL_FILTERS_CONFIG_RELPATH
    tiers_config_relpath: Path = CANONICAL_TIERS_CONFIG_RELPATH
    person_selection_policy: PersonSelectionPolicy = DEFAULT_PERSON_SELECTION_POLICY
    debug_reports_relroot: Path = Path("reports/debug_samples")

    def __post_init__(self) -> None:
        for field_name in (
            "project_root",
            "drive_project_root",
            "runtime_root",
        ):
            object.__setattr__(self, field_name, _coerce_path(getattr(self, field_name)))
        object.__setattr__(
            self,
            "debug_reports_relroot",
            _coerce_relative_path(self.debug_reports_relroot, "debug_reports_relroot"),
        )
        for field_name in (
            "gates_config_relpath",
            "filters_config_relpath",
            "tiers_config_relpath",
        ):
            object.__setattr__(
                self,
                field_name,
                _coerce_relative_path(getattr(self, field_name), field_name),
            )
        _require_canonical_config_path(
            self.gates_config_relpath,
            CANONICAL_GATES_CONFIG_RELPATH,
            "gates_config_relpath",
        )
        _require_canonical_config_path(
            self.filters_config_relpath,
            CANONICAL_FILTERS_CONFIG_RELPATH,
            "filters_config_relpath",
        )
        _require_canonical_config_path(
            self.tiers_config_relpath,
            CANONICAL_TIERS_CONFIG_RELPATH,
            "tiers_config_relpath",
        )
        object.__setattr__(
            self,
            "person_selection_policy",
            _coerce_person_selection_policy(self.person_selection_policy),
        )

    @classmethod
    def from_roots(
        cls,
        *,
        project_root: str | os.PathLike[str],
        drive_project_root: str | os.PathLike[str],
        runtime_root: str | os.PathLike[str] | None = None,
    ) -> DebugWorkflowConfig:
        project = _coerce_path(project_root)
        drive = _coerce_path(drive_project_root)
        runtime = (
            _coerce_path(runtime_root)
            if runtime_root is not None
            else project / "runtime" / "debug"
        )
        return cls(
            project_root=project,
            drive_project_root=drive,
            runtime_root=runtime,
        )

    @property
    def configs_runtime_root(self) -> Path:
        return self.runtime_root / "provenance" / "config"

    @property
    def assets_runtime_root(self) -> Path:
        return self.runtime_root / "assets"

    @property
    def manifests_runtime_root(self) -> Path:
        return self.runtime_root / "manifests"

    @property
    def samples_runtime_root(self) -> Path:
        return self.runtime_root / "samples"

    @property
    def reports_runtime_root(self) -> Path:
        return self.runtime_root / "reports"

    @property
    def gates_config_runtime_path(self) -> Path:
        return self.configs_runtime_root / "gates.yaml"

    @property
    def filters_config_runtime_path(self) -> Path:
        return self.configs_runtime_root / "filters.yaml"

    @property
    def tiers_config_runtime_path(self) -> Path:
        return self.configs_runtime_root / "tiers.yaml"


def _coerce_path(value: object) -> Path:
    if value is None:
        raise DebugWorkflowConfigError("path values must not be None")
    if isinstance(value, str):
        raw_path = value
    elif isinstance(value, os.PathLike):
        raw_path = os.fspath(cast(os.PathLike[str], value))
    else:
        raise DebugWorkflowConfigError("path values must be strings or path-like objects")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise DebugWorkflowConfigError("path values must be non-empty text paths")
    return Path(raw_path)


def _coerce_relative_path(value: object, field_name: str) -> Path:
    path = _coerce_path(value)
    if path.is_absolute() or ".." in path.parts:
        raise DebugWorkflowConfigError(f"{field_name} must be a safe relative path")
    return path


def _require_canonical_config_path(path: Path, expected: Path, field_name: str) -> None:
    if path != expected:
        raise DebugWorkflowConfigError(f"{field_name} must be {expected.as_posix()}")


def _coerce_person_selection_policy(value: object) -> PersonSelectionPolicy:
    if isinstance(value, PersonSelectionPolicy):
        return value
    if isinstance(value, str):
        try:
            return PersonSelectionPolicy(value.strip())
        except ValueError as exc:
            allowed = ", ".join(policy.value for policy in PersonSelectionPolicy)
            raise DebugWorkflowConfigError(
                f"person_selection_policy must be one of: {allowed}"
            ) from exc
    raise DebugWorkflowConfigError(
        "person_selection_policy must be a PersonSelectionPolicy or string value"
    )


__all__ = [
    "CANONICAL_FILTERS_CONFIG_RELPATH",
    "CANONICAL_GATES_CONFIG_RELPATH",
    "CANONICAL_TIERS_CONFIG_RELPATH",
    "DebugWorkflowConfig",
    "DebugWorkflowConfigError",
]
