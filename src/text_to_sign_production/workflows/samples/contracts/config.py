from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from text_to_sign_production.data.pose.people import DEFAULT_PERSON_SELECTION_POLICY
from text_to_sign_production.data.pose.types import PersonSelectionPolicy


class SamplesWorkflowError(Exception):
    pass


class SamplesWorkflowInputError(SamplesWorkflowError):
    pass


class SamplesWorkflowInvariantError(SamplesWorkflowError):
    pass


@dataclass(frozen=True, slots=True)
class SamplesWorkflowConfig:
    project_root: Path
    drive_project_root: Path
    splits: tuple[str, ...]
    gates_config_relpath: Path = Path("configs/data/gates.yaml")
    person_selection_policy: PersonSelectionPolicy = DEFAULT_PERSON_SELECTION_POLICY
    materialize_dropped_debug_payloads: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", _coerce_path(self.project_root))
        object.__setattr__(
            self,
            "drive_project_root",
            _coerce_path(self.drive_project_root),
        )
        object.__setattr__(
            self,
            "gates_config_relpath",
            _coerce_relative_path(self.gates_config_relpath),
        )
        object.__setattr__(
            self,
            "person_selection_policy",
            _coerce_person_selection_policy(self.person_selection_policy),
        )
        object.__setattr__(self, "splits", _coerce_splits(self.splits))
        object.__setattr__(
            self,
            "materialize_dropped_debug_payloads",
            _coerce_debug_payload_flag(self.materialize_dropped_debug_payloads),
        )


def _coerce_path(value: object) -> Path:
    if value is None:
        raise SamplesWorkflowInputError("path values must not be None")
    if isinstance(value, str):
        raw_path = value
    elif isinstance(value, os.PathLike):
        raw_path = os.fspath(cast(os.PathLike[str], value))
    else:
        raise SamplesWorkflowInputError("path values must be strings or path-like objects")
    if not isinstance(raw_path, str):
        raise SamplesWorkflowInputError("path values must be text paths")
    if not raw_path.strip():
        raise SamplesWorkflowInputError("path values must not be blank")
    return Path(raw_path)


def _coerce_relative_path(value: object) -> Path:
    path = _coerce_path(value)
    if path.is_absolute():
        raise SamplesWorkflowInputError("gates_config_relpath must be relative")
    return path


def _coerce_person_selection_policy(value: object) -> PersonSelectionPolicy:
    if isinstance(value, PersonSelectionPolicy):
        return value
    if isinstance(value, str):
        try:
            return PersonSelectionPolicy(value.strip())
        except ValueError as exc:
            allowed = ", ".join(policy.value for policy in PersonSelectionPolicy)
            raise SamplesWorkflowInputError(
                f"person_selection_policy must be one of: {allowed}"
            ) from exc
    raise SamplesWorkflowInputError(
        "person_selection_policy must be a PersonSelectionPolicy or string value"
    )


def _coerce_splits(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_splits = (value,)
    elif isinstance(value, Iterable):
        raw_splits = tuple(value)
    else:
        raise SamplesWorkflowInputError("splits must be an iterable of split names")
    splits = tuple(_coerce_split_item(split) for split in raw_splits)
    for split in splits:
        _validate_non_empty_split(split)
    if len(set(splits)) != len(splits):
        raise SamplesWorkflowInputError("splits must not contain duplicates")
    return splits


def _coerce_split_item(value: object) -> str:
    if not isinstance(value, str):
        raise SamplesWorkflowInputError("split names must be strings")
    return value.strip()


def _validate_non_empty_split(value: str) -> None:
    if not value:
        raise SamplesWorkflowInputError("split names must be non-empty")


def _coerce_debug_payload_flag(value: object) -> bool:
    if not isinstance(value, bool):
        raise SamplesWorkflowInputError("materialize_dropped_debug_payloads must be a bool")
    return value
