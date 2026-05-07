from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast


class TiersWorkflowError(Exception):
    pass


class TiersWorkflowInputError(TiersWorkflowError):
    pass


class TiersWorkflowInvariantError(TiersWorkflowError):
    pass


@dataclass(frozen=True, slots=True)
class TiersWorkflowConfig:
    project_root: Path
    drive_project_root: Path
    splits: tuple[str, ...]
    filters_config_relpath: Path = Path("configs/data/filters.yaml")
    tiers_config_relpath: Path = Path("configs/data/tiers.yaml")

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", _coerce_path(self.project_root))
        object.__setattr__(
            self,
            "drive_project_root",
            _coerce_path(self.drive_project_root),
        )
        object.__setattr__(
            self,
            "filters_config_relpath",
            _coerce_relative_path(
                "filters_config_relpath",
                self.filters_config_relpath,
            ),
        )
        object.__setattr__(
            self,
            "tiers_config_relpath",
            _coerce_relative_path(
                "tiers_config_relpath",
                self.tiers_config_relpath,
            ),
        )
        object.__setattr__(self, "splits", _coerce_splits(self.splits))


def _coerce_path(value: object) -> Path:
    if value is None:
        raise TiersWorkflowInputError("path values must not be None")
    if isinstance(value, str):
        raw_path = value
    elif isinstance(value, os.PathLike):
        raw_path = os.fspath(cast(os.PathLike[str], value))
    else:
        raise TiersWorkflowInputError("path values must be strings or path-like objects")
    if not isinstance(raw_path, str):
        raise TiersWorkflowInputError("path values must be text paths")
    if not raw_path.strip():
        raise TiersWorkflowInputError("path values must not be blank")
    return Path(raw_path)


def _coerce_relative_path(field_name: str, value: object) -> Path:
    path = _coerce_path(value)
    if path.is_absolute():
        raise TiersWorkflowInputError(f"{field_name} must be relative")
    return path


def _coerce_splits(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_splits = (value,)
    elif isinstance(value, Iterable):
        raw_splits = tuple(value)
    else:
        raise TiersWorkflowInputError("splits must be an iterable of split names")
    splits = tuple(_coerce_split_item(split) for split in raw_splits)
    for split in splits:
        _validate_non_empty_split(split)
    if len(set(splits)) != len(splits):
        raise TiersWorkflowInputError("splits must not contain duplicates")
    return splits


def _coerce_split_item(value: object) -> str:
    if not isinstance(value, str):
        raise TiersWorkflowInputError("split names must be strings")
    return value.strip()


def _validate_non_empty_split(value: str) -> None:
    if not value:
        raise TiersWorkflowInputError("split names must be non-empty")
