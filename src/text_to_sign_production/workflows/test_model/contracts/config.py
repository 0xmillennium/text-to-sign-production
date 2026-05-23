"""Configuration contract for the single-sample test-model workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast


class TestModelWorkflowError(Exception):
    """Base error for test-model workflow failures."""


class TestModelWorkflowInputError(TestModelWorkflowError):
    """Raised when operator input is invalid."""


class TestModelWorkflowInvariantError(TestModelWorkflowError):
    """Raised when a workflow invariant is violated."""


@dataclass(frozen=True, slots=True)
class TestModelWorkflowConfig:
    project_root: Path
    drive_project_root: Path
    runtime_root: Path | None = None

    def __post_init__(self) -> None:
        project_root = _coerce_path("project_root", self.project_root)
        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(
            self,
            "drive_project_root",
            _coerce_path("drive_project_root", self.drive_project_root),
        )
        runtime_root = (
            project_root / "runtime" / "test_model"
            if self.runtime_root is None
            else _coerce_path("runtime_root", self.runtime_root)
        )
        object.__setattr__(self, "runtime_root", runtime_root)


def _coerce_path(field_name: str, value: object) -> Path:
    if isinstance(value, str):
        raw_path = value
    elif isinstance(value, os.PathLike):
        raw_path = os.fspath(cast(os.PathLike[str], value))
    else:
        raise TestModelWorkflowInputError(f"{field_name} must be a path-like value")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise TestModelWorkflowInputError(f"{field_name} must be a non-empty path")
    return Path(raw_path)


__all__ = [
    "TestModelWorkflowConfig",
    "TestModelWorkflowError",
    "TestModelWorkflowInputError",
    "TestModelWorkflowInvariantError",
]
