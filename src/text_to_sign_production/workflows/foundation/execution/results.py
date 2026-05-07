from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.foundation.execution.contracts import (
    OperationKind,
    OperationProgressSpec,
)


@dataclass(frozen=True, slots=True)
class RenderedShellCommand:
    label: str
    operation_kind: OperationKind
    shell_script: str
    display_command: str
    failure_message: str
    progress: OperationProgressSpec | None = None
    expected_outputs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("shell_script", self.shell_script)
        _validate_non_empty_text("display_command", self.display_command)
        _validate_non_empty_text("failure_message", self.failure_message)


@dataclass(frozen=True, slots=True)
class ShellExecutionResult:
    shell_script: str
    returncode: int
    succeeded: bool
    execution_mode: str
    stdout_captured: bool = False
    stderr_captured: bool = False

    def __post_init__(self) -> None:
        _validate_non_empty_text("execution_mode", self.execution_mode)


@dataclass(frozen=True, slots=True)
class ExecutionFailureDetail:
    label: str
    operation_kind: OperationKind
    failure_message: str
    returncode: int
    execution_mode: str

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("failure_message", self.failure_message)
        _validate_non_empty_text("execution_mode", self.execution_mode)


@dataclass(frozen=True, slots=True)
class OperationExecutionResult:
    label: str
    operation_kind: OperationKind
    succeeded: bool
    returncode: int
    failure: ExecutionFailureDetail | None
    execution_mode: str
    progress_stage_id: str | None
    expected_outputs: tuple[Path, ...]
    observed_outputs: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.failure is None and not self.succeeded:
            raise ValueError("succeeded must be True when failure is None")
        if self.failure is not None and self.succeeded:
            raise ValueError("succeeded must be False when failure is not None")
        _validate_non_empty_text("execution_mode", self.execution_mode)


@dataclass(frozen=True, slots=True)
class OperationBatchExecutionResult:
    results: tuple[OperationExecutionResult, ...]

    @property
    def succeeded(self) -> bool:
        return all(result.succeeded for result in self.results)

    def failed_results(self) -> tuple[OperationExecutionResult, ...]:
        return tuple(result for result in self.results if not result.succeeded)

    def successful_results(self) -> tuple[OperationExecutionResult, ...]:
        return tuple(result for result in self.results if result.succeeded)


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
