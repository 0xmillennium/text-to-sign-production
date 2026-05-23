"""Operator request contract for one selected model checkpoint and test sample."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.test_model.contracts.config import (
    TestModelWorkflowInputError,
)


class CheckpointPolicy(enum.StrEnum):
    BEST = "best"
    LAST = "last"


@dataclass(frozen=True, slots=True)
class TestModelRequest:
    model_run_name: str
    checkpoint_policy: CheckpointPolicy
    target_sentence_name: str

    def __post_init__(self) -> None:
        _validate_safe_token("MODEL_RUN_NAME", self.model_run_name)
        try:
            object.__setattr__(
                self,
                "checkpoint_policy",
                CheckpointPolicy(self.checkpoint_policy),
            )
        except (TypeError, ValueError) as exc:
            raise TestModelWorkflowInputError(
                "CHECKPOINT_POLICY must be 'best' or 'last'"
            ) from exc
        if not isinstance(self.target_sentence_name, str) or not self.target_sentence_name.strip():
            raise TestModelWorkflowInputError("TARGET_SENTENCE_NAME must be non-empty")
        object.__setattr__(self, "target_sentence_name", self.target_sentence_name.strip())


def _validate_safe_token(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TestModelWorkflowInputError(f"{field_name} must be non-empty")
    token = value.strip()
    if token in {".", ".."} or "/" in token or "\\" in token or Path(token).name != token:
        raise TestModelWorkflowInputError(f"{field_name} must be a safe path token")


__all__ = ["CheckpointPolicy", "TestModelRequest"]
