"""Strict checkpoint selection for test_model."""

from __future__ import annotations

from text_to_sign_production.workflows.test_model.contracts import (
    CheckpointPolicy,
    TestModelCheckpointSelection,
    TestModelRunResolution,
)
from text_to_sign_production.workflows.test_model.layout import TestModelLayout


def select_checkpoint(
    layout: TestModelLayout,
    model_run: TestModelRunResolution,
    policy: CheckpointPolicy,
) -> TestModelCheckpointSelection:
    role = "best" if policy is CheckpointPolicy.BEST else "last"
    checkpoint_name = f"{role}.pt"
    checkpoint_path = layout.stores.runtime.models.model_checkpoint_file(
        model_run.model_key,
        model_run.model_run_name,
        checkpoint_name,
    ).path
    errors: list[str] = []
    metadata_key = f"{role}_checkpoint_path"
    if not model_run.run_metadata.get(metadata_key):
        errors.append(
            f"checkpoint policy {policy.value!r} requires {metadata_key} in model run metadata"
        )
    if not checkpoint_path.is_file():
        errors.append(
            f"checkpoint policy {policy.value!r} requires restored checkpoint file: {checkpoint_path}"
        )
    return TestModelCheckpointSelection(
        policy=policy,
        checkpoint_role=role,
        checkpoint_path=checkpoint_path,
        checkpoint_exists=checkpoint_path.is_file(),
        warnings=(),
        errors=tuple(errors),
    )


__all__ = ["select_checkpoint"]
