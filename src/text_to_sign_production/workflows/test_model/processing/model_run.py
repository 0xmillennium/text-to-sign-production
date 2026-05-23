"""Resolve model run metadata for a single-sample model test."""

from __future__ import annotations

import json
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelRestorePlan,
    TestModelRunResolution,
)


def resolve_model_run(plan: TestModelRestorePlan) -> TestModelRunResolution:
    metadata_path = _target_for_group(plan, "model_run_metadata")
    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, object] = {}
    if metadata_path is None:
        errors.append("model run metadata was not included in the restore plan")
        metadata_path = Path(".")
    elif not metadata_path.is_file():
        errors.append(f"model run metadata is missing from runtime: {metadata_path}")
    else:
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"model run metadata is unreadable: {exc}")
        else:
            if isinstance(loaded, dict):
                metadata = loaded
            else:
                errors.append("model run metadata root must be a JSON object")
    status = str(metadata.get("status", "unknown"))
    if status != "completed":
        errors.append(
            f"model run status must be completed before test_model; observed {status!r}"
        )
    test_split_value = str(metadata.get("test_split", "test"))
    try:
        train_split = SampleSplit(str(metadata.get("train_split", "train")))
        validation_split = SampleSplit(str(metadata.get("validation_split", "val")))
        test_split = SampleSplit(test_split_value)
    except (TypeError, ValueError) as exc:
        errors.append(f"model run split metadata is invalid: {exc}")
        train_split = SampleSplit.TRAIN
        validation_split = SampleSplit.VAL
        test_split = SampleSplit.TEST
    if test_split is not SampleSplit.TEST:
        errors.append(f"test_model requires test_split='test'; observed {test_split.value!r}")
    for field_name in ("model_key", "manifest_family", "run_mode"):
        if not isinstance(metadata.get(field_name), str) or not str(metadata.get(field_name)).strip():
            errors.append(f"model run metadata is missing {field_name}")
    manifest_family = None
    try:
        manifest_family = parse_modeling_manifest_family(
            str(metadata.get("manifest_family", ""))
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"model run manifest_family metadata is invalid: {exc}")
    model_key = str(metadata.get("model_key", ""))
    if model_key:
        try:
            ensure_model_provider_registered(model_key)
        except (TypeError, ValueError) as exc:
            errors.append(f"model provider resolution failed: {exc}")
    return TestModelRunResolution(
        model_run_name=str(metadata.get("model_run_name") or metadata.get("run_name") or plan.model_run_name),
        model_key=model_key,
        manifest_family=manifest_family,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        run_mode=str(metadata.get("run_mode", "")),
        run_metadata_path=metadata_path,
        run_metadata=metadata,
        status=status,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _target_for_group(plan: TestModelRestorePlan, group: str) -> Path | None:
    for operation in plan.operations:
        if operation.group == group:
            return operation.target
    return None


__all__ = ["resolve_model_run"]
