"""Resolve one target sample from the model run's test split."""

from __future__ import annotations

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.modeling.data import (
    load_manifest_sample,
    read_modeling_manifest,
)
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelRequest,
    TestModelRunResolution,
    TestModelTargetResolution,
)
from text_to_sign_production.workflows.test_model.layout import TestModelLayout
from text_to_sign_production.workflows.test_model.constants import TEST_MODEL_STAGE_TARGET_RESOLVE
from text_to_sign_production.workflows.test_model.progress import test_model_progress_stage


def resolve_target_sample(
    layout: TestModelLayout,
    request: TestModelRequest,
    model_run: TestModelRunResolution,
    *,
    progress_session: ProgressSession | None = None,
) -> TestModelTargetResolution:
    if progress_session is None:
        return _resolve_target_sample(layout, request, model_run)
    with progress_session.task(
        test_model_progress_stage(
            stage_id=TEST_MODEL_STAGE_TARGET_RESOLVE,
            label="test_model target resolve",
            unit="sample",
            owner_module=__name__,
            operation_kind="target_resolve",
            total_semantics="target sample lookup",
            allowed_counters=("found", "missing", "error"),
        ),
        total=1,
    ) as task:
        try:
            result = _resolve_target_sample(layout, request, model_run)
        except Exception:
            task.advance(1, counters={"error": 1})
            raise
        counters = {
            "found": 1 if result.status == "found" else 0,
            "missing": 1 if result.status in {"missing", "ambiguous"} else 0,
            "error": 1 if result.status == "error" else 0,
        }
        task.advance(1, counters=counters)
        return result


def _resolve_target_sample(
    layout: TestModelLayout,
    request: TestModelRequest,
    model_run: TestModelRunResolution,
) -> TestModelTargetResolution:
    issues: list[str] = []
    warnings: list[str] = []
    manifest_sample = None
    entry = None
    if model_run.manifest_family is None:
        return TestModelTargetResolution(
            status="error",
            target_sentence_name=request.target_sentence_name,
            resolved_sample_id=None,
            source_sentence_name=None,
            manifest_family=None,
            split=SampleSplit.TEST,
            manifest_path=layout.stores.runtime.manifests_root,
            manifest_entry=None,
            manifest_sample=None,
            issues=("model run manifest_family was not resolved",),
            warnings=(),
        )
    try:
        manifest = read_modeling_manifest(
            layout.stores.runtime,
            model_run.manifest_family,
            SampleSplit.TEST,
        )
    except (OSError, ValueError, TypeError) as exc:
        return TestModelTargetResolution(
            status="error",
            target_sentence_name=request.target_sentence_name,
            resolved_sample_id=None,
            source_sentence_name=None,
            manifest_family=model_run.manifest_family,
            split=SampleSplit.TEST,
            manifest_path=layout.stores.runtime.manifests_root,
            manifest_entry=None,
            manifest_sample=None,
            issues=(str(exc),),
            warnings=(),
        )
    matches = tuple(
        candidate
        for candidate in manifest.entries
        if candidate.source_sentence_name == request.target_sentence_name
    )
    if not matches:
        issues.append(
            "TARGET_SENTENCE_NAME was not found as an exact source_sentence_name in the "
            "model run test split manifest. TARGET_SENTENCE_NAME must match "
            "source_sentence_name from that manifest, not sample_id."
        )
        status = "missing"
    elif len(matches) > 1:
        issues.append(
            "TARGET_SENTENCE_NAME matched multiple source_sentence_name rows in the test "
            "split manifest; select an unambiguous sentence name."
        )
        status = "ambiguous"
    else:
        status = "found"
        entry = matches[0]
        try:
            manifest_sample = load_manifest_sample(
                layout.stores.runtime,
                model_run.manifest_family,
                manifest.manifest_path,
                entry,
            )
        except (OSError, ValueError, TypeError) as exc:
            status = "error"
            issues.append(f"prepared payload could not be loaded: {exc}")
    return TestModelTargetResolution(
        status=status,
        target_sentence_name=request.target_sentence_name,
        resolved_sample_id=None if entry is None else entry.sample_id,
        source_sentence_name=None if entry is None else entry.source_sentence_name,
        manifest_family=model_run.manifest_family,
        split=SampleSplit.TEST,
        manifest_path=manifest.manifest_path,
        manifest_entry=entry,
        manifest_sample=manifest_sample,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


__all__ = ["resolve_target_sample"]
