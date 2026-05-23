"""Collect a debug-style evidence dossier for one test sample."""

from __future__ import annotations

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.test_model.constants import TEST_MODEL_STAGE_EVIDENCE_COLLECT
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelCheckpointSelection,
    TestModelRunResolution,
    TestModelSampleEvidence,
    TestModelTargetResolution,
)
from text_to_sign_production.workflows.test_model.layout import TestModelLayout
from text_to_sign_production.workflows.test_model.progress import test_model_progress_stage


def collect_sample_evidence(
    layout: TestModelLayout,
    *,
    model_run: TestModelRunResolution,
    checkpoint: TestModelCheckpointSelection,
    target: TestModelTargetResolution,
    progress_session: ProgressSession | None = None,
) -> TestModelSampleEvidence:
    if progress_session is None:
        return _collect_sample_evidence(
            layout,
            model_run=model_run,
            checkpoint=checkpoint,
            target=target,
        )
    with progress_session.task(
        test_model_progress_stage(
            stage_id=TEST_MODEL_STAGE_EVIDENCE_COLLECT,
            label="test_model evidence collect",
            unit="sample",
            owner_module=__name__,
            operation_kind="evidence_collect",
            total_semantics="target sample evidence collection",
            allowed_counters=("source_video", "prepared_payload"),
        ),
        total=1,
    ) as task:
        result = _collect_sample_evidence(
            layout,
            model_run=model_run,
            checkpoint=checkpoint,
            target=target,
        )
        task.advance(
            1,
            counters={
                "source_video": 1 if result.source_video_exists else 0,
                "prepared_payload": 1 if result.prepared_payload_exists else 0,
            },
        )
        return result


def _collect_sample_evidence(
    layout: TestModelLayout,
    *,
    model_run: TestModelRunResolution,
    checkpoint: TestModelCheckpointSelection,
    target: TestModelTargetResolution,
) -> TestModelSampleEvidence:
    errors = list(target.issues)
    warnings = list(target.warnings)
    entry = target.manifest_entry
    sentence_name = "" if entry is None else entry.source_sentence_name
    source_video_path = (
        layout.stores.runtime.assets.keypoint_video_dir(SampleSplit.TEST).path
        / f"{sentence_name}.mp4"
    )
    prepared_payload_path = None if target.manifest_sample is None else target.manifest_sample.payload_path
    notes = [
        "Target resolution is constrained to the test split.",
        f"Model key was resolved from model run metadata: {model_run.model_key}.",
        "Manifest family was resolved from model run metadata: "
        f"{None if model_run.manifest_family is None else model_run.manifest_family.family_id}.",
        f"Checkpoint policy selected without fallback: {checkpoint.policy.value}.",
    ]
    if source_video_path.is_file():
        notes.append("Raw source video is available for side-by-side visualization.")
    else:
        warnings.append(f"raw source video is missing: {source_video_path}")
        notes.append("Raw source video is missing; side-by-side visualization is not ready.")
    return TestModelSampleEvidence(
        target=target,
        source_video_path=source_video_path,
        source_video_exists=source_video_path.is_file(),
        prepared_payload_path=prepared_payload_path,
        prepared_payload_exists=prepared_payload_path.is_file() if prepared_payload_path else False,
        model_context={
            "model_run_name": model_run.model_run_name,
            "model_key": model_run.model_key,
            "manifest_family": (
                None
                if model_run.manifest_family is None
                else model_run.manifest_family.family_id
            ),
            "train_split": model_run.train_split.value,
            "validation_split": model_run.validation_split.value,
            "test_split": model_run.test_split.value,
            "run_mode": model_run.run_mode,
        },
        checkpoint_context={
            "policy": checkpoint.policy.value,
            "checkpoint_role": checkpoint.checkpoint_role,
            "checkpoint_path": str(checkpoint.checkpoint_path),
            "checkpoint_exists": checkpoint.checkpoint_exists,
        },
        consistency_notes=tuple(notes),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


__all__ = ["collect_sample_evidence"]
