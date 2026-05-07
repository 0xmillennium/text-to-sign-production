from __future__ import annotations

from pathlib import Path
from typing import Any

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    ProgressTaskHandle,
)
from text_to_sign_production.data.gates import ProcessingDecision
from text_to_sign_production.data.samples import (
    DroppedDebugMaterializationOutcome,
    DroppedManifestEntry,
    ManifestWriteProgressEvent,
    ManifestWriteProgressSink,
    PassedManifestEntry,
    build_dropped_entry,
    build_dropped_materialization_lifecycle,
    build_passed_entry,
    validate_manifest_entry,
    write_manifest_jsonl,
)
from text_to_sign_production.data.sources import (
    SourceCandidate,
    SourceMatchResult,
    sample_id_from_translation,
)
from text_to_sign_production.workflows.samples.constants import (
    SAMPLES_STAGE_MANIFEST_WRITE,
    SAMPLES_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.samples.contracts import SamplesWorkflowInvariantError
from text_to_sign_production.workflows.samples.layout import SamplesLayout
from text_to_sign_production.workflows.samples.processing.models import (
    SamplesExecutionBundle,
    SamplesPayloadMaterialization,
    SamplesSplitProcessingResult,
)


def build_unmatched_dropped_entry(
    *,
    match: SourceMatchResult,
    decision: ProcessingDecision,
) -> DroppedManifestEntry:
    entry = build_dropped_entry(
        sample_id=sample_id_from_translation(match.translation),
        split=match.split,
        drop_stage=decision.drop_stage.value if decision.drop_stage is not None else "source",
        drop_reasons=decision.drop_reasons,
        materialization=build_dropped_materialization_lifecycle(
            debug_materialization_eligible=False,
            debug_materialization_attempted=False,
            debug_materialization_outcome=DroppedDebugMaterializationOutcome.NOT_ATTEMPTED,
        ),
        drop_details={"unmatched_reason": match.unmatched_reason or "unmatched_source"},
        text=match.translation.text,
    )
    _raise_manifest_contract_issues(entry)
    return entry


def build_decision_dropped_entry(
    *,
    candidate: SourceCandidate,
    decision: ProcessingDecision,
    materialization: SamplesPayloadMaterialization,
) -> DroppedManifestEntry:
    if materialization.dropped_materialization is None:
        raise SamplesWorkflowInvariantError(
            "Dropped manifest entries require materialization facts"
        )

    drop_details: dict[str, Any] = {
        "gate_statuses": {
            stage.value: result.status.value
            for stage, result in decision.gate_results.items()
        }
    }
    if materialization.not_attempted_reason is not None:
        drop_details[
            "debug_materialization_not_attempted_reason"
        ] = materialization.not_attempted_reason

    payload = materialization.payload
    entry = build_dropped_entry(
        sample_id=candidate.sample_id,
        split=candidate.split,
        drop_stage=decision.drop_stage.value if decision.drop_stage is not None else "dropped",
        drop_reasons=decision.drop_reasons if decision.drop_reasons else ("dropped",),
        materialization=materialization.dropped_materialization,
        drop_details=drop_details,
        text=payload.text if payload is not None else None,
        num_frames=payload.num_frames if payload is not None else None,
        fps=payload.fps if payload is not None else None,
        selected_person=payload.selected_person if payload is not None else None,
        frame_quality=payload.frame_quality if payload is not None else None,
    )
    _raise_manifest_contract_issues(entry)
    return entry


def build_passed_manifest_entry(
    *,
    candidate: SourceCandidate,
    payload_materialization: SamplesPayloadMaterialization,
) -> PassedManifestEntry:
    payload = payload_materialization.payload
    if payload is None:
        raise SamplesWorkflowInvariantError("Passed manifest entries require a payload")
    if payload_materialization.payload_relative_path is None:
        raise SamplesWorkflowInvariantError("Passed manifest entries require a payload path")

    entry = build_passed_entry(
        sample_id=candidate.sample_id,
        text=candidate.text,
        split=candidate.split,
        num_frames=candidate.frame_count,
        fps=candidate.video_metadata.fps,
        sample_path=payload_materialization.payload_relative_path,
        source_video_id=candidate.video_id,
        source_sentence_id=candidate.sentence_id,
        source_sentence_name=candidate.sentence_name,
        selected_person=payload.selected_person,
        frame_quality=payload.frame_quality,
    )
    _raise_manifest_contract_issues(entry)
    return entry


def write_samples_manifests(
    *,
    layout: SamplesLayout,
    execution_bundle: SamplesExecutionBundle,
    progress_session: ProgressSession | None = None,
) -> None:
    for split_result in _split_results_by_config_order(execution_bundle):
        passed_path = _manifest_path(layout, "passed", split_result.split)
        dropped_path = _manifest_path(layout, "dropped", split_result.split)
        total_entries = len(split_result.passed_entries) + len(split_result.dropped_entries)
        if progress_session is not None and total_entries > 0:
            with progress_session.task(
                _manifest_progress_spec(split_result.split),
                total=total_entries,
            ) as progress_task:
                progress_adapter = _ManifestWriteProgressAdapter(progress_task)
                write_manifest_jsonl(
                    passed_path,
                    split_result.passed_entries,
                    progress_sink=progress_adapter,
                )
                write_manifest_jsonl(
                    dropped_path,
                    split_result.dropped_entries,
                    progress_sink=progress_adapter,
                )
        else:
            write_manifest_jsonl(passed_path, split_result.passed_entries)
            write_manifest_jsonl(dropped_path, split_result.dropped_entries)


def _split_results_by_config_order(
    execution_bundle: SamplesExecutionBundle,
) -> tuple[SamplesSplitProcessingResult, ...]:
    by_split = {result.split: result for result in execution_bundle.split_results}
    ordered_results = []
    for split in execution_bundle.workflow_result.config.splits:
        result = by_split.get(split)
        if result is None:
            raise SamplesWorkflowInvariantError(f"Missing split processing result: {split}")
        ordered_results.append(result)
    if len(by_split) != len(execution_bundle.split_results):
        raise SamplesWorkflowInvariantError("Duplicate split processing results are not allowed")
    return tuple(ordered_results)


def _manifest_path(layout: SamplesLayout, partition: str, split: str) -> Path:
    for manifest in layout.outputs.manifest_outputs:
        if manifest.partition == partition and manifest.split == split:
            return manifest.path
    raise SamplesWorkflowInvariantError(
        f"Missing {partition} manifest output path for split: {split}"
    )


def _raise_manifest_contract_issues(
    entry: PassedManifestEntry | DroppedManifestEntry,
) -> None:
    issues = validate_manifest_entry(entry)
    if issues:
        details = ", ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise SamplesWorkflowInvariantError(f"Invalid sample manifest entry: {details}")


def _manifest_progress_spec(split: str) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=SAMPLES_WORKFLOW_NAME,
        stage_id=SAMPLES_STAGE_MANIFEST_WRITE,
        label=f"manifest write [{split}]",
        unit="entry",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="manifest_write",
        total_semantics="passed plus dropped manifest entries for split",
        bar_eligible=True,
    )


class _ManifestWriteProgressAdapter(ManifestWriteProgressSink):
    def __init__(self, progress_task: ProgressTaskHandle) -> None:
        self._progress_task = progress_task
        self._completed_by_path: dict[Path, int] = {}

    def update_manifest_write_progress(self, event: ManifestWriteProgressEvent) -> None:
        path = Path(event.path)
        completed = int(event.completed)
        previous = self._completed_by_path.get(path, 0)
        delta = completed - previous
        if delta > 0:
            self._progress_task.advance(delta)
        self._completed_by_path[path] = max(previous, completed)
