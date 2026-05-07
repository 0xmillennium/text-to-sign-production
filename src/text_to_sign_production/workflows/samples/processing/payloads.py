from __future__ import annotations

from text_to_sign_production.artifacts.store import sample_manifest_relative_path
from text_to_sign_production.core.progress import ProgressTaskHandle
from text_to_sign_production.data.gates import ProcessingDecision
from text_to_sign_production.data.pose import (
    FrameFileListing,
    PersonSelectionPolicy,
    PoseBuildInput,
    PoseBuildOutput,
    PoseProgressEvent,
    PoseProgressSink,
    build_pose_tensors,
)
from text_to_sign_production.data.samples import (
    DroppedDebugMaterializationOutcome,
    ProcessedSamplePayload,
    SampleStatus,
    build_dropped_materialization_lifecycle,
    build_payload,
    validate_payload,
    write_processed_sample_payload,
)
from text_to_sign_production.data.sources import SourceCandidate
from text_to_sign_production.workflows.samples.contracts import (
    SamplesWorkflowConfig,
    SamplesWorkflowInvariantError,
)
from text_to_sign_production.workflows.samples.layout import SamplesLayout
from text_to_sign_production.workflows.samples.processing.models import (
    SamplesPayloadMaterialization,
)


def build_pose_output(
    *,
    candidate: SourceCandidate,
    frames: FrameFileListing,
    person_selection_policy: PersonSelectionPolicy,
    progress_task: ProgressTaskHandle | None = None,
) -> PoseBuildOutput:
    return build_pose_tensors(
        PoseBuildInput(
            candidate=candidate,
            frames=frames,
            person_selection_policy=person_selection_policy,
        ),
        progress_sink=(_PoseProgressAdapter(progress_task) if progress_task is not None else None),
    )


def build_passed_payload_materialization(
    *,
    candidate: SourceCandidate,
    pose_output: PoseBuildOutput,
    layout: SamplesLayout,
) -> SamplesPayloadMaterialization:
    relative_path = sample_manifest_relative_path(
        SampleStatus.PASSED,
        candidate.split,
        candidate.sample_id,
    )
    payload_path = layout.outputs.root / relative_path
    payload = _build_payload(candidate, pose_output)
    _raise_payload_contract_issues(payload)
    write_processed_sample_payload(payload_path, payload)
    return SamplesPayloadMaterialization(
        payload=payload,
        payload_path=payload_path,
        payload_relative_path=relative_path.as_posix(),
        dropped_materialization=None,
    )


def build_dropped_payload_materialization(
    *,
    config: SamplesWorkflowConfig,
    candidate: SourceCandidate,
    pose_output: PoseBuildOutput | None,
    decision: ProcessingDecision,
    layout: SamplesLayout,
) -> SamplesPayloadMaterialization:
    if not decision.can_materialize_debug:
        return SamplesPayloadMaterialization(
            payload=None,
            payload_path=None,
            payload_relative_path=None,
            dropped_materialization=build_dropped_materialization_lifecycle(
                debug_materialization_eligible=False,
                debug_materialization_attempted=False,
                debug_materialization_outcome=DroppedDebugMaterializationOutcome.NOT_ATTEMPTED,
            ),
            not_attempted_reason=None,
        )

    if not config.materialize_dropped_debug_payloads:
        return SamplesPayloadMaterialization(
            payload=None,
            payload_path=None,
            payload_relative_path=None,
            dropped_materialization=build_dropped_materialization_lifecycle(
                debug_materialization_eligible=True,
                debug_materialization_attempted=False,
                debug_materialization_outcome=DroppedDebugMaterializationOutcome.NOT_ATTEMPTED,
            ),
            not_attempted_reason="config_disabled",
        )

    if pose_output is None:
        return SamplesPayloadMaterialization(
            payload=None,
            payload_path=None,
            payload_relative_path=None,
            dropped_materialization=build_dropped_materialization_lifecycle(
                debug_materialization_eligible=True,
                debug_materialization_attempted=False,
                debug_materialization_outcome=DroppedDebugMaterializationOutcome.NOT_ATTEMPTED,
            ),
            not_attempted_reason="missing_pose_output",
        )

    return _attempt_dropped_payload_materialization(
        candidate=candidate,
        pose_output=pose_output,
        layout=layout,
    )


def _attempt_dropped_payload_materialization(
    *,
    candidate: SourceCandidate,
    pose_output: PoseBuildOutput,
    layout: SamplesLayout,
) -> SamplesPayloadMaterialization:
    relative_path = sample_manifest_relative_path(
        SampleStatus.DROPPED,
        candidate.split,
        candidate.sample_id,
    )
    payload_path = layout.outputs.root / relative_path
    try:
        payload = _build_payload(candidate, pose_output)
        _raise_payload_contract_issues(payload)
        write_processed_sample_payload(payload_path, payload)
    except Exception as exc:
        return SamplesPayloadMaterialization(
            payload=None,
            payload_path=None,
            payload_relative_path=None,
            dropped_materialization=build_dropped_materialization_lifecycle(
                debug_materialization_eligible=True,
                debug_materialization_attempted=True,
                debug_materialization_outcome=DroppedDebugMaterializationOutcome.FAILED,
                failure_reason=str(exc),
            ),
            not_attempted_reason=None,
        )

    return SamplesPayloadMaterialization(
        payload=payload,
        payload_path=payload_path,
        payload_relative_path=relative_path.as_posix(),
        dropped_materialization=build_dropped_materialization_lifecycle(
            debug_materialization_eligible=True,
            debug_materialization_attempted=True,
            debug_materialization_outcome=DroppedDebugMaterializationOutcome.SUCCEEDED,
            payload_path=relative_path.as_posix(),
            payload_exists=True,
            archive_publishable=True,
        ),
        not_attempted_reason=None,
    )


def _build_payload(
    candidate: SourceCandidate,
    pose_output: PoseBuildOutput,
) -> ProcessedSamplePayload:
    return build_payload(
        sample_id=candidate.sample_id,
        text=candidate.text,
        split=candidate.split,
        num_frames=candidate.frame_count,
        fps=candidate.video_metadata.fps,
        selected_person=pose_output.selected_person,
        frame_quality=pose_output.frame_quality,
        pose=pose_output.pose,
        people_per_frame=pose_output.people_per_frame,
        frame_valid_mask=pose_output.frame_valid_mask,
    )


def _raise_payload_contract_issues(payload: ProcessedSamplePayload) -> None:
    issues = validate_payload(payload)
    if issues:
        details = ", ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise SamplesWorkflowInvariantError(f"Invalid sample payload: {details}")


class _PoseProgressAdapter(PoseProgressSink):
    """Adapt cumulative data.pose events into the task owned by split.py."""

    def __init__(self, progress_task: ProgressTaskHandle) -> None:
        self._progress_task = progress_task
        self._completed_by_sample: dict[str, int] = {}

    def update_pose_progress(self, event: PoseProgressEvent) -> None:
        sample_id = str(event.sample_id)
        completed = int(event.completed)
        previous = self._completed_by_sample.get(sample_id, 0)
        delta = completed - previous
        if delta > 0:
            self._progress_task.advance(delta)
        self._completed_by_sample[sample_id] = max(previous, completed)
