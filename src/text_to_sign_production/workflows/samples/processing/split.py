from __future__ import annotations

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    ProgressTaskHandle,
)
from text_to_sign_production.data.gates import (
    GatesConfig,
    ProcessingDecision,
    ProcessingStatus,
    evaluate_sample_processing,
    evaluate_unmatched_source,
)
from text_to_sign_production.data.pose import (
    FrameFileListing,
    PoseBuildOutput,
    discover_frame_files,
)
from text_to_sign_production.data.samples import DroppedManifestEntry, PassedManifestEntry
from text_to_sign_production.data.sources import (
    SourceCandidate,
    SourceMatchResult,
    load_translation_rows,
)
from text_to_sign_production.workflows.samples.constants import (
    SAMPLES_STAGE_DECISION_COMPUTE,
    SAMPLES_STAGE_POSE_BUILD,
    SAMPLES_STAGE_SOURCE_BUILD,
    SAMPLES_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.samples.contracts import (
    SamplesSplitRuntimeInputs,
    SamplesWorkflowConfig,
    SamplesWorkflowInvariantError,
)
from text_to_sign_production.workflows.samples.layout import SamplesLayout
from text_to_sign_production.workflows.samples.processing.manifests import (
    build_decision_dropped_entry,
    build_passed_manifest_entry,
    build_unmatched_dropped_entry,
)
from text_to_sign_production.workflows.samples.processing.models import (
    SamplesSourceBundle,
    SamplesSplitProcessingResult,
)
from text_to_sign_production.workflows.samples.processing.payloads import (
    build_dropped_payload_materialization,
    build_passed_payload_materialization,
    build_pose_output,
)
from text_to_sign_production.workflows.samples.processing.sources import (
    build_samples_source_bundle,
)


def process_samples_split(
    *,
    config: SamplesWorkflowConfig,
    split_inputs: SamplesSplitRuntimeInputs,
    layout: SamplesLayout,
    gates_config: GatesConfig,
    progress_session: ProgressSession | None = None,
) -> SamplesSplitProcessingResult:
    source_matches: list[SourceMatchResult] = []
    decisions: list[ProcessingDecision] = []
    passed_entries: list[PassedManifestEntry] = []
    dropped_entries: list[DroppedManifestEntry] = []
    translations = tuple(load_translation_rows(split_inputs.translation_csv_path))
    source_bundles: list[SamplesSourceBundle] = []

    source_total = len(translations)
    if progress_session is not None and source_total > 0:
        with progress_session.task(
            _split_progress_spec(
                stage_id=SAMPLES_STAGE_SOURCE_BUILD,
                label=f"source build [{split_inputs.split}]",
                unit="sample",
                operation_kind="source_build",
                total_semantics="translation rows in split",
            ),
            total=source_total,
        ) as source_progress:
            for translation in translations:
                source_bundles.append(
                    build_samples_source_bundle(
                        split_inputs=split_inputs,
                        translation=translation,
                    )
                )
                source_progress.advance()
    else:
        source_bundles = [
            build_samples_source_bundle(
                split_inputs=split_inputs,
                translation=translation,
            )
            for translation in translations
        ]

    source_matches.extend(source_bundle.match for source_bundle in source_bundles)
    frames_by_sample_id = _discover_matched_frames(source_bundles)
    pose_outputs = _build_pose_outputs(
        config=config,
        source_bundles=source_bundles,
        frames_by_sample_id=frames_by_sample_id,
        split=split_inputs.split,
        progress_session=progress_session,
    )

    decision_total = len(source_bundles)
    if progress_session is not None and decision_total > 0:
        with progress_session.task(
            _split_progress_spec(
                stage_id=SAMPLES_STAGE_DECISION_COMPUTE,
                label=f"decision/materialization [{split_inputs.split}]",
                unit="sample",
                operation_kind="decision_materialization",
                total_semantics="translation rows in split",
                allowed_counters=("passed", "dropped", "pass_rate", "drop_rate"),
            ),
            total=decision_total,
        ) as decision_progress:
            _compute_decisions_and_outputs(
                config=config,
                gates_config=gates_config,
                layout=layout,
                source_bundles=source_bundles,
                frames_by_sample_id=frames_by_sample_id,
                pose_outputs=pose_outputs,
                decisions=decisions,
                passed_entries=passed_entries,
                dropped_entries=dropped_entries,
                progress_task=decision_progress,
            )
    else:
        _compute_decisions_and_outputs(
            config=config,
            gates_config=gates_config,
            layout=layout,
            source_bundles=source_bundles,
            frames_by_sample_id=frames_by_sample_id,
            pose_outputs=pose_outputs,
            decisions=decisions,
            passed_entries=passed_entries,
            dropped_entries=dropped_entries,
            progress_task=None,
        )

    return SamplesSplitProcessingResult(
        split=split_inputs.split,
        source_matches=tuple(source_matches),
        decisions=tuple(decisions),
        passed_entries=tuple(passed_entries),
        dropped_entries=tuple(dropped_entries),
    )


def _compute_decisions_and_outputs(
    *,
    config: SamplesWorkflowConfig,
    gates_config: GatesConfig,
    layout: SamplesLayout,
    source_bundles: list[SamplesSourceBundle],
    frames_by_sample_id: dict[str, FrameFileListing],
    pose_outputs: dict[str, PoseBuildOutput],
    decisions: list[ProcessingDecision],
    passed_entries: list[PassedManifestEntry],
    dropped_entries: list[DroppedManifestEntry],
    progress_task: ProgressTaskHandle | None,
) -> None:
    decision_total = len(source_bundles)
    passed_count = 0
    dropped_count = 0
    for source_bundle in source_bundles:
        if not source_bundle.match.matched:
            decision = evaluate_unmatched_source(source_bundle.match)
            decisions.append(decision)
            dropped_entries.append(
                build_unmatched_dropped_entry(match=source_bundle.match, decision=decision)
            )
            dropped_count += 1
            if progress_task is not None:
                progress_task.advance(
                    counters=_decision_progress_counters(
                        passed=passed_count,
                        dropped=dropped_count,
                        total=decision_total,
                    )
                )
            continue

        candidate = _matched_candidate(source_bundle)
        frames = frames_by_sample_id[candidate.sample_id]
        pose_output = pose_outputs.get(candidate.sample_id)

        decision = evaluate_sample_processing(
            config=gates_config,
            candidate=candidate,
            frames_listing=frames,
            pose_output=pose_output,
        )
        decisions.append(decision)

        if decision.status is ProcessingStatus.PROCESSED:
            if pose_output is None:
                raise SamplesWorkflowInvariantError("Processed samples require pose output")
            payload_materialization = build_passed_payload_materialization(
                candidate=candidate,
                pose_output=pose_output,
                layout=layout,
            )
            passed_entries.append(
                build_passed_manifest_entry(
                    candidate=candidate,
                    payload_materialization=payload_materialization,
                )
            )
            passed_count += 1
        else:
            payload_materialization = build_dropped_payload_materialization(
                config=config,
                candidate=candidate,
                pose_output=pose_output,
                decision=decision,
                layout=layout,
            )
            dropped_entries.append(
                build_decision_dropped_entry(
                    candidate=candidate,
                    decision=decision,
                    materialization=payload_materialization,
                )
            )
            dropped_count += 1

        if progress_task is not None:
            progress_task.advance(
                counters=_decision_progress_counters(
                    passed=passed_count,
                    dropped=dropped_count,
                    total=decision_total,
                )
            )


def _discover_matched_frames(
    source_bundles: list[SamplesSourceBundle],
) -> dict[str, FrameFileListing]:
    frames_by_sample_id: dict[str, FrameFileListing] = {}
    for source_bundle in source_bundles:
        if not source_bundle.match.matched:
            continue
        candidate = _matched_candidate(source_bundle)
        frames_by_sample_id[candidate.sample_id] = discover_frame_files(candidate.keypoints_dir)
    return frames_by_sample_id


def _build_pose_outputs(
    *,
    config: SamplesWorkflowConfig,
    source_bundles: list[SamplesSourceBundle],
    frames_by_sample_id: dict[str, FrameFileListing],
    split: str,
    progress_session: ProgressSession | None,
) -> dict[str, PoseBuildOutput]:
    pose_jobs = []
    pose_total = 0
    for source_bundle in source_bundles:
        if not source_bundle.match.matched:
            continue
        candidate = _matched_candidate(source_bundle)
        frames = frames_by_sample_id[candidate.sample_id]
        if _pose_eligible(candidate, frames):
            pose_jobs.append((candidate, frames))
            pose_total += _pose_frame_step_total(candidate)

    pose_outputs: dict[str, PoseBuildOutput] = {}
    if progress_session is not None and pose_total > 0:
        with progress_session.task(
            _split_progress_spec(
                stage_id=SAMPLES_STAGE_POSE_BUILD,
                label=f"pose build [{split}]",
                unit="frame_step",
                operation_kind="pose_build",
                total_semantics="parse plus tensor frame steps for pose-eligible samples",
            ),
            total=pose_total,
        ) as pose_progress:
            for candidate, frames in pose_jobs:
                pose_outputs[candidate.sample_id] = build_pose_output(
                    candidate=candidate,
                    frames=frames,
                    person_selection_policy=config.person_selection_policy,
                    progress_task=pose_progress,
                )
    else:
        for candidate, frames in pose_jobs:
            pose_outputs[candidate.sample_id] = build_pose_output(
                candidate=candidate,
                frames=frames,
                person_selection_policy=config.person_selection_policy,
                progress_task=None,
            )
    return pose_outputs


def _matched_candidate(source_bundle: SamplesSourceBundle) -> SourceCandidate:
    candidate = source_bundle.candidate
    if candidate is None:
        raise SamplesWorkflowInvariantError("Matched source bundle must include a candidate")
    return candidate


def _pose_eligible(candidate: SourceCandidate, frames: FrameFileListing) -> bool:
    return candidate.structurally_viable and not frames.missing and frames.frame_count > 0


def _pose_frame_step_total(candidate: SourceCandidate) -> int:
    """Match data.pose progress semantics: parse + tensor steps per declared frame."""
    return 2 * candidate.frame_count


def _split_progress_spec(
    *,
    stage_id: str,
    label: str,
    unit: str,
    operation_kind: str,
    total_semantics: str,
    allowed_counters: tuple[str, ...] = (),
) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=SAMPLES_WORKFLOW_NAME,
        stage_id=stage_id,
        label=label,
        unit=unit,
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind=operation_kind,
        total_semantics=total_semantics,
        bar_eligible=True,
        allowed_counters=allowed_counters,
    )


def _decision_progress_counters(
    *,
    passed: int,
    dropped: int,
    total: int,
) -> dict[str, object]:
    return {
        "passed": passed,
        "dropped": dropped,
        "pass_rate": _format_percent(passed, total),
        "drop_rate": _format_percent(dropped, total),
    }


def _format_percent(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"
