from __future__ import annotations

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    GateDecisionBundle,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
)
from text_to_sign_production.data.gate.pose import (
    FrameFileListing,
    PersonSelectionPolicy,
    PoseBuildOutput,
    build_pose_input,
    build_pose_output,
    build_tracking_result,
    discover_frame_files,
    parse_frame_file,
)
from text_to_sign_production.data.dataset.build import build_prepared_sample
from text_to_sign_production.data.gate.policies import (
    GatesConfig,
    evaluate_sample_gates,
)
from text_to_sign_production.data.gate.sources import (
    SourceCandidate,
    TranslationSourceRecord,
    assemble_candidate,
    load_translation_records,
)
from text_to_sign_production.workflows.samples.constants import (
    SAMPLES_STAGE_DECISION_COMPUTE,
    SAMPLES_STAGE_SOURCE_BUILD,
    SAMPLES_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.samples.contracts import (
    SamplesSplitRuntimeInputs,
    SamplesWorkflowConfig,
)
from text_to_sign_production.workflows.samples.layout import SamplesLayout
from text_to_sign_production.workflows.samples.processing.manifests import (
    build_candidate_pose_dropped_entry,
    build_gate_dropped_entry,
    build_passed_workflow_entry,
    build_unmatched_dropped_entry,
)
from text_to_sign_production.workflows.samples.processing.models import (
    SamplesPayloadOutput,
    SamplesSourceBundle,
    SamplesSplitProcessingResult,
)
from text_to_sign_production.workflows.samples.processing.payloads import (
    write_prepared_sample_workflow_payload,
)
from text_to_sign_production.workflows.samples.processing.sources import (
    build_samples_source_bundle,
)

SAMPLE_SCHEMA_VERSION = "prepared_sample.v1"


def process_samples_split(
    *,
    config: SamplesWorkflowConfig,
    split_inputs: SamplesSplitRuntimeInputs,
    layout: SamplesLayout,
    gates_config: GatesConfig,
    progress_session: ProgressSession | None = None,
) -> SamplesSplitProcessingResult:
    translations = load_translation_records(
        split_inputs.translation_csv_path,
        canonical_text_column=config.translation_canonical_text_column,
    )
    source_bundles = _build_source_bundles(
        split_inputs=split_inputs,
        translations=translations,
        progress_session=progress_session,
    )
    source_matches = tuple(bundle.match for bundle in source_bundles)

    prepared_samples: list[PreparedSample] = []
    gate_bundles: list[GateDecisionBundle] = []
    passed_payloads: list[SamplesPayloadOutput] = []
    dropped_debug_payloads: list[SamplesPayloadOutput] = []
    passed_entries: list[PassedManifestEntry] = []
    dropped_entries: list[DroppedManifestEntry] = []

    _process_source_bundles(
        config=config,
        layout=layout,
        gates_config=gates_config,
        source_bundles=source_bundles,
        prepared_samples=prepared_samples,
        gate_bundles=gate_bundles,
        passed_payloads=passed_payloads,
        dropped_debug_payloads=dropped_debug_payloads,
        passed_entries=passed_entries,
        dropped_entries=dropped_entries,
        progress_session=progress_session,
    )
    return SamplesSplitProcessingResult(
        split=split_inputs.split,
        source_matches=source_matches,
        prepared_samples=tuple(prepared_samples),
        gate_bundles=tuple(gate_bundles),
        passed_payloads=tuple(passed_payloads),
        dropped_debug_payloads=tuple(dropped_debug_payloads),
        passed_entries=tuple(passed_entries),
        dropped_entries=tuple(dropped_entries),
    )


def _build_source_bundles(
    *,
    split_inputs: SamplesSplitRuntimeInputs,
    translations: tuple[TranslationSourceRecord, ...],
    progress_session: ProgressSession | None,
) -> list[SamplesSourceBundle]:
    source_bundles: list[SamplesSourceBundle] = []
    if progress_session is not None and translations:
        with progress_session.task(
            _split_progress_spec(
                stage_id=SAMPLES_STAGE_SOURCE_BUILD,
                label=f"source build [{split_inputs.split}]",
                unit="sample",
                operation_kind="source_build",
                total_semantics="translation rows in split",
            ),
            total=len(translations),
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
    return source_bundles


def _process_source_bundles(
    *,
    config: SamplesWorkflowConfig,
    layout: SamplesLayout,
    gates_config: GatesConfig,
    source_bundles: list[SamplesSourceBundle],
    prepared_samples: list[PreparedSample],
    gate_bundles: list[GateDecisionBundle],
    passed_payloads: list[SamplesPayloadOutput],
    dropped_debug_payloads: list[SamplesPayloadOutput],
    passed_entries: list[PassedManifestEntry],
    dropped_entries: list[DroppedManifestEntry],
    progress_session: ProgressSession | None,
) -> None:
    total = len(source_bundles)
    if progress_session is not None and total > 0:
        with progress_session.task(
            _split_progress_spec(
                stage_id=SAMPLES_STAGE_DECISION_COMPUTE,
                label="prepared sample/gate/write",
                unit="sample",
                operation_kind="prepared_sample_gate_write",
                total_semantics="translation rows in split",
                allowed_counters=("passed", "dropped", "pass_rate", "drop_rate"),
            ),
            total=total,
        ) as progress_task:
            for source_bundle in source_bundles:
                _process_one_source_bundle(
                    config=config,
                    layout=layout,
                    gates_config=gates_config,
                    source_bundle=source_bundle,
                    prepared_samples=prepared_samples,
                    gate_bundles=gate_bundles,
                    passed_payloads=passed_payloads,
                    dropped_debug_payloads=dropped_debug_payloads,
                    passed_entries=passed_entries,
                    dropped_entries=dropped_entries,
                )
                progress_task.advance(
                    counters=_decision_progress_counters(
                        passed=len(passed_entries),
                        dropped=len(dropped_entries),
                        total=total,
                    )
                )
    else:
        for source_bundle in source_bundles:
            _process_one_source_bundle(
                config=config,
                layout=layout,
                gates_config=gates_config,
                source_bundle=source_bundle,
                prepared_samples=prepared_samples,
                gate_bundles=gate_bundles,
                passed_payloads=passed_payloads,
                dropped_debug_payloads=dropped_debug_payloads,
                passed_entries=passed_entries,
                dropped_entries=dropped_entries,
            )


def _process_one_source_bundle(
    *,
    config: SamplesWorkflowConfig,
    layout: SamplesLayout,
    gates_config: GatesConfig,
    source_bundle: SamplesSourceBundle,
    prepared_samples: list[PreparedSample],
    gate_bundles: list[GateDecisionBundle],
    passed_payloads: list[SamplesPayloadOutput],
    dropped_debug_payloads: list[SamplesPayloadOutput],
    passed_entries: list[PassedManifestEntry],
    dropped_entries: list[DroppedManifestEntry],
) -> None:
    if not source_bundle.match.matched:
        dropped_entries.append(build_unmatched_dropped_entry(source_bundle.match))
        return

    candidate = assemble_candidate(source_bundle.match)
    pose_output = _build_pose_output(
        candidate=candidate,
        person_selection_policy=config.person_selection_policy,
    )
    if pose_output is None:
        dropped_entries.append(build_candidate_pose_dropped_entry(candidate=candidate))
        return

    sample = build_prepared_sample(
        candidate,
        pose_output,
        schema_version=SAMPLE_SCHEMA_VERSION,
    )
    prepared_samples.append(sample)
    gate = evaluate_sample_gates(sample, gates_config)
    gate_bundles.append(gate)

    if gate.final_status is SampleStatus.PASSED:
        payload = write_prepared_sample_workflow_payload(
            sample=sample,
            status=SampleStatus.PASSED,
            layout=layout,
        )
        passed_payloads.append(payload)
        passed_entries.append(
            build_passed_workflow_entry(sample=sample, gate=gate, payload=payload)
        )
        return

    debug_payload = _maybe_write_dropped_debug_payload(config=config, layout=layout, sample=sample)
    if debug_payload is not None:
        dropped_debug_payloads.append(debug_payload)
    dropped_entries.append(
        build_gate_dropped_entry(
            sample=sample,
            debug_ref=None if debug_payload is None else debug_payload.payload_ref,
        )
    )


def _build_pose_output(
    *,
    candidate: SourceCandidate,
    person_selection_policy: PersonSelectionPolicy,
) -> PoseBuildOutput | None:
    frame_listing = discover_frame_files(candidate)
    if not _pose_eligible(candidate, frame_listing):
        return None
    parsed_frames = tuple(
        parse_frame_file(path, frame_index=index) for index, path in enumerate(frame_listing.files)
    )
    tracking = build_tracking_result(parsed_frames, person_selection_policy)
    return build_pose_output(
        build_pose_input(
            candidate=candidate,
            frame_listing=frame_listing,
            parsed_frames=parsed_frames,
            tracking=tracking,
        )
    )


def _maybe_write_dropped_debug_payload(
    *,
    config: SamplesWorkflowConfig,
    layout: SamplesLayout,
    sample: PreparedSample,
) -> SamplesPayloadOutput | None:
    if not config.materialize_dropped_debug_payloads:
        return None
    return write_prepared_sample_workflow_payload(
        sample=sample,
        status=SampleStatus.DROPPED,
        layout=layout,
    )


def _pose_eligible(candidate: SourceCandidate, frames: FrameFileListing) -> bool:
    return candidate.structurally_viable and not frames.missing and frames.frame_count > 0


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


__all__ = ["SAMPLE_SCHEMA_VERSION", "process_samples_split"]
