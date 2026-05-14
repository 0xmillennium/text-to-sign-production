from __future__ import annotations

from text_to_sign_production.artifacts.store import sample_manifest_relative_path
from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    PassedManifestEntry,
)
from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
)
from text_to_sign_production.data.dataset.manifests import GATE_MANIFEST_SCHEMA_VERSION
from text_to_sign_production.data.dataset.types import (
    DatasetDroppedSampleProduction,
    DatasetPayloadProduction,
)
from text_to_sign_production.data.gate.policies import (
    GatesConfig,
)
from text_to_sign_production.data.gate.processing.evaluation import (
    evaluate_gate_source_bundle,
)
from text_to_sign_production.data.gate.processing.outputs import (
    GateSourceEvaluation,
    GateSourceOutputRefs,
)
from text_to_sign_production.data.gate.processing.sources import (
    GateSourceBundle,
    build_gate_source_bundle,
)
from text_to_sign_production.data.gate.sources.candidates import (
    assemble_candidate,
    sample_id_from_translation,
)
from text_to_sign_production.data.gate.sources.translations import load_translation_records
from text_to_sign_production.data.gate.sources.types import (
    TranslationSourceRecord,
)
from text_to_sign_production.data.gate.sources.validate import validate_candidate_set
from text_to_sign_production.workflows.gate.constants import (
    GATE_STAGE_DECISION_COMPUTE,
    GATE_STAGE_PAYLOAD_WRITE,
    GATE_STAGE_SOURCE_BUILD,
    GATE_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.gate.contracts import (
    GateSplitRuntimeInputs,
    GateWorkflowConfig,
)
from text_to_sign_production.workflows.gate.layout import GateLayout
from text_to_sign_production.workflows.gate.processing.manifests import (
    write_gate_payloads,
)
from text_to_sign_production.workflows.gate.processing.models import (
    GateDroppedSamplePayloadOutput,
    GatePayloadOutput,
    GateSplitProcessingResult,
)


def process_gate_split(
    *,
    config: GateWorkflowConfig,
    split_inputs: GateSplitRuntimeInputs,
    layout: GateLayout,
    gates_config: GatesConfig,
    progress_session: ProgressSession | None = None,
) -> GateSplitProcessingResult:
    translations = load_translation_records(split_inputs.translation_csv_path)
    source_bundles = _build_source_bundles(
        split_inputs=split_inputs,
        translations=translations,
        progress_session=progress_session,
    )
    _validate_source_bundle_identity_set(source_bundles)
    evaluations = _evaluate_source_bundles(
        config=config,
        layout=layout,
        gates_config=gates_config,
        source_bundles=source_bundles,
        progress_session=progress_session,
    )
    _validate_split_output_uniqueness(evaluations)
    _write_payload_outputs(evaluations, progress_session)
    return _build_split_processing_result(
        split=split_inputs.split,
        source_bundles=source_bundles,
        evaluations=evaluations,
    )


def _build_split_processing_result(
    *,
    split: str,
    source_bundles: list[GateSourceBundle],
    evaluations: tuple[GateSourceEvaluation, ...],
) -> GateSplitProcessingResult:
    return GateSplitProcessingResult(
        split=split,
        source_matches=tuple(bundle.match for bundle in source_bundles),
        viability_reports=tuple(
            evaluation.candidate_viability_report
            for evaluation in evaluations
            if evaluation.candidate_viability_report is not None
        ),
        prepared_samples=tuple(
            evaluation.prepared_sample
            for evaluation in evaluations
            if evaluation.prepared_sample is not None
        ),
        gate_bundles=tuple(
            evaluation.gate for evaluation in evaluations if evaluation.gate is not None
        ),
        passed_payloads=tuple(
            _workflow_payload_output(evaluation.passed_payload)
            for evaluation in evaluations
            if evaluation.passed_payload is not None
        ),
        dropped_sample_payloads=tuple(
            _workflow_dropped_payload_output(evaluation.dropped_sample_payload)
            for evaluation in evaluations
            if evaluation.dropped_sample_payload is not None
        ),
        passed_entries=tuple(
            evaluation.passed_entry
            for evaluation in evaluations
            if evaluation.passed_entry is not None
        ),
        dropped_entries=tuple(
            evaluation.dropped_entry
            for evaluation in evaluations
            if evaluation.dropped_entry is not None
        ),
    )


def _build_source_bundles(
    *,
    split_inputs: GateSplitRuntimeInputs,
    translations: tuple[TranslationSourceRecord, ...],
    progress_session: ProgressSession | None,
) -> list[GateSourceBundle]:
    source_bundles: list[GateSourceBundle] = []
    if progress_session is not None and translations:
        with progress_session.task(
            _split_progress_spec(
                stage_id=GATE_STAGE_SOURCE_BUILD,
                label=f"source build [{split_inputs.split}]",
                unit="sample",
                operation_kind="source_build",
                total_semantics="translation rows in split",
            ),
            total=len(translations),
        ) as source_progress:
            for translation in translations:
                source_bundles.append(
                    build_gate_source_bundle(
                        split=split_inputs.split,
                        translation=translation,
                        keypoint_json_root=split_inputs.keypoint_json_root,
                        keypoint_video_root=split_inputs.keypoint_video_root,
                    )
                )
                source_progress.advance()
    else:
        source_bundles = [
            build_gate_source_bundle(
                split=split_inputs.split,
                translation=translation,
                keypoint_json_root=split_inputs.keypoint_json_root,
                keypoint_video_root=split_inputs.keypoint_video_root,
            )
            for translation in translations
        ]
    return source_bundles


def _evaluate_source_bundles(
    *,
    config: GateWorkflowConfig,
    layout: GateLayout,
    gates_config: GatesConfig,
    source_bundles: list[GateSourceBundle],
    progress_session: ProgressSession | None,
) -> tuple[GateSourceEvaluation, ...]:
    total = len(source_bundles)
    evaluations: list[GateSourceEvaluation] = []
    passed_count = 0
    dropped_count = 0
    if progress_session is not None and total > 0:
        with progress_session.task(
            _split_progress_spec(
                stage_id=GATE_STAGE_DECISION_COMPUTE,
                label="prepared sample/gate decision",
                unit="sample",
                operation_kind="prepared_sample_gate_decision",
                total_semantics="translation rows in split",
                allowed_counters=("passed", "dropped", "pass_rate", "drop_rate"),
            ),
            total=total,
        ) as progress_task:
            for source_bundle in source_bundles:
                evaluation = _evaluate_one_source_bundle(
                    config=config,
                    layout=layout,
                    gates_config=gates_config,
                    source_bundle=source_bundle,
                )
                evaluations.append(evaluation)
                if evaluation.passed_entry is not None:
                    passed_count += 1
                if evaluation.dropped_entry is not None:
                    dropped_count += 1
                progress_task.advance(
                    counters=_decision_progress_counters(
                        passed=passed_count,
                        dropped=dropped_count,
                        total=total,
                    )
                )
    else:
        for source_bundle in source_bundles:
            evaluation = _evaluate_one_source_bundle(
                config=config,
                layout=layout,
                gates_config=gates_config,
                source_bundle=source_bundle,
            )
            evaluations.append(evaluation)
    return tuple(evaluations)


def _evaluate_one_source_bundle(
    *,
    config: GateWorkflowConfig,
    layout: GateLayout,
    gates_config: GatesConfig,
    source_bundle: GateSourceBundle,
) -> GateSourceEvaluation:
    output_refs = _source_output_refs(layout=layout, source_bundle=source_bundle)
    return evaluate_gate_source_bundle(
        source_bundle=source_bundle,
        gates_config=gates_config,
        person_selection_policy=config.person_selection_policy,
        output_refs=output_refs,
        manifest_schema_version=GATE_MANIFEST_SCHEMA_VERSION,
    )


def _write_payload_outputs(
    evaluations: tuple[GateSourceEvaluation, ...],
    progress_session: ProgressSession | None,
) -> None:
    prepared_payloads = tuple(
        payload
        for evaluation in evaluations
        for payload in (evaluation.passed_payload,)
        if payload is not None
    )
    dropped_payloads = tuple(
        evaluation.dropped_sample_payload
        for evaluation in evaluations
        if evaluation.dropped_sample_payload is not None
    )
    if progress_session is not None and prepared_payloads:
        with progress_session.task(
            _split_progress_spec(
                stage_id=GATE_STAGE_PAYLOAD_WRITE,
                label="write prepared sample payloads",
                unit="payload",
                operation_kind="prepared_sample_payload_write",
                total_semantics="planned prepared sample payload writes",
            ),
            total=len(prepared_payloads),
        ) as progress_task:
            for payload in prepared_payloads:
                write_gate_payloads(prepared_payloads=(payload,), dropped_sample_payloads=())
                progress_task.advance()
    else:
        write_gate_payloads(prepared_payloads=prepared_payloads, dropped_sample_payloads=())
    if progress_session is not None and dropped_payloads:
        with progress_session.task(
            _split_progress_spec(
                stage_id=GATE_STAGE_PAYLOAD_WRITE,
                label="write dropped sample payloads",
                unit="payload",
                operation_kind="dropped_sample_payload_write",
                total_semantics="planned dropped sample payload writes",
            ),
            total=len(dropped_payloads),
        ) as progress_task:
            for payload in dropped_payloads:
                write_gate_payloads(prepared_payloads=(), dropped_sample_payloads=(payload,))
                progress_task.advance()
    else:
        write_gate_payloads(prepared_payloads=(), dropped_sample_payloads=dropped_payloads)


def _validate_source_bundle_identity_set(source_bundles: list[GateSourceBundle]) -> None:
    candidates = tuple(
        assemble_candidate(bundle.match) for bundle in source_bundles if bundle.match.matched
    )
    issues = validate_candidate_set(candidates)
    if issues:
        raise ValueError(f"Invalid source candidate identity set: {issues}")


def _validate_split_output_uniqueness(evaluations: tuple[GateSourceEvaluation, ...]) -> None:
    payloads = tuple(
        payload
        for evaluation in evaluations
        for payload in (evaluation.passed_payload, evaluation.dropped_sample_payload)
        if payload is not None
    )
    _ensure_unique(
        tuple(str(payload.path) for payload in payloads),
        "payload paths",
    )
    _ensure_unique(
        tuple(payload.payload_ref for payload in payloads),
        "payload physical refs",
    )
    passed_entries = tuple(
        evaluation.passed_entry for evaluation in evaluations if evaluation.passed_entry is not None
    )
    dropped_entries = tuple(
        evaluation.dropped_entry
        for evaluation in evaluations
        if evaluation.dropped_entry is not None
    )
    manifest_entries: tuple[PassedManifestEntry | DroppedManifestEntry, ...] = (
        *passed_entries,
        *dropped_entries,
    )
    _ensure_unique(
        tuple((entry.split.value, entry.sample_id) for entry in passed_entries),
        "passed manifest physical sample ids",
    )
    _ensure_unique(
        tuple((entry.split.value, entry.sample_id) for entry in dropped_entries),
        "dropped manifest physical sample ids",
    )
    _ensure_unique(
        tuple((entry.split.value, entry.sample_id) for entry in manifest_entries),
        "manifest physical sample ids",
    )
    _ensure_unique(
        tuple(payload.payload_ref for payload in payloads),
        "archive member physical identities",
    )


def _ensure_unique(values: tuple[object, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label} are not allowed: {values}.")


def _source_output_refs(
    *,
    layout: GateLayout,
    source_bundle: GateSourceBundle,
) -> GateSourceOutputRefs:
    sample_id = _source_sample_id(source_bundle)
    split = source_bundle.match.split
    passed_ref = _payload_ref(SampleStatus.PASSED, split, sample_id)
    dropped_ref = _payload_ref(SampleStatus.DROPPED, split, sample_id)
    return GateSourceOutputRefs(
        passed_payload_path=layout.outputs.root / passed_ref,
        passed_payload_ref=passed_ref,
        dropped_payload_path=layout.outputs.root / dropped_ref,
        dropped_payload_ref=dropped_ref,
    )


def _payload_ref(status: SampleStatus, split: str, sample_id: str) -> str:
    return sample_manifest_relative_path(status, split, sample_id).as_posix()


def _workflow_dropped_payload_output(
    payload: DatasetDroppedSampleProduction,
) -> GateDroppedSamplePayloadOutput:
    return GateDroppedSamplePayloadOutput(
        sample=payload.sample,
        path=payload.path,
        payload_ref=payload.payload_ref,
    )


def _source_sample_id(source_bundle: GateSourceBundle) -> str:
    identity = source_bundle.match.candidate_identity
    if identity is not None:
        return identity.keypoint.sample_key.value
    return sample_id_from_translation(source_bundle.translation)


def _workflow_payload_output(production: DatasetPayloadProduction) -> GatePayloadOutput:
    return GatePayloadOutput(
        sample=production.sample,
        path=production.path,
        payload_ref=production.payload_ref,
        status=production.status,
    )


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
        workflow_id=GATE_WORKFLOW_NAME,
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
) -> dict[str, int | str]:
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


__all__ = ["process_gate_split"]
