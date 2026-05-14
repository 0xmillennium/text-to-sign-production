from __future__ import annotations

from text_to_sign_production.artifacts.store import sample_manifest_relative_path
from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.data.dataset.manifests import GATE_MANIFEST_SCHEMA_VERSION
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.gate.policies import load_gates_config
from text_to_sign_production.data.gate.processing.evaluation import evaluate_gate_source_bundle
from text_to_sign_production.data.gate.processing.outputs import GateSourceOutputRefs
from text_to_sign_production.data.gate.processing.sources import build_gate_source_bundle
from text_to_sign_production.data.gate.sources.types import TranslationSourceRecord
from text_to_sign_production.workflows.debug.contracts import (
    DebugGateComparison,
    DebugGateFrameMetrics,
    DebugGatePreflight,
    DebugGateResult,
    DebugSampleDossier,
    DebugWorkflowConfig,
    ExistingGateStatus,
)
from text_to_sign_production.workflows.debug.contracts.verdicts import DebugGateVerdict
from text_to_sign_production.workflows.debug.constants import DEBUG_STAGE_GATE_RECOMPUTE
from text_to_sign_production.workflows.debug.layout import (
    DebugLayout,
    runtime_sample_payload_path,
)
from text_to_sign_production.workflows.debug.progress import debug_progress_stage


def debug_gate(
    config: DebugWorkflowConfig,
    layout: DebugLayout,
    dossier: DebugSampleDossier,
    *,
    progress_session: ProgressSession | None = None,
) -> DebugGateResult:
    if progress_session is not None:
        with progress_session.task(_gate_recompute_progress_spec(), total=1) as progress_task:
            result = _debug_gate_impl(config, layout, dossier)
            progress_task.advance()
            return result
    return _debug_gate_impl(config, layout, dossier)


def _debug_gate_impl(
    config: DebugWorkflowConfig,
    layout: DebugLayout,
    dossier: DebugSampleDossier,
) -> DebugGateResult:
    preflight = _preflight(config, dossier)
    if not preflight.can_compute:
        return DebugGateResult(
            preflight=preflight,
            existing_gate=dossier.existing_gate,
            existing_status=dossier.existing_gate.status,
            evaluation=None,
            computed_status=None,
            computed_gate=None,
            computed_sample=None,
            computed_passed_entry=None,
            computed_dropped_entry=None,
            frame_metrics=DebugGateFrameMetrics(None, None, None, None, None, None),
            comparison=DebugGateComparison(False, None, (), preflight.warnings),
            verdict=DebugGateVerdict.COMPUTE_FAILED,
            errors=preflight.errors,
            warnings=preflight.warnings,
        )
    try:
        gates_config = load_gates_config(config.gates_config_runtime_path)
        translation = _translation_record(dossier)
        source_bundle = build_gate_source_bundle(
            split=dossier.translation.split.value,
            translation=translation,
            keypoint_json_root=layout.runtime.assets.keypoint_json_dir(
                dossier.translation.split
            ).path,
            keypoint_video_root=layout.runtime.assets.keypoint_video_dir(
                dossier.translation.split
            ).path,
        )
        evaluation = evaluate_gate_source_bundle(
            source_bundle=source_bundle,
            gates_config=gates_config,
            person_selection_policy=config.person_selection_policy,
            output_refs=_output_refs(
                layout,
                dossier.translation.split,
                dossier.translation.sentence_name,
            ),
            manifest_schema_version=GATE_MANIFEST_SCHEMA_VERSION,
        )
    except (OSError, TypeError, ValueError) as exc:
        return DebugGateResult(
            preflight=preflight,
            existing_gate=dossier.existing_gate,
            existing_status=dossier.existing_gate.status,
            evaluation=None,
            computed_status=None,
            computed_gate=None,
            computed_sample=None,
            computed_passed_entry=None,
            computed_dropped_entry=None,
            frame_metrics=DebugGateFrameMetrics(None, None, None, None, None, None),
            comparison=DebugGateComparison(False, None, (), (str(exc),)),
            verdict=DebugGateVerdict.COMPUTE_FAILED,
            errors=(str(exc),),
            warnings=preflight.warnings,
        )
    computed_status = (
        SampleStatus.PASSED if evaluation.passed_entry is not None else SampleStatus.DROPPED
    )
    comparison = _compare_gate(dossier, computed_status, evaluation)
    verdict = _gate_verdict(dossier.existing_gate.status, comparison)
    return DebugGateResult(
        preflight=preflight,
        existing_gate=dossier.existing_gate,
        existing_status=dossier.existing_gate.status,
        evaluation=evaluation,
        computed_status=computed_status,
        computed_gate=evaluation.gate,
        computed_sample=evaluation.prepared_sample,
        computed_passed_entry=evaluation.passed_entry,
        computed_dropped_entry=evaluation.dropped_entry,
        frame_metrics=_frame_metrics(evaluation),
        comparison=comparison,
        verdict=verdict,
        errors=(),
        warnings=preflight.warnings,
    )


def _preflight(config: DebugWorkflowConfig, dossier: DebugSampleDossier) -> DebugGatePreflight:
    errors: list[str] = []
    if dossier.translation is None:
        errors.append("resolved translation row is unavailable")
    if not dossier.source.keypoint_json_dir_exists:
        errors.append(f"keypoint JSON directory is missing: {dossier.source.keypoint_json_dir}")
    if not config.gates_config_runtime_path.is_file():
        errors.append(f"gate config is missing: {config.gates_config_runtime_path}")
    return DebugGatePreflight(
        can_compute=not errors,
        errors=tuple(errors),
        warnings=(f"person selection policy: {config.person_selection_policy.value}",),
    )


def _translation_record(dossier: DebugSampleDossier) -> TranslationSourceRecord:
    row = dossier.translation
    return TranslationSourceRecord(
        video_id=row.video_id,
        video_name=row.video_name,
        sentence_id=row.sentence_id,
        sentence_name=row.sentence_name,
        start_time=row.start_realigned,
        end_time=row.end_realigned,
        text=row.sentence,
    )


def _output_refs(layout: DebugLayout, split, sample_id: str) -> GateSourceOutputRefs:
    passed_ref = sample_manifest_relative_path(SampleStatus.PASSED, split, sample_id).as_posix()
    dropped_ref = sample_manifest_relative_path(SampleStatus.DROPPED, split, sample_id).as_posix()
    return GateSourceOutputRefs(
        passed_payload_path=runtime_sample_payload_path(layout, passed_ref),
        passed_payload_ref=passed_ref,
        dropped_payload_path=runtime_sample_payload_path(layout, dropped_ref),
        dropped_payload_ref=dropped_ref,
    )


def _compare_gate(
    dossier: DebugSampleDossier,
    computed_status: SampleStatus,
    evaluation,
) -> DebugGateComparison:
    existing = dossier.existing_gate
    if existing.status in {
        ExistingGateStatus.MISSING,
        ExistingGateStatus.AMBIGUOUS,
        ExistingGateStatus.CORRUPT,
    }:
        return DebugGateComparison(False, None, (), existing.warnings)
    differences: list[str] = []
    if existing.status.value != computed_status.value:
        differences.append(
            f"status: computed={computed_status.value}, existing={existing.status.value}"
        )
    if evaluation.passed_entry is not None and existing.passed_entry is not None:
        _compare_passed_fields(evaluation.passed_entry, existing.passed_entry, differences)
    if evaluation.dropped_entry is not None and existing.dropped_entry is not None:
        _compare_dropped_fields(evaluation.dropped_entry, existing.dropped_entry, differences)
    return DebugGateComparison(
        compared=True,
        status_matches=not differences,
        differences=tuple(differences),
        warnings=existing.warnings,
    )


def _compare_passed_fields(computed, existing, differences: list[str]) -> None:
    for field_name in (
        "sample_id",
        "split",
        "payload_ref",
        "text",
        "fps",
        "frame_count",
        "source_video_id",
        "source_sentence_id",
        "source_sentence_name",
        "valid_frame_count",
        "body_nonzero_frame_count",
        "face_nonzero_frame_count",
        "left_hand_nonzero_frame_count",
        "right_hand_nonzero_frame_count",
    ):
        computed_value = getattr(computed, field_name)
        existing_value = getattr(existing, field_name)
        if computed_value != existing_value:
            differences.append(
                f"{field_name}: computed={computed_value}, existing={existing_value}"
            )


def _compare_dropped_fields(computed, existing, differences: list[str]) -> None:
    for field_name in (
        "sample_id",
        "split",
        "text",
        "source_video_id",
        "source_sentence_id",
        "source_sentence_name",
        "drop_stage",
        "dropped_sample_ref",
    ):
        computed_value = getattr(computed, field_name)
        existing_value = getattr(existing, field_name)
        if computed_value != existing_value:
            differences.append(
                f"{field_name}: computed={_value(computed_value)}, existing={_value(existing_value)}"
            )
    computed_issue_codes = tuple(sorted(code.value for code in computed.issue_codes))
    existing_issue_codes = tuple(sorted(code.value for code in existing.issue_codes))
    if computed_issue_codes != existing_issue_codes:
        differences.append(
            f"issue_codes: computed={computed_issue_codes}, existing={existing_issue_codes}"
        )


def _frame_metrics(evaluation) -> DebugGateFrameMetrics:
    if evaluation.passed_entry is not None:
        entry = evaluation.passed_entry
        return DebugGateFrameMetrics(
            frame_count=entry.frame_count,
            valid_frame_count=entry.valid_frame_count,
            body_nonzero_frame_count=entry.body_nonzero_frame_count,
            face_nonzero_frame_count=entry.face_nonzero_frame_count,
            left_hand_nonzero_frame_count=entry.left_hand_nonzero_frame_count,
            right_hand_nonzero_frame_count=entry.right_hand_nonzero_frame_count,
        )
    if evaluation.prepared_sample is not None:
        sample = evaluation.prepared_sample
        return DebugGateFrameMetrics(
            frame_count=sample.pose.frame_count,
            valid_frame_count=int(sample.pose.valid_frame_mask.sum()),
            body_nonzero_frame_count=sample.pose.body_nonzero_frame_count,
            face_nonzero_frame_count=sample.pose.face_nonzero_frame_count,
            left_hand_nonzero_frame_count=sample.pose.left_hand_nonzero_frame_count,
            right_hand_nonzero_frame_count=sample.pose.right_hand_nonzero_frame_count,
        )
    return DebugGateFrameMetrics(None, None, None, None, None, None)


def _value(value):
    return value.value if hasattr(value, "value") else value


def _gate_verdict(
    existing_status: ExistingGateStatus,
    comparison: DebugGateComparison,
) -> DebugGateVerdict:
    if existing_status is ExistingGateStatus.MISSING:
        return DebugGateVerdict.EXISTING_MISSING
    if existing_status in {ExistingGateStatus.AMBIGUOUS, ExistingGateStatus.CORRUPT}:
        return DebugGateVerdict.EXISTING_AMBIGUOUS
    if comparison.differences:
        return DebugGateVerdict.MISMATCH
    return DebugGateVerdict.MATCH


def _gate_recompute_progress_spec() -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=DEBUG_STAGE_GATE_RECOMPUTE,
        label="gate recompute",
        unit="sample",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="gate_recompute",
        total_semantics="target sample gate recomputation",
    )


__all__ = ["debug_gate"]
