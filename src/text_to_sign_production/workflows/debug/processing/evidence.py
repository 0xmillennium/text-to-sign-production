from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.ids import TierMembership, TierName
from text_to_sign_production.core.models import DroppedManifestEntry, PassedManifestEntry
from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    ProgressTaskHandle,
)
from text_to_sign_production.data.dataset.manifests import (
    read_dropped_manifest_json,
    read_passed_manifest_json,
    read_tier_manifest_json,
)
from text_to_sign_production.workflows.debug.constants import DEBUG_STAGE_EVIDENCE_COLLECTION
from text_to_sign_production.workflows.debug.contracts import (
    DebugExistingGateEvidence,
    DebugExistingTierEvidence,
    DebugSampleDossier,
    DebugSourceEvidence,
    DebugTargetResolution,
    ExistingGateStatus,
    ExistingTierMembershipState,
    TargetResolutionStatus,
)
from text_to_sign_production.workflows.debug.layout import (
    DebugLayout,
    runtime_dropped_manifest_path,
    runtime_keypoint_json_dir,
    runtime_keypoint_video_path,
    runtime_passed_manifest_path,
    runtime_sample_payload_path,
    runtime_tier_manifest_path,
)
from text_to_sign_production.workflows.debug.progress import debug_progress_stage


def collect_sample_evidence(
    layout: DebugLayout,
    resolution: DebugTargetResolution,
    *,
    progress_session: ProgressSession | None = None,
) -> DebugSampleDossier:
    if resolution.status is not TargetResolutionStatus.FOUND or resolution.match is None:
        raise ValueError("Target resolution must be FOUND before collecting a dossier.")
    translation = resolution.match
    if progress_session is not None:
        with progress_session.task(_evidence_progress_spec(), total=9) as progress_task:
            source = _source_evidence(
                layout,
                translation.split,
                translation.sentence_name,
                progress_task=progress_task,
            )
            gate = _existing_gate_evidence(
                layout,
                translation.split,
                translation.sentence_name,
                progress_task=progress_task,
            )
            tiers = tuple(
                _existing_tier_evidence(
                    layout,
                    translation.split,
                    translation.sentence_name,
                    tier,
                    progress_task=progress_task,
                )
                for tier in TierName
            )
    else:
        source = _source_evidence(layout, translation.split, translation.sentence_name)
        gate = _existing_gate_evidence(layout, translation.split, translation.sentence_name)
        tiers = tuple(
            _existing_tier_evidence(layout, translation.split, translation.sentence_name, tier)
            for tier in TierName
        )
    return DebugSampleDossier(
        resolution=resolution,
        translation=translation,
        source=source,
        existing_gate=gate,
        existing_tiers=tiers,
        consistency_notes=_consistency_notes(
            resolution=resolution,
            source=source,
            gate=gate,
            tiers=tiers,
        ),
    )


def _source_evidence(
    layout: DebugLayout,
    split,
    sentence_name: str,
    *,
    progress_task: ProgressTaskHandle | None = None,
) -> DebugSourceEvidence:
    keypoint_dir = runtime_keypoint_json_dir(layout, split, sentence_name)
    video_path = runtime_keypoint_video_path(layout, split, sentence_name)
    issues: list[str] = []
    frame_count = None
    if keypoint_dir.is_dir():
        frame_count = len(tuple(sorted(keypoint_dir.glob("*.json"))))
        if frame_count == 0:
            issues.append("keypoint JSON directory exists but contains no frame JSON files")
    else:
        issues.append("keypoint JSON directory is missing")
    if not video_path.is_file():
        issues.append("raw source video is missing")
    _advance(progress_task)
    return DebugSourceEvidence(
        keypoint_json_dir=keypoint_dir,
        keypoint_json_dir_exists=keypoint_dir.is_dir(),
        keypoint_frame_json_count=frame_count,
        raw_video_path=video_path,
        raw_video_exists=video_path.is_file(),
        source_issues=tuple(issues),
    )


def _existing_gate_evidence(
    layout: DebugLayout,
    split,
    sentence_name: str,
    *,
    progress_task: ProgressTaskHandle | None = None,
) -> DebugExistingGateEvidence:
    passed_path = runtime_passed_manifest_path(layout, split)
    dropped_path = runtime_dropped_manifest_path(layout, split)
    errors: list[str] = []
    warnings: list[str] = []
    passed_entries = _read_passed_matches(
        passed_path,
        sentence_name,
        errors,
        warnings,
        optional=True,
    )
    _advance(progress_task)
    dropped_entries = _read_dropped_matches(
        dropped_path,
        sentence_name,
        errors,
        warnings,
        optional=True,
    )
    _advance(progress_task)
    passed_entry = passed_entries[0] if len(passed_entries) == 1 else None
    dropped_entry = dropped_entries[0] if len(dropped_entries) == 1 else None
    if errors:
        status = ExistingGateStatus.CORRUPT
    elif (
        len(passed_entries) > 1 or len(dropped_entries) > 1 or (passed_entries and dropped_entries)
    ):
        status = ExistingGateStatus.AMBIGUOUS
    elif passed_entries:
        status = ExistingGateStatus.PASSED
    elif dropped_entries:
        status = ExistingGateStatus.DROPPED
    else:
        status = ExistingGateStatus.MISSING

    try:
        passed_payload_path = (
            runtime_sample_payload_path(layout, passed_entry.payload_ref) if passed_entry else None
        )
    except ValueError as exc:
        errors.append(str(exc))
        passed_payload_path = None
    try:
        dropped_record_path = (
            runtime_sample_payload_path(layout, dropped_entry.dropped_sample_ref)
            if dropped_entry
            else None
        )
    except ValueError as exc:
        errors.append(str(exc))
        dropped_record_path = None
    if errors and status is not ExistingGateStatus.CORRUPT:
        status = ExistingGateStatus.CORRUPT
    if status is ExistingGateStatus.PASSED and passed_payload_path is not None:
        if not passed_payload_path.is_file():
            warnings.append("existing gate passed manifest row has no restored payload file")
    return DebugExistingGateEvidence(
        status=status,
        passed_manifest_path=passed_path,
        dropped_manifest_path=dropped_path,
        passed_entry=passed_entry,
        dropped_entry=dropped_entry,
        passed_payload_path=passed_payload_path,
        passed_payload_exists=passed_payload_path.is_file() if passed_payload_path else False,
        dropped_record_path=dropped_record_path,
        dropped_record_exists=dropped_record_path.is_file() if dropped_record_path else False,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _existing_tier_evidence(
    layout: DebugLayout,
    split,
    sentence_name: str,
    tier: TierName,
    *,
    progress_task: ProgressTaskHandle | None = None,
) -> DebugExistingTierEvidence:
    included_path = runtime_tier_manifest_path(
        layout,
        tier=tier,
        membership=TierMembership.INCLUDED,
        split=split,
    )
    excluded_path = runtime_tier_manifest_path(
        layout,
        tier=tier,
        membership=TierMembership.EXCLUDED,
        split=split,
    )
    errors: list[str] = []
    warnings: list[str] = []
    included = _read_tier_matches(
        included_path,
        sentence_name,
        errors,
        warnings,
        optional=True,
    )
    _advance(progress_task)
    excluded = _read_tier_matches(
        excluded_path,
        sentence_name,
        errors,
        warnings,
        optional=True,
    )
    _advance(progress_task)
    included_entry = included[0] if len(included) == 1 else None
    excluded_entry = excluded[0] if len(excluded) == 1 else None
    if errors or len(included) > 1 or len(excluded) > 1 or (included and excluded):
        state = ExistingTierMembershipState.AMBIGUOUS
    elif included:
        state = ExistingTierMembershipState.INCLUDED
    elif excluded:
        state = ExistingTierMembershipState.EXCLUDED
    else:
        state = ExistingTierMembershipState.MISSING
    return DebugExistingTierEvidence(
        tier=tier,
        state=state,
        included_manifest_path=included_path,
        excluded_manifest_path=excluded_path,
        included_entry=included_entry,
        excluded_entry=excluded_entry,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _read_passed_matches(
    path: Path,
    sentence_name: str,
    errors: list[str],
    warnings: list[str],
    *,
    optional: bool,
) -> tuple[PassedManifestEntry, ...]:
    if optional and not path.exists():
        warnings.append(f"passed manifest is missing: {path}")
        return ()
    try:
        entries = read_passed_manifest_json(path)
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"could not read passed manifest {path}: {exc}")
        return ()
    return tuple(
        sorted(
            (entry for entry in entries if entry.source_sentence_name == sentence_name),
            key=lambda entry: (entry.split.value, entry.sample_id),
        )
    )


def _read_dropped_matches(
    path: Path,
    sentence_name: str,
    errors: list[str],
    warnings: list[str],
    *,
    optional: bool,
) -> tuple[DroppedManifestEntry, ...]:
    if optional and not path.exists():
        warnings.append(f"dropped manifest is missing: {path}")
        return ()
    try:
        entries = read_dropped_manifest_json(path)
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"could not read dropped manifest {path}: {exc}")
        return ()
    return tuple(
        sorted(
            (entry for entry in entries if entry.source_sentence_name == sentence_name),
            key=lambda entry: (entry.split.value, entry.sample_id),
        )
    )


def _read_tier_matches(
    path: Path,
    sentence_name: str,
    errors: list[str],
    warnings: list[str],
    *,
    optional: bool,
) -> tuple[PassedManifestEntry, ...]:
    if optional and not path.exists():
        warnings.append(f"tier manifest is missing: {path}")
        return ()
    try:
        entries = read_tier_manifest_json(path)
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"could not read tier manifest {path}: {exc}")
        return ()
    return tuple(
        sorted(
            (entry for entry in entries if entry.source_sentence_name == sentence_name),
            key=lambda entry: (entry.split.value, entry.sample_id),
        )
    )


def _consistency_notes(
    *,
    resolution: DebugTargetResolution,
    source: DebugSourceEvidence,
    gate: DebugExistingGateEvidence,
    tiers: tuple[DebugExistingTierEvidence, ...],
) -> tuple[str, ...]:
    notes = [
        "Target sample exists in restored translation evidence.",
        (
            "Raw keypoint directory is available."
            if source.keypoint_json_dir_exists
            else "Raw keypoint directory is missing."
        ),
        "Raw source video is available."
        if source.raw_video_exists
        else "Raw source video is missing.",
        f"Existing gate output status: {gate.status.value}.",
    ]
    if gate.status is ExistingGateStatus.PASSED:
        notes.append(
            "Existing passed payload is available."
            if gate.passed_payload_exists
            else "Existing passed payload is missing."
        )
    for tier in tiers:
        notes.append(f"Existing {tier.tier.value} tier output status: {tier.state.value}.")
    if len(resolution.request.debug_splits) == 1:
        notes.append("DEBUG_SPLITS is single-split; leakage comparison is context-limited.")
    return tuple(notes)


def _advance(progress_task: ProgressTaskHandle | None) -> None:
    if progress_task is not None:
        progress_task.advance()


def _evidence_progress_spec() -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=DEBUG_STAGE_EVIDENCE_COLLECTION,
        label="sample evidence collection",
        unit="check",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="evidence_collection",
        total_semantics="restored runtime evidence checks for one target sample",
    )


__all__ = ["collect_sample_evidence"]
