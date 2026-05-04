"""Main orchestration entry point for the samples workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from text_to_sign_production.artifacts.store import (
    SampleStatus,
    build_artifact_topology,
    sample_manifest_relative_path,
)
from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.gates import (
    evaluate_sample_processing,
    evaluate_unmatched_source,
    load_gates_config,
)
from text_to_sign_production.data.gates.types import (
    GateStage,
    ProcessingDecision,
    ProcessingStatus,
)
from text_to_sign_production.data.pose import (
    PoseBuildInput,
    build_pose_tensors,
    discover_frame_files,
)
from text_to_sign_production.data.pose.types import PoseBuildOutput
from text_to_sign_production.data.samples import (
    DroppedManifestEntry,
    ManifestEntry,
    PassedManifestEntry,
    ProcessedSamplePayload,
    build_dropped_entry,
    build_passed_entry,
    build_payload,
    validate_dropped_entry,
    validate_passed_entry,
    validate_payload,
    write_manifest_jsonl,
    write_processed_sample_payload,
)
from text_to_sign_production.data.sources import (
    SourceCandidate,
    SourceMatchResult,
    assemble_candidate,
    load_translation_rows,
    match_sources,
    read_video_metadata,
    resolve_keypoint_source,
    sample_id_from_translation,
)
from text_to_sign_production.data.sources.types import TranslationRow
from text_to_sign_production.foundation.progress import (
    ItemProgress,
    ProgressReporter,
    TqdmProgressReporter,
)
from text_to_sign_production.foundation.roots import build_repo_roots
from text_to_sign_production.workflows.samples.reports import write_samples_reports
from text_to_sign_production.workflows.samples.types import (
    SamplesWorkflowConfig,
    SamplesWorkflowError,
    SamplesWorkflowOutputSummary,
    SamplesWorkflowResult,
)


def run_samples_workflow(
    config: SamplesWorkflowConfig,
    *,
    reporter: ProgressReporter | None = None,
) -> SamplesWorkflowResult:
    progress_reporter = reporter or TqdmProgressReporter(prefix="[samples]")
    topology = build_artifact_topology(build_repo_roots(config.project_root))
    gates_config = load_gates_config(config.project_root / config.gates_config_relative_path)

    passed_entries_by_split: dict[SampleSplit, tuple[PassedManifestEntry, ...]] = {}
    dropped_entries_by_split: dict[SampleSplit, tuple[DroppedManifestEntry, ...]] = {}
    decisions_by_split: dict[SampleSplit, tuple[ProcessingDecision, ...]] = {}
    matches_by_split: dict[SampleSplit, tuple[SourceMatchResult, ...]] = {}
    manifest_paths: list[Path] = []
    total_passed = 0
    total_dropped = 0
    rows_by_split: dict[SampleSplit, list[TranslationRow]] = {
        split: load_translation_rows(topology.assets.translation_csv(split).path)
        for split in config.splits
    }
    total_samples = sum(len(rows) for rows in rows_by_split.values())

    with ItemProgress(
        label="process samples",
        total=total_samples,
        unit="sample",
        reporter=progress_reporter,
    ) as sample_progress:
        sample_progress.note(_sample_progress_note(None, None, total_samples, 0, 0))
        for split in config.splits:
            rows = rows_by_split[split]
            split_passed: list[PassedManifestEntry] = []
            split_dropped: list[DroppedManifestEntry] = []
            split_decisions: list[ProcessingDecision] = []
            split_matches: list[SourceMatchResult] = []

            topology.samples.passed_split_dir(split).path.mkdir(parents=True, exist_ok=True)
            topology.samples.dropped_split_dir(split).path.mkdir(parents=True, exist_ok=True)

            for row in rows:
                current_sample_id = sample_id_from_translation(row)
                match = _build_source_match(topology.assets, split, row)
                split_matches.append(match)

                if not match.matched:
                    decision = evaluate_unmatched_source(match)
                    split_decisions.append(decision)
                    split_dropped.append(_dropped_from_unmatched(match, decision))
                    total_dropped += 1
                    sample_progress.advance()
                    sample_progress.note(
                        _sample_progress_note(
                            split,
                            current_sample_id,
                            total_samples,
                            total_passed,
                            total_dropped,
                        )
                    )
                    continue

                candidate = assemble_candidate(match, _video_path(topology.assets, split, row))
                current_sample_id = candidate.sample_id
                frames = discover_frame_files(candidate.keypoints_dir)

                pose_output: PoseBuildOutput | None = None
                if (
                    candidate.structurally_viable
                    and not frames.missing
                    and frames.frame_count > 0
                ):
                    pose_output = build_pose_tensors(
                        PoseBuildInput(
                            candidate=candidate,
                            frames=frames,
                            person_selection_policy=config.person_selection_policy,
                        )
                    )

                decision = evaluate_sample_processing(
                    config=gates_config,
                    candidate=candidate,
                    frames_listing=frames,
                    pose_output=pose_output,
                )
                split_decisions.append(decision)

                if decision.status is ProcessingStatus.PROCESSED:
                    if pose_output is None:
                        raise SamplesWorkflowError(
                            f"Passed sample has no pose output: {candidate.sample_id}"
                        )
                    payload = _build_valid_payload(candidate, pose_output)
                    sample_path = topology.samples.sample_file(
                        SampleStatus.PASSED,
                        split,
                        candidate.sample_id,
                    ).path
                    write_processed_sample_payload(sample_path, payload)
                    passed_entry = build_passed_entry(
                        sample_id=candidate.sample_id,
                        text=candidate.text,
                        split=split,
                        num_frames=candidate.frame_count,
                        fps=candidate.video_metadata.fps,
                        sample_path=sample_manifest_relative_path(
                            SampleStatus.PASSED,
                            split,
                            candidate.sample_id,
                        ).as_posix(),
                        source_video_id=candidate.video_id,
                        source_sentence_id=candidate.sentence_id,
                        source_sentence_name=candidate.sentence_name,
                        selected_person=payload.selected_person,
                        frame_quality=payload.frame_quality,
                    )
                    _raise_if_issues(validate_passed_entry(passed_entry), candidate.sample_id)
                    split_passed.append(passed_entry)
                    total_passed += 1
                else:
                    sample_path_value = None
                    payload = None
                    debug_materialization_error = None
                    if decision.can_materialize_debug and pose_output is not None:
                        payload, debug_materialization_error = _try_build_valid_payload(
                            candidate,
                            pose_output,
                        )
                        if payload is not None:
                            sample_path = topology.samples.sample_file(
                                SampleStatus.DROPPED,
                                split,
                                candidate.sample_id,
                            ).path
                            write_processed_sample_payload(sample_path, payload)
                            sample_path_value = sample_manifest_relative_path(
                                SampleStatus.DROPPED,
                                split,
                                candidate.sample_id,
                            ).as_posix()
                    dropped_entry = _dropped_from_decision(
                        candidate=candidate,
                        decision=decision,
                        sample_path=sample_path_value,
                        payload=payload,
                        debug_materialization_error=debug_materialization_error,
                    )
                    _raise_if_issues(validate_dropped_entry(dropped_entry), candidate.sample_id)
                    split_dropped.append(dropped_entry)
                    total_dropped += 1

                sample_progress.advance()
                sample_progress.note(
                    _sample_progress_note(
                        split,
                        current_sample_id,
                        total_samples,
                        total_passed,
                        total_dropped,
                    )
                )

            passed_entries_by_split[split] = tuple(split_passed)
            dropped_entries_by_split[split] = tuple(split_dropped)
            decisions_by_split[split] = tuple(split_decisions)
            matches_by_split[split] = tuple(split_matches)

    manifest_paths.extend(
        _write_split_manifests_with_progress(
            topology,
            config.splits,
            passed_entries_by_split,
            dropped_entries_by_split,
            reporter=progress_reporter,
        )
    )

    report_paths = write_samples_reports(
        topology.reports.samples_root,
        processing_decisions=decisions_by_split,
        passed_entries=passed_entries_by_split,
        dropped_entries=dropped_entries_by_split,
        source_matches=matches_by_split,
        reporter=progress_reporter,
        progress_label="write samples report",
    )
    output_summary = SamplesWorkflowOutputSummary(
        passed_counts_by_split={
            split: len(passed_entries_by_split.get(split, ())) for split in config.splits
        },
        dropped_counts_by_split={
            split: len(dropped_entries_by_split.get(split, ())) for split in config.splits
        },
        report_paths=report_paths,
        untiered_manifest_paths=tuple(manifest_paths),
    )
    return SamplesWorkflowResult(output_summary=output_summary)


def _build_source_match(assets: Any, split: SampleSplit, row: TranslationRow) -> SourceMatchResult:
    keypoints_dir = assets.keypoint_json_dir(split).path / row.sentence_name
    video_path = _video_path(assets, split, row)
    return match_sources(
        translation=row,
        split=split,
        keypoints=resolve_keypoint_source(keypoints_dir),
        video_metadata=read_video_metadata(video_path),
    )


def _video_path(assets: Any, split: SampleSplit, row: TranslationRow) -> Path:
    return cast(Path, assets.keypoint_video_dir(split).path / f"{row.sentence_name}.mp4")


def _build_valid_payload(
    candidate: SourceCandidate,
    pose_output: PoseBuildOutput,
) -> ProcessedSamplePayload:
    payload = build_payload(
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
    _raise_if_issues(validate_payload(payload), candidate.sample_id)
    return payload


def _try_build_valid_payload(
    candidate: SourceCandidate,
    pose_output: PoseBuildOutput,
) -> tuple[ProcessedSamplePayload | None, str | None]:
    try:
        return _build_valid_payload(candidate, pose_output), None
    except SamplesWorkflowError as exc:
        return None, str(exc)


def _sample_progress_note(
    split: SampleSplit | None,
    sample_id: str | None,
    total_samples: int,
    passed: int,
    dropped: int,
) -> str:
    denominator = total_samples or 1
    pass_rate = passed / denominator * 100.0
    drop_rate = dropped / denominator * 100.0
    split_text = split.value if split is not None else "-"
    sample_text = sample_id or "-"
    return (
        f"split: {split_text} | sample: {sample_text} | "
        f"pass: {pass_rate:.1f}% | drop: {drop_rate:.1f}%"
    )


def _write_split_manifests_with_progress(
    topology: Any,
    splits: Sequence[SampleSplit],
    passed_entries_by_split: dict[SampleSplit, tuple[PassedManifestEntry, ...]],
    dropped_entries_by_split: dict[SampleSplit, tuple[DroppedManifestEntry, ...]],
    *,
    reporter: ProgressReporter,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    paths.extend(
        _write_manifest_family_with_progress(
            label="write passed manifests",
            splits=splits,
            entries_by_split=passed_entries_by_split,
            path_for_split=lambda split: topology.manifests.untiered_passed_manifest(split).path,
            reporter=reporter,
        )
    )
    paths.extend(
        _write_manifest_family_with_progress(
            label="write dropped manifests",
            splits=splits,
            entries_by_split=dropped_entries_by_split,
            path_for_split=lambda split: topology.manifests.untiered_dropped_manifest(split).path,
            reporter=reporter,
        )
    )
    return tuple(paths)


def _write_manifest_family_with_progress(
    *,
    label: str,
    splits: Sequence[SampleSplit],
    entries_by_split: Mapping[SampleSplit, Sequence[ManifestEntry]],
    path_for_split: Callable[[SampleSplit], Path],
    reporter: ProgressReporter,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    total_records = sum(len(entries_by_split.get(split, ())) for split in splits)
    with ItemProgress(
        label=label,
        total=total_records,
        unit="record",
        reporter=reporter,
    ) as progress:
        progress.note("split: -")
        for split in splits:
            path = path_for_split(split)
            entries = entries_by_split.get(split, ())

            def progress_callback(
                *,
                _split: SampleSplit = split,
                **_: object,
            ) -> None:
                _advance_manifest_family_progress(
                    progress,
                    split=_split,
                )

            write_manifest_jsonl(
                path,
                entries,
                progress_callback=progress_callback,
            )
            progress.note(f"split: {split.value}")
            paths.append(path)
    return tuple(paths)


def _advance_manifest_family_progress(
    progress: ItemProgress,
    *,
    split: SampleSplit,
) -> None:
    progress.advance()
    progress.note(f"split: {split.value}")


def _dropped_from_unmatched(
    match: SourceMatchResult,
    decision: ProcessingDecision,
) -> DroppedManifestEntry:
    translation = match.translation
    drop_stage = (
        decision.drop_stage.value
        if decision.drop_stage is not None
        else GateStage.SOURCE.value
    )
    return build_dropped_entry(
        sample_id=sample_id_from_translation(translation),
        split=match.split,
        drop_stage=drop_stage,
        drop_reasons=decision.drop_reasons,
        debug_only=False,
        drop_details={"unmatched_reason": match.unmatched_reason or "unmatched_source"},
        text=translation.text,
    )


def _dropped_from_decision(
    *,
    candidate: SourceCandidate,
    decision: ProcessingDecision,
    sample_path: str | None,
    payload: ProcessedSamplePayload | None,
    debug_materialization_error: str | None,
) -> DroppedManifestEntry:
    drop_stage = (
        decision.drop_stage.value
        if decision.drop_stage is not None
        else ProcessingStatus.DROPPED.value
    )
    return build_dropped_entry(
        sample_id=candidate.sample_id,
        split=candidate.split,
        drop_stage=drop_stage,
        drop_reasons=decision.drop_reasons or ("dropped",),
        debug_only=sample_path is not None,
        sample_path=sample_path,
        drop_details={
            "can_materialize_debug": decision.can_materialize_debug,
            "debug_materialized": sample_path is not None,
            **(
                {"debug_materialization_error": debug_materialization_error}
                if debug_materialization_error is not None
                else {}
            ),
            "gate_statuses": {
                stage.value: result.status.value
                for stage, result in decision.gate_results.items()
            },
        },
        text=candidate.text,
        num_frames=payload.num_frames if payload is not None else None,
        fps=payload.fps if payload is not None else None,
        selected_person=payload.selected_person if payload is not None else None,
        frame_quality=payload.frame_quality if payload is not None else None,
    )


def _raise_if_issues(issues: list[Any], sample_id: str) -> None:
    if issues:
        codes = ", ".join(str(issue.code) for issue in issues)
        raise SamplesWorkflowError(f"Invalid sample contract for {sample_id}: {codes}")


__all__ = ["run_samples_workflow"]
