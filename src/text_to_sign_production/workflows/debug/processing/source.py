from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.gate.sources.translations import load_translation_records
from text_to_sign_production.workflows.debug.constants import DEBUG_STAGE_TARGET_RESOLVE
from text_to_sign_production.workflows.debug.contracts import (
    DebugSampleRequest,
    DebugTargetResolution,
    DebugTranslationMatch,
    TargetResolutionStatus,
)
from text_to_sign_production.workflows.debug.layout import DebugLayout, runtime_translation_path
from text_to_sign_production.workflows.debug.progress import debug_progress_stage


def resolve_target_sample(
    layout: DebugLayout,
    request: DebugSampleRequest,
    *,
    progress_session: ProgressSession | None = None,
) -> DebugTargetResolution:
    matches: list[DebugTranslationMatch] = []
    errors: list[str] = []
    for split in request.debug_splits:
        path = runtime_translation_path(layout, split)
        try:
            records = load_translation_records(path)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{split.value}: could not load translation file {path}: {exc}")
            continue
        stage = _target_resolve_progress_spec(split.value)
        if progress_session is not None:
            with progress_session.task(stage, total=len(records)) as progress_task:
                for row_index, record in enumerate(records):
                    if record.sentence_name == request.target_sentence_name:
                        matches.append(_translation_match(split, path, row_index, record))
                    progress_task.advance()
        else:
            for row_index, record in enumerate(records):
                if record.sentence_name == request.target_sentence_name:
                    matches.append(_translation_match(split, path, row_index, record))
    sorted_matches = tuple(
        sorted(
            matches,
            key=lambda item: (item.split.value, item.translation_path.as_posix(), item.row_index),
        )
    )
    if errors:
        return DebugTargetResolution(
            request=request,
            status=TargetResolutionStatus.INVALID_REQUEST,
            match=None,
            matches=sorted_matches,
            errors=tuple(errors),
            warnings=(),
        )
    if not sorted_matches:
        return DebugTargetResolution(
            request=request,
            status=TargetResolutionStatus.NOT_FOUND,
            match=None,
            matches=(),
            errors=(f"SENTENCE_NAME not found: {request.target_sentence_name}",),
            warnings=(),
        )
    if len(sorted_matches) > 1:
        return DebugTargetResolution(
            request=request,
            status=TargetResolutionStatus.AMBIGUOUS,
            match=None,
            matches=sorted_matches,
            errors=(f"SENTENCE_NAME matched {len(sorted_matches)} rows.",),
            warnings=(),
        )
    return DebugTargetResolution(
        request=request,
        status=TargetResolutionStatus.FOUND,
        match=sorted_matches[0],
        matches=sorted_matches,
        errors=(),
        warnings=(),
    )


def _translation_match(split, path, row_index: int, record) -> DebugTranslationMatch:
    return DebugTranslationMatch(
        split=split,
        video_id=record.video_id,
        video_name=record.video_name,
        sentence_id=record.sentence_id,
        sentence_name=record.sentence_name,
        start_realigned=record.start_time,
        end_realigned=record.end_time,
        sentence=record.text,
        translation_path=path,
        row_index=row_index,
    )


def _target_resolve_progress_spec(split: str) -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=f"{DEBUG_STAGE_TARGET_RESOLVE}.{split}",
        label=f"target resolution [{split}]",
        unit="row",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="translation_scan",
        total_semantics="translation rows scanned for target SENTENCE_NAME",
    )


__all__ = ["resolve_target_sample"]
