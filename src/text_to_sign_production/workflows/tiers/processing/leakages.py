from __future__ import annotations

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    ProgressTaskHandle,
)
from text_to_sign_production.data.leakages import (
    LeakageBundle,
    LeakageProgressEvent,
    LeakageProgressSink,
    build_leakage_bundle,
    build_leakage_input,
    validate_leakage_bundle,
)
from text_to_sign_production.data.metrics import MetricBundle
from text_to_sign_production.data.samples import PassedManifestEntry
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_LEAKAGE_DUPLICATES,
    TIERS_STAGE_LEAKAGE_FACTS,
    TIERS_STAGE_LEAKAGE_INDEX_RELATIONS,
    TIERS_STAGE_LEAKAGE_RELATIONS,
    TIERS_STAGE_LEAKAGE_SUMMARIES,
    TIERS_STAGE_LEAKAGE_SUMMARY_PAIRS,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import TiersWorkflowInvariantError


def build_tiers_leakage_bundle(
    manifests: tuple[PassedManifestEntry, ...],
    metric_bundles: tuple[MetricBundle, ...],
    *,
    progress_session: ProgressSession | None = None,
) -> LeakageBundle:
    _validate_manifest_metric_alignment(manifests, metric_bundles)
    inputs = tuple(
        build_leakage_input(manifest, metric_bundle)
        for manifest, metric_bundle in zip(manifests, metric_bundles, strict=True)
    )
    progress_adapter = (
        _LeakageProgressAdapter(progress_session) if progress_session is not None else None
    )
    try:
        leakage_bundle = build_leakage_bundle(inputs, progress_sink=progress_adapter)
    finally:
        if progress_adapter is not None:
            progress_adapter.close_all()
    issues = validate_leakage_bundle(leakage_bundle)
    if issues:
        raise TiersWorkflowInvariantError("Leakage bundle validation failed")
    return leakage_bundle


def _validate_manifest_metric_alignment(
    manifests: tuple[PassedManifestEntry, ...],
    metric_bundles: tuple[MetricBundle, ...],
) -> None:
    manifest_keys = tuple((manifest.split, manifest.sample_id) for manifest in manifests)
    metric_keys = tuple((bundle.split, bundle.sample_id) for bundle in metric_bundles)
    if manifest_keys != metric_keys:
        raise TiersWorkflowInvariantError("manifest and metric bundle identity order is misaligned")


_LEAKAGE_PHASE_PROGRESS: dict[str, tuple[str, str]] = {
    "duplicates": (TIERS_STAGE_LEAKAGE_DUPLICATES, "sample"),
    "index_relations": (TIERS_STAGE_LEAKAGE_INDEX_RELATIONS, "relation_input"),
    "relations": (TIERS_STAGE_LEAKAGE_RELATIONS, "pair"),
    "facts": (TIERS_STAGE_LEAKAGE_FACTS, "pair"),
    "summary_pairs": (TIERS_STAGE_LEAKAGE_SUMMARY_PAIRS, "pair"),
    "summaries": (TIERS_STAGE_LEAKAGE_SUMMARIES, "sample"),
}


class _LeakageProgressAdapter(LeakageProgressSink):
    def __init__(self, progress_session: ProgressSession) -> None:
        self._progress_session = progress_session
        self._tasks: dict[str, ProgressTaskHandle] = {}

    def update_leakage_progress(self, event: LeakageProgressEvent) -> None:
        phase = str(event.phase)
        kind = str(event.kind)
        if phase not in _LEAKAGE_PHASE_PROGRESS:
            raise TiersWorkflowInvariantError(f"Unknown leakage progress phase: {phase}")
        if kind == "start":
            self._start_phase(phase, event.total)
        elif kind == "advance":
            task = self._tasks.get(phase)
            if task is not None:
                task.advance()
        elif kind == "finish":
            self._finish_phase(phase)

    def close_all(self) -> None:
        for phase in tuple(self._tasks):
            self._finish_phase(phase)

    def _start_phase(self, phase: str, total: int | None) -> None:
        if phase in self._tasks:
            self._finish_phase(phase)
        if total is None or total <= 0:
            return
        stage_id, unit = _LEAKAGE_PHASE_PROGRESS[phase]
        task = self._progress_session.task(
            _leakage_progress_spec(
                phase=phase,
                stage_id=stage_id,
                unit=unit,
            ),
            total=total,
        )
        self._tasks[phase] = task.__enter__()

    def _finish_phase(self, phase: str) -> None:
        task = self._tasks.pop(phase, None)
        if task is not None:
            task.close()


def _leakage_progress_spec(
    *,
    phase: str,
    stage_id: str,
    unit: str,
) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIERS_WORKFLOW_NAME,
        stage_id=stage_id,
        label=f"leakage {phase}",
        unit=unit,
        owner_module=__name__,
        split_behavior="global",
        operation_kind=f"leakage_{phase}",
        total_semantics=f"data-layer leakage {phase} progress events",
        bar_eligible=True,
    )
