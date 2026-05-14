from __future__ import annotations

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.dataset.manifests import read_passed_manifest_json
from text_to_sign_production.data.tier.leakages import (
    LeakageBundle,
    LeakageProgressSpecs,
    build_leakage_bundle_from_inputs,
    build_leakage_input_from_manifest,
    sample_leakage_summary,
    validate_leakage_bundle,
)
from text_to_sign_production.workflows.debug.contracts import (
    DebugLeakageContext,
)
from text_to_sign_production.workflows.debug.constants import (
    DEBUG_STAGE_LEAKAGE_INPUTS,
    DEBUG_STAGE_LEAKAGE_RELATIONS,
    DEBUG_STAGE_LEAKAGE_SUMMARIES,
)
from text_to_sign_production.workflows.debug.contracts.verdicts import DebugLeakageContextMode
from text_to_sign_production.workflows.debug.layout import DebugLayout, runtime_passed_manifest_path
from text_to_sign_production.workflows.debug.progress import debug_progress_stage


def build_leakage_context(debug_splits: tuple[SampleSplit, ...]) -> DebugLeakageContext:
    split_set = set(debug_splits)
    all_splits = {SampleSplit.TRAIN, SampleSplit.VAL, SampleSplit.TEST}
    if len(debug_splits) == 1:
        return DebugLeakageContext(
            debug_splits=debug_splits,
            mode=DebugLeakageContextMode.SINGLE_SPLIT_CONTEXT,
            production_like=False,
            warning="cross-split leakage cannot be observed with one selected split",
        )
    if split_set == all_splits:
        return DebugLeakageContext(
            debug_splits=debug_splits,
            mode=DebugLeakageContextMode.GLOBAL_SPLIT_CONTEXT,
            production_like=True,
            warning=None,
        )
    return DebugLeakageContext(
        debug_splits=debug_splits,
        mode=DebugLeakageContextMode.MULTI_SPLIT_CONTEXT,
        production_like=False,
        warning="selected splits are a subset of train/val/test; leakage context is partial",
    )


def build_manifest_only_leakage_bundle(
    layout: DebugLayout,
    debug_splits: tuple[SampleSplit, ...],
    *,
    progress_session: ProgressSession | None = None,
) -> LeakageBundle:
    inputs = []
    if progress_session is not None:
        with progress_session.task(_leakage_inputs_progress_spec(), total=len(debug_splits)) as task:
            for split in debug_splits:
                path = runtime_passed_manifest_path(layout, split)
                _require_passed_manifest(path, split)
                entries = read_passed_manifest_json(path)
                inputs.extend(build_leakage_input_from_manifest(entry) for entry in entries)
                task.advance()
    else:
        for split in debug_splits:
            path = runtime_passed_manifest_path(layout, split)
            _require_passed_manifest(path, split)
            entries = read_passed_manifest_json(path)
            inputs.extend(build_leakage_input_from_manifest(entry) for entry in entries)
    bundle = build_leakage_bundle_from_inputs(
        inputs,
        progress_session=progress_session,
        progress_specs=LeakageProgressSpecs(
            relation_scan=leakage_relation_progress_spec(),
            summary_build=leakage_summary_progress_spec(),
        ),
    )
    issues = validate_leakage_bundle(bundle)
    if issues:
        raise ValueError(f"Leakage bundle validation failed: {issues}")
    return bundle


def manifest_only_sample_leakage_summary(
    layout: DebugLayout,
    *,
    debug_splits: tuple[SampleSplit, ...],
    split: SampleSplit,
    sample_id: str,
    progress_session: ProgressSession | None = None,
):
    bundle = build_manifest_only_leakage_bundle(
        layout,
        debug_splits,
        progress_session=progress_session,
    )
    return sample_leakage_summary(bundle, split=split, sample_id=sample_id)


def leakage_relation_progress_spec() -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=DEBUG_STAGE_LEAKAGE_RELATIONS,
        label="manifest leakage relations",
        unit="pair",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="leakage_relation_scan",
        total_semantics="manifest-only leakage input pairs scanned",
    )


def leakage_summary_progress_spec() -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=DEBUG_STAGE_LEAKAGE_SUMMARIES,
        label="manifest leakage summaries",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="leakage_summary_build",
        total_semantics="manifest-only leakage sample summaries built",
    )


def _leakage_inputs_progress_spec() -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=DEBUG_STAGE_LEAKAGE_INPUTS,
        label="manifest leakage inputs",
        unit="file",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="leakage_input_load",
        total_semantics="selected passed manifests read for manifest-only leakage",
    )


def _require_passed_manifest(path, split: SampleSplit) -> None:
    if not path.is_file():
        raise ValueError(
            "Cannot build manifest-only leakage context because passed manifest is missing "
            f"for split={split.value}: {path}"
        )


__all__ = [
    "build_leakage_context",
    "build_manifest_only_leakage_bundle",
    "leakage_relation_progress_spec",
    "leakage_summary_progress_spec",
    "manifest_only_sample_leakage_summary",
]
