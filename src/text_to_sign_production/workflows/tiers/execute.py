"""Main orchestration entry point for the tiers workflow."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from text_to_sign_production.artifacts.catalog import (
    SampleHandle,
    iter_samples,
    load_passed_samples_catalog,
)
from text_to_sign_production.artifacts.store import (
    TierMembership,
    build_artifact_stores,
)
from text_to_sign_production.data.leakages import build_leakage_bundle, build_leakage_input
from text_to_sign_production.data.metrics import build_metric_bundle
from text_to_sign_production.data.samples import (
    PassedManifestEntry,
    load_processed_sample_payload,
    write_manifest_jsonl,
)
from text_to_sign_production.data.tiers import (
    build_excluded_surface_entries,
    build_included_surface_entries,
    build_tier_bundle,
    load_filter_config,
    load_tier_policies,
    validate_tier_bundle,
)
from text_to_sign_production.data.tiers.types import TierName
from text_to_sign_production.foundation.progress import (
    ItemProgress,
    ProgressReporter,
    TqdmProgressReporter,
)
from text_to_sign_production.foundation.roots import build_repo_roots
from text_to_sign_production.workflows.tiers.reports import write_tiers_reports
from text_to_sign_production.workflows.tiers.types import (
    TiersWorkflowConfig,
    TiersWorkflowError,
    TiersWorkflowOutputSummary,
    TiersWorkflowResult,
)


def run_tiers_workflow(
    config: TiersWorkflowConfig,
    *,
    reporter: ProgressReporter | None = None,
) -> TiersWorkflowResult:
    progress_reporter = reporter or TqdmProgressReporter(prefix="[tiers]")
    stores = build_artifact_stores(
        build_repo_roots(config.project_root),
        build_repo_roots(config.drive_project_root),
    )
    filter_config = load_filter_config(config.project_root / config.filters_config_relative_path)
    tier_policies = load_tier_policies(config.project_root / config.tiers_config_relative_path)

    try:
        with ItemProgress(
            label="load passed manifests",
            total=len(config.splits),
        unit="manifest",
        reporter=progress_reporter,
    ) as catalog_progress:
            catalog_progress.note("split: -")
            catalog = load_passed_samples_catalog(
                stores,
                splits=config.splits,
                progress_callback=lambda **fields: _advance_catalog_progress(
                    catalog_progress,
                    fields,
                ),
            )
    except Exception as exc:
        raise TiersWorkflowError(f"Failed to load passed samples catalog: {exc}") from exc

    handles = tuple(iter_samples(catalog))
    manifests: list[PassedManifestEntry] = []
    metric_bundles = []
    leakage_inputs = []
    with ItemProgress(
        label="prepare tier inputs",
        total=len(handles),
        unit="sample",
        reporter=progress_reporter,
    ) as sample_progress:
        sample_progress.note("split: -")
        for handle in handles:
            manifest = cast(PassedManifestEntry, handle.manifest_entry)
            sample_progress.note(_tier_input_progress_note(manifest))
            payload = _load_payload_for_handle(handle)

            sample_progress.note(_tier_input_progress_note(manifest))
            try:
                metric_bundle = build_metric_bundle(payload, manifest)
            except Exception as exc:
                raise TiersWorkflowError(
                    "Failed to build metric bundle for "
                    f"{manifest.split.value}/{manifest.sample_id}: {exc}"
                ) from exc

            sample_progress.note(_tier_input_progress_note(manifest))
            leakage_input = build_leakage_input(manifest, metric_bundle)

            manifests.append(manifest)
            metric_bundles.append(metric_bundle)
            leakage_inputs.append(leakage_input)
            sample_progress.advance()

    leakage_progress = _LeakageProgressRenderer(progress_reporter)
    try:
        leakage_bundle = build_leakage_bundle(
            leakage_inputs,
            progress_callback=leakage_progress,
        )
    finally:
        leakage_progress.close()

    total_decisions = len(manifests) * len(tuple(TierName))
    with ItemProgress(
        label="apply tier policies",
        total=total_decisions,
        unit="decision",
        reporter=progress_reporter,
    ) as decision_progress:
        decision_progress.note(
            "split: - | include: 0.0% | exclude: 0.0%"
        )
        tier_bundle = build_tier_bundle(
            manifests,
            metric_bundles,
            leakage_bundle.sample_summaries,
            filter_config,
            tier_policies,
            progress_callback=lambda **fields: _advance_tier_decision_progress(
                decision_progress,
                total_decisions,
                fields,
            ),
        )
    tier_issues = validate_tier_bundle(tier_bundle)
    if tier_issues:
        codes = ", ".join(issue.code for issue in tier_issues)
        raise TiersWorkflowError(f"Invalid tier bundle: {codes}")

    manifest_by_key = {(entry.split, entry.sample_id): entry for entry in manifests}
    tiered_manifest_paths = _write_tiered_manifests(
        stores.runtime,
        tier_bundle,
        manifest_by_key,
        config.splits,
        reporter=progress_reporter,
    )

    report_paths = write_tiers_reports(
        stores.runtime.reports.tiers_root,
        metric_bundles=metric_bundles,
        leakage_bundle=leakage_bundle,
        tier_bundle=tier_bundle,
        filter_config=filter_config,
        reporter=progress_reporter,
        progress_label="write tiers report",
    )

    included_counts, excluded_counts = _count_decisions(tier_bundle, config.splits)

    return TiersWorkflowResult(
        output_summary=TiersWorkflowOutputSummary(
            included_counts_by_tier_by_split=included_counts,
            excluded_counts_by_tier_by_split=excluded_counts,
            report_paths=report_paths,
            tiered_manifest_paths=tiered_manifest_paths,
        )
    )


def _load_payload_for_handle(handle: SampleHandle) -> Any:
    if handle.runtime_sample is None:
        raise TiersWorkflowError(
            "Passed sample has no runtime sample path: "
            f"{handle.ref.split.value}/{handle.ref.sample_id}"
        )
    sample_path = handle.runtime_sample.path
    if not sample_path.is_file():
        raise TiersWorkflowError(
            f"Passed sample payload is missing: {handle.ref.split.value}/"
            f"{handle.ref.sample_id}: {sample_path}"
        )
    try:
        return load_processed_sample_payload(sample_path)
    except Exception as exc:
        raise TiersWorkflowError(
            f"Failed to load passed sample payload {sample_path}: {exc}"
        ) from exc


def _advance_catalog_progress(progress: ItemProgress, fields: dict[str, object]) -> None:
    split = str(fields.get("manifest", "-"))
    progress.advance()
    progress.note(f"split: {split}")


def _tier_input_progress_note(manifest: PassedManifestEntry) -> str:
    return f"split: {manifest.split.value}"


def _write_tiered_manifests(
    topology: Any,
    tier_bundle: Any,
    manifest_by_key: Mapping[tuple[Any, str], PassedManifestEntry],
    splits: tuple[Any, ...],
    *,
    reporter: ProgressReporter | None = None,
) -> tuple[Path, ...]:
    """Write tiered selection manifests as passed-manifest entries."""
    progress_reporter = reporter or TqdmProgressReporter(prefix="[tiers]")
    paths: list[Path] = []
    paths.extend(
        _write_tiered_manifest_family(
            topology,
            tier_bundle,
            manifest_by_key,
            splits,
            membership=TierMembership.INCLUDED,
            label="write included manifests",
            reporter=progress_reporter,
        )
    )
    paths.extend(
        _write_tiered_manifest_family(
            topology,
            tier_bundle,
            manifest_by_key,
            splits,
            membership=TierMembership.EXCLUDED,
            label="write excluded manifests",
            reporter=progress_reporter,
        )
    )
    return tuple(paths)


def _write_tiered_manifest_family(
    topology: Any,
    tier_bundle: Any,
    manifest_by_key: Mapping[tuple[Any, str], PassedManifestEntry],
    splits: tuple[Any, ...],
    *,
    membership: TierMembership,
    label: str,
    reporter: ProgressReporter,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    sorted_splits = tuple(sorted(splits, key=lambda item: item.value))
    entries_by_tier_and_split: dict[
        tuple[TierName, Any],
        list[PassedManifestEntry],
    ] = defaultdict(list)

    for tier_name in TierName:
        surface_entries = (
            build_included_surface_entries(tier_bundle, tier_name=tier_name)
            if membership is TierMembership.INCLUDED
            else build_excluded_surface_entries(tier_bundle, tier_name=tier_name)
        )
        for surface_entry in surface_entries:
            key = (surface_entry.split, surface_entry.sample_id)
            try:
                entries_by_tier_and_split[(tier_name, surface_entry.split)].append(
                    manifest_by_key[key]
                )
            except KeyError as exc:
                raise TiersWorkflowError(
                    f"Tier surface entry has no source manifest: "
                    f"{surface_entry.split.value}/{surface_entry.sample_id}"
                ) from exc

    total_records = sum(len(entries) for entries in entries_by_tier_and_split.values())
    with ItemProgress(
        label=label,
        total=total_records,
        unit="record",
        reporter=reporter,
    ) as progress:
        progress.note("tier: - | split: -")
        for tier_name in TierName:
            for split in sorted_splits:
                path = topology.manifests.tiered_manifest(
                    tier_name.value,
                    membership,
                    split,
                ).path
                entries = entries_by_tier_and_split.get((tier_name, split), ())

                def progress_callback(
                    *,
                    _tier_name: TierName = tier_name,
                    _split: Any = split,
                    **_: object,
                ) -> None:
                    _advance_tiered_manifest_progress(
                        progress,
                        tier_name=_tier_name,
                        split=_split,
                    )

                write_manifest_jsonl(
                    path,
                    entries,
                    progress_callback=progress_callback,
                )
                progress.note(f"tier: {tier_name.value} | split: {split.value}")
                paths.append(path)
    return tuple(paths)


def _advance_tiered_manifest_progress(
    progress: ItemProgress,
    *,
    tier_name: TierName,
    split: Any,
) -> None:
    progress.advance()
    progress.note(f"tier: {tier_name.value} | split: {split.value}")


class _LeakageProgressRenderer:
    """Translate leakage phase events into finite workflow progress bars."""

    def __init__(self, reporter: ProgressReporter) -> None:
        self._reporter = reporter
        self._progress: ItemProgress | None = None

    def __call__(self, **fields: object) -> None:
        event = str(fields.pop("event", "advance"))
        if event == "start":
            self.close()
            self._progress = ItemProgress(
                label=str(fields["label"]),
                total=_progress_int(fields["total"]),
                unit=str(fields["unit"]),
                reporter=self._reporter,
            )
            return
        if event == "finish":
            self.close()
            return
        if event != "advance":
            raise ValueError(f"Unknown leakage progress event: {event!r}")
        if self._progress is None:
            return
        self._progress.advance()
        note = _leakage_progress_note(fields)
        if note:
            self._progress.note(note)

    def close(self) -> None:
        if self._progress is None:
            return
        self._progress.close()
        self._progress = None


def _leakage_progress_note(fields: Mapping[str, object]) -> str:
    phase = fields.get("phase")
    if phase == "duplicates":
        return _progress_note(fields, ("split",))
    if phase == "index_relations":
        return _progress_note(fields, ("relation", "split"))
    if phase == "relations":
        return _progress_note(fields, ("relation", "match_rate"))
    if phase == "summaries":
        return _progress_note(fields, ("split", "leakage"))
    return ""


def _progress_note(fields: Mapping[str, object], keys: tuple[str, ...]) -> str:
    parts = []
    for key in keys:
        value = fields.get(key)
        if value is None:
            continue
        label = "match" if key == "match_rate" else key
        parts.append(f"{label}: {value}")
    return " | ".join(parts)


def _advance_tier_decision_progress(
    progress: ItemProgress,
    total_decisions: int,
    fields: dict[str, object],
) -> None:
    included = _progress_int(fields.get("included", 0))
    excluded = _progress_int(fields.get("excluded", 0))
    denominator = total_decisions or 1
    include_rate = included / denominator * 100.0
    exclude_rate = excluded / denominator * 100.0
    progress.advance()
    progress.note(
        f"split: {fields.get('split', '-')} | include: {include_rate:.1f}% | "
        f"exclude: {exclude_rate:.1f}%"
    )


def _progress_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Progress count must be int-like, got {type(value).__name__}.")


def _count_decisions(
    tier_bundle: Any,
    splits: tuple[Any, ...],
) -> tuple[dict[TierName, dict[Any, int]], dict[TierName, dict[Any, int]]]:
    included = {tier: {split: 0 for split in splits} for tier in TierName}
    excluded = {tier: {split: 0 for split in splits} for tier in TierName}
    for decision in tier_bundle.decisions:
        if decision.included:
            included[decision.tier_name][decision.split] += 1
        else:
            excluded[decision.tier_name][decision.split] += 1
    return included, excluded


__all__ = ["run_tiers_workflow"]
