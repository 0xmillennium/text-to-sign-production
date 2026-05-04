"""Internal report helpers for the tiers workflow."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.leakages.types import LeakageBundle, LeakageSeverity
from text_to_sign_production.data.metrics.types import MetricBundle, MetricFamily
from text_to_sign_production.data.tiers.calibration import build_tier_calibration_surfaces
from text_to_sign_production.data.tiers.types import (
    FilterConfig,
    TierBundle,
    TierCalibrationSurfaces,
    TierName,
)
from text_to_sign_production.foundation.progress import ItemProgress, ProgressReporter


def write_tiers_reports(
    reports_root: Path,
    *,
    metric_bundles: Sequence[MetricBundle],
    leakage_bundle: LeakageBundle,
    tier_bundle: TierBundle,
    filter_config: FilterConfig,
    reporter: ProgressReporter | None = None,
    progress_label: str | None = None,
) -> tuple[Path, ...]:
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / "tiers_report.md"
    calibration_surfaces = build_tier_calibration_surfaces(
        metric_bundles,
        tier_bundle,
        filter_config,
    )
    content = _render_tiers_report(
        metric_records=build_tiers_metric_summary_records(metric_bundles),
        leakage_records=build_tiers_leakage_summary_records(leakage_bundle),
        decision_summary_records=build_tiers_decision_summary_records(tier_bundle),
        decision_detail_records=build_tiers_decision_detail_records(tier_bundle),
        calibration_surfaces=calibration_surfaces,
    )
    if reporter is None:
        _write_markdown(report_path, content)
        return (report_path,)

    with ItemProgress(
        label=progress_label or "write tiers report",
        total=1,
        unit="report",
        reporter=reporter,
    ) as progress:
        _write_markdown(report_path, content)
        progress.advance()
    return (report_path,)


def build_tiers_metric_summary_records(
    metric_bundles: Sequence[MetricBundle],
) -> tuple[dict[str, object], ...]:
    counters: dict[tuple[SampleSplit, MetricFamily], int] = defaultdict(int)
    for bundle in metric_bundles:
        for family in MetricFamily:
            counters[(bundle.split, family)] += 1

    return tuple(
        {
            "split": split.value,
            "metric_family": family.value,
            "sample_count": counters[(split, family)],
        }
        for split, family in sorted(counters, key=lambda item: (item[0].value, item[1].value))
    )


def build_tiers_leakage_summary_records(
    leakage_bundle: LeakageBundle,
) -> tuple[dict[str, object], ...]:
    counters: dict[tuple[SampleSplit, LeakageSeverity], int] = defaultdict(int)
    for summary in leakage_bundle.sample_summaries:
        counters[(summary.split, summary.max_severity)] += 1

    records: list[dict[str, object]] = [
        {
            "scope": "split",
            "split": split.value,
            "max_leakage_severity": severity.value,
            "sample_count": counters[(split, severity)],
        }
        for split, severity in sorted(counters, key=lambda item: (item[0].value, item[1].value))
    ]
    records.append(
        {
            "scope": "aggregate",
            "split": None,
            "leakage_pair_count": len(leakage_bundle.pair_facts),
        }
    )
    return tuple(records)


def build_tiers_decision_summary_records(
    tier_bundle: TierBundle,
) -> tuple[dict[str, object], ...]:
    counters: dict[tuple[TierName, SampleSplit], dict[str, int]] = defaultdict(
        lambda: {"included_count": 0, "excluded_count": 0}
    )
    for decision in tier_bundle.decisions:
        counts = counters[(decision.tier_name, decision.split)]
        if decision.included:
            counts["included_count"] += 1
        else:
            counts["excluded_count"] += 1

    return tuple(
        {
            "tier": tier.value,
            "split": split.value,
            "included_count": counters[(tier, split)]["included_count"],
            "excluded_count": counters[(tier, split)]["excluded_count"],
        }
        for tier, split in sorted(counters, key=lambda item: (item[0].value, item[1].value))
    )


def build_tiers_decision_detail_records(
    tier_bundle: TierBundle,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "tier": decision.tier_name.value,
            "split": decision.split.value,
            "sample_id": decision.sample_id,
            "included": decision.included,
            "max_leakage_severity": decision.max_leakage_severity.value,
            "applied_family_levels": {
                family.value: level.value
                for family, level in decision.applied_family_levels.items()
            },
            "metric_failures": decision.metric_failures,
            "leakage_failure": decision.leakage_failure,
        }
        for decision in tier_bundle.decisions
    )


def _render_tiers_report(
    *,
    metric_records: Sequence[Mapping[str, object]],
    leakage_records: Sequence[Mapping[str, object]],
    decision_summary_records: Sequence[Mapping[str, object]],
    decision_detail_records: Sequence[Mapping[str, object]],
    calibration_surfaces: TierCalibrationSurfaces,
) -> str:
    sections = [
        "# Tiers Report",
        "",
        "## Metric Summary",
        _markdown_table(
            ("Split", "Metric Family", "Sample Count"),
            (
                (
                    record["split"],
                    record["metric_family"],
                    record["sample_count"],
                )
                for record in metric_records
            ),
        ),
        "",
        "## Leakage Summary",
        _markdown_table(
            ("Scope", "Split", "Max Severity", "Sample Count", "Pair Count"),
            (
                (
                    record["scope"],
                    record.get("split"),
                    record.get("max_leakage_severity"),
                    record.get("sample_count"),
                    record.get("leakage_pair_count"),
                )
                for record in leakage_records
            ),
        ),
        "",
        "## Tier Decision Summary",
        _markdown_table(
            ("Tier", "Split", "Included", "Excluded"),
            (
                (
                    record["tier"],
                    record["split"],
                    record["included_count"],
                    record["excluded_count"],
                )
                for record in decision_summary_records
            ),
        ),
        "",
        "## Tier Decision Detail",
        _markdown_table(
            (
                "Tier",
                "Split",
                "Sample ID",
                "Included",
                "Max Leakage",
                "Metric Failures",
                "Leakage Failure",
            ),
            (
                (
                    record["tier"],
                    record["split"],
                    record["sample_id"],
                    record["included"],
                    record["max_leakage_severity"],
                    record["metric_failures"],
                    record["leakage_failure"],
                )
                for record in decision_detail_records
            ),
        ),
        "",
        "## Calibration Metric Distributions",
        _markdown_table(
            (
                "Split",
                "Role",
                "Family",
                "Metric",
                "Count",
                "Missing",
                "Unique",
                "Min",
                "P5",
                "P25",
                "P50",
                "P75",
                "P95",
                "P99",
                "Max",
            ),
            (
                (
                    _split_value(record.split),
                    record.role.value,
                    record.family,
                    record.metric_key,
                    record.sample_count,
                    record.missing_count,
                    record.unique_value_count,
                    record.minimum,
                    record.p5,
                    record.p25,
                    record.p50,
                    record.p75,
                    record.p95,
                    record.p99,
                    record.maximum,
                )
                for record in calibration_surfaces.metric_distributions
            ),
        ),
        "",
        "## Calibration Pass/Fail Counts",
        _markdown_table(
            ("Tier", "Split", "Role", "Family", "Metric", "Pass", "Fail"),
            (
                (
                    record.tier_name.value,
                    record.split.value,
                    record.role.value,
                    record.family.value,
                    record.metric_key,
                    record.pass_count,
                    record.fail_count,
                )
                for record in calibration_surfaces.pass_fail_counts
            ),
        ),
        "",
        "## Calibration Primary Blockers",
        _markdown_table(
            ("Tier", "Split", "Role", "Family", "Metric", "Count"),
            (
                (
                    record.tier_name.value,
                    record.split.value,
                    record.role.value,
                    record.family.value,
                    record.metric_key,
                    record.blocker_count,
                )
                for record in calibration_surfaces.primary_blockers
            ),
        ),
        "",
        "## Calibration All Blockers",
        _markdown_table(
            ("Tier", "Split", "Role", "Family", "Metric", "Count"),
            (
                (
                    record.tier_name.value,
                    record.split.value,
                    record.role.value,
                    record.family.value,
                    record.metric_key,
                    record.blocker_count,
                )
                for record in calibration_surfaces.all_blockers
            ),
        ),
        "",
        "## Calibration Family Co-Failures",
        _markdown_table(
            ("Tier", "Split", "Left Family", "Right Family", "Count"),
            (
                (
                    record.tier_name.value,
                    record.split.value,
                    record.left_family.value,
                    record.right_family.value,
                    record.decision_count,
                )
                for record in calibration_surfaces.cofailures
            ),
        ),
        "",
        "## Calibration Saturated Metrics",
        _markdown_table(
            (
                "Split",
                "Tier",
                "Role",
                "Family",
                "Metric",
                "Count",
                "Pass Rate",
                "Fail Rate",
                "Unique",
                "Reasons",
            ),
            (
                (
                    _split_value(record.split),
                    record.tier_name.value if record.tier_name is not None else None,
                    record.role.value,
                    record.family,
                    record.metric_key,
                    record.sample_count,
                    record.pass_rate,
                    record.fail_rate,
                    record.unique_value_count,
                    record.reasons,
                )
                for record in calibration_surfaces.saturated_metrics
            ),
        ),
        "",
        "## Calibration Near-Threshold Samples",
        _markdown_table(
            (
                "Tier",
                "Split",
                "Role",
                "Family",
                "Metric",
                "Sample ID",
                "Side",
                "Actual",
                "Expected",
                "Comparison",
                "Distance",
            ),
            (
                (
                    record.tier_name.value,
                    record.split.value,
                    record.role.value,
                    record.family.value,
                    record.metric_key,
                    record.sample_id,
                    record.side.value,
                    record.actual_value,
                    record.expected_value,
                    record.comparison,
                    record.threshold_distance,
                )
                for record in calibration_surfaces.near_threshold_samples
            ),
        ),
        "",
        "## Calibration Tier Deltas",
        _markdown_table(
            ("Split", "From Tier", "To Tier", "Lost Sample Count"),
            (
                (
                    record.split.value,
                    record.from_tier.value,
                    record.to_tier.value,
                    record.sample_count,
                )
                for record in calibration_surfaces.tier_delta_summaries
            ),
        ),
        "",
        "## Calibration Coverage Summary",
        _markdown_table(
            ("Split", "Metric", "Count", "Min", "P50", "P95", "Max"),
            (
                (
                    _split_value(record.split),
                    record.metric_key,
                    record.sample_count,
                    record.minimum,
                    record.p50,
                    record.p95,
                    record.maximum,
                )
                for record in calibration_surfaces.coverage_summaries
            ),
        ),
        "",
        "## Calibration Confidence Channel Summary",
        _markdown_table(
            ("Split", "Metric", "Count", "Min", "P50", "P95", "Max"),
            (
                (
                    _split_value(record.split),
                    record.metric_key,
                    record.sample_count,
                    record.minimum,
                    record.p50,
                    record.p95,
                    record.maximum,
                )
                for record in calibration_surfaces.confidence_channel_summaries
            ),
        ),
        "",
    ]
    return "\n".join(sections)


def _split_value(split: object) -> str:
    if split is None:
        return "overall"
    if hasattr(split, "value"):
        return str(split.value)
    return str(split)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rendered_rows = tuple(tuple(_markdown_value(value) for value in row) for row in rows)
    if not rendered_rows:
        return "No records."
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(row) + " |"
        for row in rendered_rows
    ]
    return "\n".join((header, separator, *body))


def _markdown_value(value: object) -> str:
    jsonable = _jsonable(value)
    if jsonable is None:
        return ""
    if isinstance(jsonable, Mapping):
        text = ", ".join(f"{key}: {_markdown_value(item)}" for key, item in jsonable.items())
    elif isinstance(jsonable, list):
        text = "; ".join(_markdown_value(item) for item in jsonable)
    elif isinstance(jsonable, bool):
        text = "yes" if jsonable else "no"
    else:
        text = str(jsonable)
    return text.replace("|", "\\|").replace("\n", " ")


def _jsonable(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
