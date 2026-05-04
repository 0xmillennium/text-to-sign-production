"""Internal report helpers for the samples workflow."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.gates.types import GateStage, ProcessingDecision
from text_to_sign_production.data.samples.types import DroppedManifestEntry, PassedManifestEntry
from text_to_sign_production.data.sources.types import SourceMatchResult
from text_to_sign_production.foundation.progress import ItemProgress, ProgressReporter


def write_samples_reports(
    reports_root: Path,
    *,
    processing_decisions: Mapping[SampleSplit, tuple[ProcessingDecision, ...]],
    passed_entries: Mapping[SampleSplit, tuple[PassedManifestEntry, ...]],
    dropped_entries: Mapping[SampleSplit, tuple[DroppedManifestEntry, ...]],
    source_matches: Mapping[SampleSplit, tuple[SourceMatchResult, ...]],
    reporter: ProgressReporter | None = None,
    progress_label: str | None = None,
) -> tuple[Path, ...]:
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / "samples_report.md"
    content = _render_samples_report(
        processing_records=build_samples_processing_summary_records(
            passed_entries,
            dropped_entries,
        ),
        gate_records=build_samples_gate_summary_records(processing_decisions),
        drop_reason_records=build_samples_drop_reason_records(dropped_entries),
        source_issue_records=build_samples_source_issue_records(source_matches),
    )
    if reporter is None:
        _write_markdown(report_path, content)
        return (report_path,)

    with ItemProgress(
        label=progress_label or "write samples report",
        total=1,
        unit="report",
        reporter=reporter,
    ) as progress:
        _write_markdown(report_path, content)
        progress.advance()
    return (report_path,)


def build_samples_processing_summary_records(
    passed_entries: Mapping[SampleSplit, tuple[PassedManifestEntry, ...]],
    dropped_entries: Mapping[SampleSplit, tuple[DroppedManifestEntry, ...]],
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for split in _ordered_splits(passed_entries, dropped_entries):
        passed_count = len(passed_entries.get(split, ()))
        dropped_count = len(dropped_entries.get(split, ()))
        records.append(
            {
                "split": split.value,
                "passed_count": passed_count,
                "dropped_count": dropped_count,
                "total_count": passed_count + dropped_count,
            }
        )
    return tuple(records)


def build_samples_gate_summary_records(
    processing_decisions: Mapping[SampleSplit, tuple[ProcessingDecision, ...]],
) -> tuple[dict[str, object], ...]:
    counters: dict[tuple[SampleSplit, GateStage], Counter[str]] = defaultdict(Counter)
    for split, decisions in processing_decisions.items():
        for decision in decisions:
            for stage, result in decision.gate_results.items():
                counters[(split, stage)][result.status.value] += 1

    records: list[dict[str, object]] = []
    for split, stage in sorted(counters, key=lambda item: (item[0].value, item[1].value)):
        counts = counters[(split, stage)]
        records.append(
            {
                "split": split.value,
                "gate_stage": stage.value,
                "passed_count": counts.get("passed", 0),
                "dropped_count": counts.get("dropped", 0),
            }
        )
    return tuple(records)


def build_samples_drop_reason_records(
    dropped_entries: Mapping[SampleSplit, tuple[DroppedManifestEntry, ...]],
) -> tuple[dict[str, object], ...]:
    counters: dict[tuple[SampleSplit, str, str], int] = defaultdict(int)
    for split, entries in dropped_entries.items():
        for entry in entries:
            for reason in entry.drop_reasons:
                counters[(split, entry.drop_stage, reason)] += 1

    records: list[dict[str, object]] = []
    for split, stage, reason in sorted(
        counters,
        key=lambda item: (item[0].value, item[1], item[2]),
    ):
        records.append(
            {
                "split": split.value,
                "drop_stage": stage,
                "drop_reason": reason,
                "count": counters[(split, stage, reason)],
            }
        )
    return tuple(records)


def build_samples_source_issue_records(
    source_matches: Mapping[SampleSplit, tuple[SourceMatchResult, ...]],
) -> tuple[dict[str, object], ...]:
    counters: dict[tuple[SampleSplit, str], int] = defaultdict(int)
    for split, matches in source_matches.items():
        for match in matches:
            issues = _ordered_unique(
                (
                    *match.source_issues,
                    *((match.unmatched_reason,) if match.unmatched_reason is not None else ()),
                )
            )
            for issue in issues:
                counters[(split, issue)] += 1

    records: list[dict[str, object]] = []
    for split, issue in sorted(counters, key=lambda item: (item[0].value, item[1])):
        records.append(
            {
                "split": split.value,
                "source_issue": issue,
                "count": counters[(split, issue)],
            }
        )
    return tuple(records)


def _ordered_splits(*mappings: Mapping[SampleSplit, object]) -> tuple[SampleSplit, ...]:
    return tuple(
        sorted(
            {split for mapping in mappings for split in mapping},
            key=lambda split: split.value,
        )
    )


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _render_samples_report(
    *,
    processing_records: Sequence[Mapping[str, object]],
    gate_records: Sequence[Mapping[str, object]],
    drop_reason_records: Sequence[Mapping[str, object]],
    source_issue_records: Sequence[Mapping[str, object]],
) -> str:
    sections = [
        "# Samples Report",
        "",
        "## Processing Summary",
        _markdown_table(
            ("Split", "Passed", "Dropped", "Total"),
            (
                (
                    record["split"],
                    record["passed_count"],
                    record["dropped_count"],
                    record["total_count"],
                )
                for record in processing_records
            ),
        ),
        "",
        "## Gate Summary",
        _markdown_table(
            ("Split", "Gate", "Passed", "Dropped"),
            (
                (
                    record["split"],
                    record["gate_stage"],
                    record["passed_count"],
                    record["dropped_count"],
                )
                for record in gate_records
            ),
        ),
        "",
        "## Drop Reasons",
        _markdown_table(
            ("Split", "Stage", "Reason", "Count"),
            (
                (
                    record["split"],
                    record["drop_stage"],
                    record["drop_reason"],
                    record["count"],
                )
                for record in drop_reason_records
            ),
        ),
        "",
        "## Source Issues",
        _markdown_table(
            ("Split", "Issue", "Count"),
            (
                (
                    record["split"],
                    record["source_issue"],
                    record["count"],
                )
                for record in source_issue_records
            ),
        ),
        "",
    ]
    return "\n".join(sections)


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
    text = _jsonable(value)
    if text is None:
        return ""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value
