"""Markdown and JSON reports for candidate-agnostic validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_to_sign_production.modeling.validation.comparison import ValidationComparisonResult


@dataclass(frozen=True, slots=True)
class ValidationEngineReportPaths:
    surface_summary_path: Path
    comparison_report_path: Path
    limitations_report_path: Path
    retrieval_metadata_report_path: Path
    metric_interpretation_path: Path
    comparison_json_path: Path | None
    retrieval_metadata_json_path: Path | None


def write_validation_engine_reports(
    *,
    report_root: Path,
    surface_results: tuple[object, ...],
    comparison: ValidationComparisonResult | None,
) -> ValidationEngineReportPaths:
    report_root.mkdir(parents=True, exist_ok=True)
    surface_summary_path = report_root / "surface_summary.md"
    comparison_report_path = report_root / "comparison_report.md"
    limitations_report_path = report_root / "limitations_report.md"
    retrieval_metadata_report_path = report_root / "retrieval_metadata_report.md"
    metric_interpretation_path = report_root / "metric_interpretation.md"
    comparison_json_path = report_root / "comparison_table.json" if comparison is not None else None
    retrieval_metadata_json_path = report_root / "retrieval_metadata.json"

    surface_summary_path.write_text(_surface_summary(surface_results), encoding="utf-8")
    comparison_report_path.write_text(_comparison_report(comparison), encoding="utf-8")
    limitations_report_path.write_text(_limitations_report(surface_results), encoding="utf-8")
    retrieval_metadata = _retrieval_metadata(surface_results)
    retrieval_metadata_report_path.write_text(
        _retrieval_metadata_report(retrieval_metadata),
        encoding="utf-8",
    )
    metric_interpretation_path.write_text(_metric_interpretation_report(), encoding="utf-8")
    if comparison_json_path is not None and comparison is not None:
        _write_json(comparison_json_path, comparison.to_dict())
    _write_json(retrieval_metadata_json_path, {"surfaces": retrieval_metadata})
    return ValidationEngineReportPaths(
        surface_summary_path=surface_summary_path,
        comparison_report_path=comparison_report_path,
        limitations_report_path=limitations_report_path,
        retrieval_metadata_report_path=retrieval_metadata_report_path,
        metric_interpretation_path=metric_interpretation_path,
        comparison_json_path=comparison_json_path,
        retrieval_metadata_json_path=retrieval_metadata_json_path,
    )


def _surface_summary(surface_results: tuple[object, ...]) -> str:
    rows = []
    for result in surface_results:
        surface = result.surface
        limitations = result.limitations
        rows.append(
            "| "
            + " | ".join(
                (
                    surface.label,
                    surface.producer_type,
                    surface.producer_key,
                    surface.run_name,
                    str(limitations.paired_count),
                    str(limitations.generated_missing_count),
                    str(limitations.failed_generated_count),
                    str(limitations.identity_mismatch_count),
                )
            )
            + " |"
        )
    return (
        "# Validation Surface Summary\n\n"
        "| Surface | Producer type | Producer key | Run | Paired | Missing generated | Generated failed | Identity mismatch |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\nRetrieval comparator surfaces reuse whole source poses and must not be read as learned generation.\n"
    )


def _comparison_report(comparison: ValidationComparisonResult | None) -> str:
    if comparison is None:
        return "# Validation Comparison Report\n\nComparison was disabled.\n"
    if comparison.status != "ok":
        return f"# Validation Comparison Report\n\n{comparison.message}\n"
    rows = []
    for row in comparison.rows:
        best = "true" if row.is_best_by_metric else "false"
        rows.append(
            "| "
            + " | ".join(
                (
                    row.metric_key,
                    row.surface_label,
                    row.direction or "unranked",
                    str(row.count),
                    str(row.mean),
                    str(row.rank),
                    best,
                    row.availability_status,
                )
            )
            + " |"
        )
    return (
        "# Validation Comparison Report\n\n"
        "Rows identify `best_by_metric` only. They do not identify model-level, semantic, sign-quality, or intelligibility winners.\n\n"
        f"- Shared paired subset required: `{comparison.require_shared_pairing_subset}`\n"
        f"- Shared paired key count: `{len(comparison.shared_pairing_subset_keys)}`\n\n"
        "| Metric | Surface | Direction | Count | Mean | Rank | best_by_metric | Availability |\n"
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )


def _limitations_report(surface_results: tuple[object, ...]) -> str:
    blocks = []
    for result in surface_results:
        notes = "\n".join(f"- {note}" for note in result.limitations.notes)
        blocks.append(
            f"## {result.surface.label}\n\n"
            f"- Paired: `{result.limitations.paired_count}`\n"
            f"- Missing generated: `{result.limitations.generated_missing_count}`\n"
            f"- Missing reference: `{result.limitations.reference_missing_count}`\n"
            f"- Failed generated: `{result.limitations.failed_generated_count}`\n"
            f"- Identity mismatch: `{result.limitations.identity_mismatch_count}`\n\n"
            f"{notes}\n"
        )
    return "# Validation Limitations Report\n\n" + "\n".join(blocks)


def _retrieval_metadata(surface_results: tuple[object, ...]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for result in surface_results:
        surface = result.surface
        if not surface.is_comparator:
            continue
        entries = surface.entries
        scores = np.asarray(
            [entry.retrieval_score for entry in entries if entry.retrieval_score is not None],
            dtype=np.float64,
        )
        leakage_values = [entry.leakage_checked for entry in entries if entry.leakage_checked is not None]
        source_split_counts: dict[str, int] = {}
        for entry in entries:
            if entry.retrieval_source_split is not None:
                key = entry.retrieval_source_split.value
                source_split_counts[key] = source_split_counts.get(key, 0) + 1
        summaries.append(
            {
                "surface_label": surface.label,
                "producer_key": surface.producer_key,
                "retrieved_sample_count": sum(entry.retrieved_sample_id is not None for entry in entries),
                "failed_retrieval_count": sum(entry.failure_reason is not None for entry in entries),
                "leakage_checked_ratio": (
                    None
                    if not leakage_values
                    else sum(value is True for value in leakage_values) / len(leakage_values)
                ),
                "retrieval_score": _score_distribution(scores),
                "retrieval_source_split_counts": source_split_counts,
            }
        )
    return summaries


def _retrieval_metadata_report(summaries: list[dict[str, object]]) -> str:
    if not summaries:
        return "# Retrieval Metadata Report\n\nNo retrieval comparator surfaces were validated.\n"
    blocks = []
    for summary in summaries:
        blocks.append(
            f"## {summary['surface_label']}\n\n"
            f"- Retrieved sample count: `{summary['retrieved_sample_count']}`\n"
            f"- Failed retrieval count: `{summary['failed_retrieval_count']}`\n"
            f"- Leakage checked ratio: `{summary['leakage_checked_ratio']}`\n"
            f"- Retrieval score distribution: `{summary['retrieval_score']}`\n"
            f"- Retrieval source split summary: `{summary['retrieval_source_split_counts']}`\n\n"
            "Retrieval scores are companion retrieval evidence, not pose accuracy metrics.\n"
        )
    return "# Retrieval Metadata Report\n\n" + "\n".join(blocks)


def _metric_interpretation_report() -> str:
    return (
        "# Metric Interpretation\n\n"
        "- Automatic pose/keypoint metrics do not prove sign intelligibility.\n"
        "- Validation split results are not final test performance.\n"
        "- Prefix alignment does not solve temporal alignment.\n"
        "- Retrieval comparator uses real source poses and must not be read as generated motion.\n"
        "- Semantic-objective variants require with/without-objective ablation to support contribution claims.\n"
    )


def _score_distribution(scores: np.ndarray) -> dict[str, object]:
    if scores.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(scores.size),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")


__all__ = ["ValidationEngineReportPaths", "write_validation_engine_reports"]
