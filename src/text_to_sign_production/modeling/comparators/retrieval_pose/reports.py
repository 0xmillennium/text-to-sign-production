"""Cautious Markdown reports for the retrieval comparator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from text_to_sign_production.modeling.comparators.retrieval_pose.config import (
    RetrievalPoseComparatorConfig,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.leakage import (
    summarize_leakage,
)


def write_retrieval_reports(
    *,
    report_root: Path,
    config: RetrievalPoseComparatorConfig,
    results: tuple[object, ...],
) -> tuple[Path, ...]:
    report_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "retrieval_policy.md": _policy(config),
        "leakage_report.md": _leakage(config, results),
        "retrieval_results_summary.md": _summary(results),
        "comparator_report.md": _comparator(config),
        "metric_limitations.md": _limitations(),
    }
    written: list[Path] = []
    for filename, content in paths.items():
        path = report_root / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return tuple(written)


def _policy(config: RetrievalPoseComparatorConfig) -> str:
    leakage = config.leakage
    return "\n".join(
        [
            "# Retrieval Policy",
            "",
            f"- Bank split: `{config.data.bank_split.value}`",
            f"- Query splits: `{', '.join(split.value for split in config.data.query_splits)}`",
            f"- Scoring metric: `{config.retrieval.score_metric}`",
            f"- Selection policy: `{config.retrieval.selection_policy}`",
            f"- Reported top-k: `{config.retrieval.top_k}`",
            "- Scope: whole-pose reuse only.",
            "- Out of scope: stitching, gloss lookup, dictionaries, segmentation, avatar, and rendering.",
            f"- Same split allowed: `{leakage.allow_same_split}`",
            f"- Exclude same sample id: `{leakage.exclude_same_sample_id}`",
            f"- Exclude same source sentence id: `{leakage.exclude_same_source_sentence_id}`",
            f"- Exclude same source video id: `{leakage.exclude_same_source_video_id}`",
            f"- Exclude identical text: `{leakage.exclude_identical_text}`",
            f"- Fail on leakage: `{leakage.fail_on_leakage}`",
            "",
        ]
    )


def _leakage(config: RetrievalPoseComparatorConfig, results: tuple[object, ...]) -> str:
    summary = summarize_leakage(results)
    lines = [
        "# Leakage Report",
        "",
        f"- Same split policy allows same split: `{config.leakage.allow_same_split}`",
        f"- Leakage checked: `{summary.leakage_checked}`",
        f"- Blocked candidates: `{summary.blocked_candidates}`",
        f"- Queries without safe candidate: `{summary.queries_without_safe_candidate}`",
        "",
        "## Blocked Reasons",
        "",
    ]
    if summary.blocked_by_reason:
        lines.extend(f"- `{reason}`: `{count}`" for reason, count in summary.blocked_by_reason.items())
    else:
        lines.append("- None observed.")
    lines.append("")
    return "\n".join(lines)


def _summary(results: tuple[object, ...]) -> str:
    success = [result for result in results if getattr(result, "success", False)]
    scores = [
        float(result.selected_candidate.score)
        for result in success
        if getattr(result, "selected_candidate", None) is not None
    ]
    lengths = [
        int(result.selected_candidate.valid_frame_count)
        for result in success
        if getattr(result, "selected_candidate", None) is not None
    ]
    return "\n".join(
        [
            "# Retrieval Results Summary",
            "",
            f"- Query count: `{len(results)}`",
            f"- Success count: `{len(success)}`",
            f"- Failure count: `{len(results) - len(success)}`",
            f"- Score distribution: `{_describe(scores)}`",
            f"- Retrieved valid length distribution: `{_describe(lengths)}`",
            "",
        ]
    )


def _comparator(config: RetrievalPoseComparatorConfig) -> str:
    return "\n".join(
        [
            "# Comparator Report",
            "",
            f"- Comparator key: `{config.identity.comparator_key}`",
            f"- Canonical id: `{config.identity.canonical_id}`",
            f"- Research role: `{config.identity.research_role}`",
            "- Registry role: counter alternative comparator; not a primary learned model.",
            "- Generated-pose contract compatibility: writes existing generated-pose payload and manifest schemas.",
            "- Comparison readiness depends on leakage policy, shared evaluation metrics, and matching query splits.",
            "- Retrieved pose is real source data, not generated motion.",
            "- Retrieval realism is not semantic correctness.",
            "",
        ]
    )


def _limitations() -> str:
    return "\n".join(
        [
            "# Metric Limitations",
            "",
            "- Retrieval realism is not semantic correctness.",
            "- Retrieved pose is real source data, not generated motion.",
            "- This comparator is a leakage-safe counter baseline, not a learned generator.",
            "- Automatic metrics do not prove sign intelligibility.",
            "- Leakage policy determines whether comparison is meaningful.",
            "",
        ]
    )


def _describe(values: list[float] | list[int]) -> str:
    if not values:
        return "n=0"
    array = np.asarray(values, dtype=np.float64)
    return (
        f"n={array.size}, min={array.min():.6g}, mean={array.mean():.6g}, "
        f"max={array.max():.6g}"
    )


__all__ = ["write_retrieval_reports"]
