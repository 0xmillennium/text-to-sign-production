"""Validation for quality-domain leakage facts."""

from __future__ import annotations

from text_to_sign_production.data.tier.leakages.types import (
    LEAKAGE_RELATION_ORDER,
    LeakageBundle,
    LeakageInput,
    LeakageValidationIssue,
    LeakageValidationIssueCode,
)


def validate_leakage_input(input: LeakageInput) -> tuple[LeakageValidationIssue, ...]:
    """Validate one leakage input."""
    if not input.canonical_normalized_text:
        return (
            LeakageValidationIssue(
                LeakageValidationIssueCode.NORMALIZED_TEXT_AUTHORITY_MISSING,
                "Leakage input requires authoritative canonical_normalized_text.",
                "canonical_normalized_text",
            ),
        )
    return ()


def validate_leakage_bundle(bundle: LeakageBundle) -> tuple[LeakageValidationIssue, ...]:
    """Validate a leakage bundle."""
    issues: list[LeakageValidationIssue] = []
    seen_pairs: set[tuple[object, ...]] = set()
    summary_keys = {(summary.split, summary.sample_id) for summary in bundle.sample_summaries}
    if len(summary_keys) != len(bundle.sample_summaries):
        issues.append(
            LeakageValidationIssue(
                LeakageValidationIssueCode.DUPLICATE_SAMPLE_SUMMARY,
                "Duplicate leakage sample summary.",
                "sample_summaries",
            )
        )
    for index, fact in enumerate(bundle.pair_facts):
        pair_key = (
            fact.left_split,
            fact.left_sample_id,
            fact.right_split,
            fact.right_sample_id,
        )
        if pair_key in seen_pairs:
            issues.append(
                LeakageValidationIssue(
                    LeakageValidationIssueCode.DUPLICATE_PAIR,
                    "Duplicate leakage pair.",
                    f"pair_facts.{index}",
                )
            )
        seen_pairs.add(pair_key)
        if fact.left_sample_id == fact.right_sample_id and fact.left_split is fact.right_split:
            issues.append(
                LeakageValidationIssue(
                    LeakageValidationIssueCode.SELF_PAIR,
                    "Leakage pair cannot reference itself.",
                    f"pair_facts.{index}",
                )
            )
        if fact.left_split is fact.right_split:
            issues.append(
                LeakageValidationIssue(
                    LeakageValidationIssueCode.SAME_SPLIT_PAIR,
                    "Leakage pair must be cross-split.",
                    f"pair_facts.{index}",
                )
            )
        if not fact.relations:
            issues.append(
                LeakageValidationIssue(
                    LeakageValidationIssueCode.EMPTY_RELATIONS,
                    "Leakage pair requires at least one relation.",
                    f"pair_facts.{index}.relations",
                )
            )
        if fact.relations != tuple(
            relation for relation in LEAKAGE_RELATION_ORDER if relation in fact.relations
        ):
            issues.append(
                LeakageValidationIssue(
                    LeakageValidationIssueCode.INVALID_RELATION_ORDER,
                    "Leakage relations must follow canonical order.",
                    f"pair_facts.{index}.relations",
                )
            )
        if (fact.left_split, fact.left_sample_id) not in summary_keys or (
            fact.right_split,
            fact.right_sample_id,
        ) not in summary_keys:
            issues.append(
                LeakageValidationIssue(
                    LeakageValidationIssueCode.MISSING_SAMPLE_SUMMARY_FOR_PAIR,
                    "Each leakage pair endpoint must have a sample summary.",
                    f"pair_facts.{index}",
                )
            )
    for index, summary in enumerate(bundle.sample_summaries):
        for count in (
            summary.same_source_sentence_match_count,
            summary.exact_normalized_text_match_count,
            summary.same_source_video_match_count,
        ):
            if count < 0:
                issues.append(
                    LeakageValidationIssue(
                        LeakageValidationIssueCode.NEGATIVE_COUNT,
                        "Leakage summary counts cannot be negative.",
                        f"sample_summaries.{index}",
                    )
                )
    return tuple(issues)


__all__ = ["validate_leakage_bundle", "validate_leakage_input"]
