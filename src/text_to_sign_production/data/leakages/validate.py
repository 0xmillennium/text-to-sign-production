"""Validation for the leakages package contracts."""

from __future__ import annotations

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.leakages.severity import (
    classify_leakage_severity,
    max_leakage_severity,
)
from text_to_sign_production.data.leakages.types import (
    LEAKAGE_RELATION_ORDER,
    LeakageBundle,
    LeakageRelation,
    LeakageSampleRef,
    LeakageSeverity,
    LeakageValidationIssue,
)


def validate_leakage_bundle(bundle: LeakageBundle) -> list[LeakageValidationIssue]:
    """Validate the deterministic leakage bundle."""
    issues: list[LeakageValidationIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(LeakageValidationIssue(code=code, message=message))

    # 1. Pair fact integrity
    seen_pairs: set[tuple[tuple[SampleSplit, str], tuple[SampleSplit, str]]] = set()
    for pf in bundle.pair_facts:
        left_key = (pf.left_split, pf.left_sample_id)
        right_key = (pf.right_split, pf.right_sample_id)
        if left_key == right_key:
            add("self_pair", f"Leakage pair references the same sample: {left_key}.")
        if pf.left_split == pf.right_split:
            add("same_split_pair", "Leakage pair is not cross-split.")

        pair_key = (left_key, right_key) if left_key <= right_key else (right_key, left_key)
        if pair_key in seen_pairs:
            add("duplicate_pair", f"Duplicate leakage pair: {pair_key}.")
        seen_pairs.add(pair_key)

        if not pf.relations:
            add("empty_relations", f"Leakage pair has no relations: {pair_key}.")

        expected_order = tuple(
            sorted(
                set(pf.relations),
                key=LEAKAGE_RELATION_ORDER.index,
            )
        )
        if pf.relations != expected_order:
            add("invalid_relation_order", f"Leakage pair relations are not canonical: {pair_key}.")

        expected_severity = classify_leakage_severity(pf.relations)
        if pf.severity != expected_severity:
            add("invalid_severity", f"Leakage pair severity mismatch: {pair_key}.")

    # 2. Sample summary integrity
    seen_samples: set[tuple[SampleSplit, str]] = set()
    for s in bundle.sample_summaries:
        key = (s.split, s.sample_id)
        if key in seen_samples:
            add("duplicate_summary", f"Duplicate leakage summary for {key}.")
        seen_samples.add(key)

        if s.same_source_sentence_match_count < 0:
            add("negative_count", f"Negative same-source-sentence count for {key}.")
        if s.exact_normalized_text_match_count < 0:
            add("negative_count", f"Negative exact-normalized-text count for {key}.")
        if s.same_source_video_match_count < 0:
            add("negative_count", f"Negative same-source-video count for {key}.")

        matched_keys = tuple((ref.split, ref.sample_id) for ref in s.matched_samples)
        if matched_keys != tuple(sorted(set(matched_keys))):
            add("matched_samples_not_sorted_or_unique", f"Matched samples invalid for {key}.")

        expected_has_leakage = s.max_severity != LeakageSeverity.NONE
        if s.has_leakage != expected_has_leakage:
            add("has_leakage_mismatch", f"has_leakage does not match severity for {key}.")

        if not s.has_leakage:
            if s.same_source_sentence_match_count > 0:
                add("nonzero_count_without_leakage", f"Nonzero relation count for {key}.")
            if s.exact_normalized_text_match_count > 0:
                add("nonzero_count_without_leakage", f"Nonzero relation count for {key}.")
            if s.same_source_video_match_count > 0:
                add("nonzero_count_without_leakage", f"Nonzero relation count for {key}.")
            if s.matched_samples:
                add(
                    "matched_samples_without_leakage",
                    f"Matched samples without leakage for {key}.",
                )
        else:
            if not s.matched_samples:
                add("missing_matched_samples_with_leakage", f"Missing matched samples for {key}.")

        # 3. Cross-consistency
        actual_matches: set[tuple[SampleSplit, str]] = set()
        actual_severity = LeakageSeverity.NONE
        actual_counts = {
            relation: 0
            for relation in LEAKAGE_RELATION_ORDER
        }
        for pf in bundle.pair_facts:
            left_key = (pf.left_split, pf.left_sample_id)
            right_key = (pf.right_split, pf.right_sample_id)
            if left_key == key:
                other = right_key
            elif right_key == key:
                other = left_key
            else:
                continue

            actual_matches.add(other)
            actual_severity = max_leakage_severity(actual_severity, pf.severity)
            for r in pf.relations:
                actual_counts[r] += 1

        expected_matched_samples = tuple(
            LeakageSampleRef(split=split, sample_id=sample_id)
            for split, sample_id in sorted(actual_matches)
        )
        if expected_matched_samples != s.matched_samples:
            add("matched_samples_cross_mismatch", f"Matched samples mismatch for {key}.")

        if actual_severity != s.max_severity:
            add("max_severity_cross_mismatch", f"Max severity mismatch for {key}.")

        if (
            actual_counts[LeakageRelation.SAME_SOURCE_SENTENCE]
            != s.same_source_sentence_match_count
        ):
            add("count_mismatch_same_source_sentence", f"Relation count mismatch for {key}.")
        if (
            actual_counts[LeakageRelation.EXACT_NORMALIZED_TEXT]
            != s.exact_normalized_text_match_count
        ):
            add("count_mismatch_exact_normalized_text", f"Relation count mismatch for {key}.")
        if actual_counts[LeakageRelation.SAME_SOURCE_VIDEO] != s.same_source_video_match_count:
            add("count_mismatch_same_source_video", f"Relation count mismatch for {key}.")

    return issues
