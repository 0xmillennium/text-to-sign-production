"""Validation for the leakages package contracts."""

from __future__ import annotations

from text_to_sign_production.data.leakages.severity import classify_leakage_severity
from text_to_sign_production.data.leakages.types import (
    LeakageBundle,
    LeakageRelation,
    LeakageSampleRef,
    LeakageSeverity,
)


def validate_leakage_bundle(bundle: LeakageBundle) -> list[str]:
    """Validate the deterministic leakage bundle."""
    issues: list[str] = []

    # 1. Pair fact integrity
    seen_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for pf in bundle.pair_facts:
        left_key = (pf.left_split, pf.left_sample_id)
        right_key = (pf.right_split, pf.right_sample_id)
        if left_key == right_key:
            issues.append(f"self_pair:{left_key}")
        if pf.left_split == pf.right_split:
            issues.append(f"same_split_pair:{pf.left_sample_id}_{pf.right_sample_id}")

        pair_key = (left_key, right_key) if left_key <= right_key else (right_key, left_key)
        if pair_key in seen_pairs:
            issues.append(f"duplicate_pair:{pair_key}")
        seen_pairs.add(pair_key)

        if not pf.relations:
            issues.append(f"empty_relations:{pair_key}")

        expected_order = tuple(
            sorted(
                set(pf.relations),
                key=lambda r: [
                    LeakageRelation.SAME_SOURCE_SENTENCE,
                    LeakageRelation.EXACT_NORMALIZED_TEXT,
                    LeakageRelation.SAME_SOURCE_VIDEO,
                ].index(r),
            )
        )
        if pf.relations != expected_order:
            issues.append(f"invalid_relation_order:{pair_key}")

        expected_severity = classify_leakage_severity(pf.relations)
        if pf.severity != expected_severity:
            issues.append(f"invalid_severity:{pair_key}:{pf.severity}!={expected_severity}")

    # 2. Sample summary integrity
    seen_samples: set[tuple[str, str]] = set()
    for s in bundle.sample_summaries:
        key = (s.split, s.sample_id)
        if key in seen_samples:
            issues.append(f"duplicate_summary:{key}")
        seen_samples.add(key)

        if s.same_source_sentence_match_count < 0:
            issues.append(f"negative_count:{key}")
        if s.exact_normalized_text_match_count < 0:
            issues.append(f"negative_count:{key}")
        if s.same_source_video_match_count < 0:
            issues.append(f"negative_count:{key}")

        matched_keys = tuple((ref.split, ref.sample_id) for ref in s.matched_samples)
        if matched_keys != tuple(sorted(set(matched_keys))):
            issues.append(f"matched_samples_not_sorted_or_unique:{key}")

        expected_has_leakage = s.max_severity != LeakageSeverity.NONE
        if s.has_leakage != expected_has_leakage:
            issues.append(f"has_leakage_mismatch:{key}")

        if not s.has_leakage:
            if s.same_source_sentence_match_count > 0:
                issues.append(f"nonzero_count_without_leakage:{key}")
            if s.exact_normalized_text_match_count > 0:
                issues.append(f"nonzero_count_without_leakage:{key}")
            if s.same_source_video_match_count > 0:
                issues.append(f"nonzero_count_without_leakage:{key}")
            if s.matched_samples:
                issues.append(f"matched_samples_without_leakage:{key}")
        else:
            if not s.matched_samples:
                issues.append(f"missing_matched_samples_with_leakage:{key}")

        # 3. Cross-consistency
        actual_matches: set[tuple[str, str]] = set()
        actual_severity = LeakageSeverity.NONE
        actual_counts = {
            LeakageRelation.SAME_SOURCE_SENTENCE: 0,
            LeakageRelation.EXACT_NORMALIZED_TEXT: 0,
            LeakageRelation.SAME_SOURCE_VIDEO: 0,
        }
        sev_val = {
            LeakageSeverity.NONE: 0,
            LeakageSeverity.LOW: 1,
            LeakageSeverity.MEDIUM: 2,
            LeakageSeverity.HIGH: 3,
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
            if sev_val[pf.severity] > sev_val[actual_severity]:
                actual_severity = pf.severity
            for r in pf.relations:
                actual_counts[r] += 1

        expected_matched_samples = tuple(
            LeakageSampleRef(split=split, sample_id=sample_id)
            for split, sample_id in sorted(actual_matches)
        )
        if expected_matched_samples != s.matched_samples:
            issues.append(f"matched_samples_cross_mismatch:{key}")

        if actual_severity != s.max_severity:
            issues.append(f"max_severity_cross_mismatch:{key}")

        if (
            actual_counts[LeakageRelation.SAME_SOURCE_SENTENCE]
            != s.same_source_sentence_match_count
        ):
            issues.append(f"count_mismatch_same_source_sentence:{key}")
        if (
            actual_counts[LeakageRelation.EXACT_NORMALIZED_TEXT]
            != s.exact_normalized_text_match_count
        ):
            issues.append(f"count_mismatch_exact_normalized_text:{key}")
        if actual_counts[LeakageRelation.SAME_SOURCE_VIDEO] != s.same_source_video_match_count:
            issues.append(f"count_mismatch_same_source_video:{key}")

    return issues
