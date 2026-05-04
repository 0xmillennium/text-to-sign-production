"""Leakage bundle composition entry point."""

from __future__ import annotations

from collections.abc import Sequence

from text_to_sign_production.data.leakages.overlap import detect_pair_relations
from text_to_sign_production.data.leakages.severity import classify_leakage_severity
from text_to_sign_production.data.leakages.types import (
    LeakageBundle,
    LeakageInput,
    LeakagePairFact,
    LeakageRelation,
    LeakageSampleRef,
    LeakageSampleSummary,
    LeakageSeverity,
)


def build_leakage_bundle(inputs: Sequence[LeakageInput]) -> LeakageBundle:
    """Detect and compose deterministic leakage facts for a set of inputs."""
    sorted_inputs = sorted(inputs, key=lambda x: (x.split, x.sample_id))

    pair_facts: list[LeakagePairFact] = []

    n = len(sorted_inputs)
    for i in range(n):
        left = sorted_inputs[i]
        for j in range(i + 1, n):
            right = sorted_inputs[j]
            relations = detect_pair_relations(left, right)
            if relations is not None:
                severity = classify_leakage_severity(relations)
                pair_facts.append(
                    LeakagePairFact(
                        left_sample_id=left.sample_id,
                        right_sample_id=right.sample_id,
                        left_split=left.split,
                        right_split=right.split,
                        relations=relations,
                        severity=severity,
                    )
                )

    pair_facts.sort(
        key=lambda x: (x.left_split, x.left_sample_id, x.right_split, x.right_sample_id)
    )

    summary_pairs: dict[
        tuple[str, str],
        list[tuple[LeakagePairFact, tuple[str, str]]],
    ] = {(sample.split, sample.sample_id): [] for sample in sorted_inputs}
    for pf in pair_facts:
        left_key = (pf.left_split, pf.left_sample_id)
        right_key = (pf.right_split, pf.right_sample_id)
        if left_key in summary_pairs:
            summary_pairs[left_key].append((pf, right_key))
        if right_key in summary_pairs:
            summary_pairs[right_key].append((pf, left_key))

    sample_summaries: list[LeakageSampleSummary] = []
    for sample in sorted_inputs:
        sample_key = (sample.split, sample.sample_id)
        matched_samples: set[tuple[str, str]] = set()
        max_severity = LeakageSeverity.NONE

        counts = {
            LeakageRelation.SAME_SOURCE_SENTENCE: 0,
            LeakageRelation.EXACT_NORMALIZED_TEXT: 0,
            LeakageRelation.SAME_SOURCE_VIDEO: 0,
        }

        # A mapping from Severity to integer for max evaluation
        severity_val = {
            LeakageSeverity.NONE: 0,
            LeakageSeverity.LOW: 1,
            LeakageSeverity.MEDIUM: 2,
            LeakageSeverity.HIGH: 3,
        }

        for pf, matched_sample_key in summary_pairs[sample_key]:
            matched_samples.add(matched_sample_key)
            if severity_val[pf.severity] > severity_val[max_severity]:
                max_severity = pf.severity

            for rel in pf.relations:
                counts[rel] += 1

        has_leakage = max_severity != LeakageSeverity.NONE

        sample_summaries.append(
            LeakageSampleSummary(
                sample_id=sample.sample_id,
                split=sample.split,
                has_leakage=has_leakage,
                max_severity=max_severity,
                same_source_sentence_match_count=counts[LeakageRelation.SAME_SOURCE_SENTENCE],
                exact_normalized_text_match_count=counts[LeakageRelation.EXACT_NORMALIZED_TEXT],
                same_source_video_match_count=counts[LeakageRelation.SAME_SOURCE_VIDEO],
                matched_samples=tuple(
                    LeakageSampleRef(split=split, sample_id=sample_id)
                    for split, sample_id in sorted(matched_samples)
                ),
            )
        )

    return LeakageBundle(
        pair_facts=tuple(pair_facts),
        sample_summaries=tuple(sample_summaries),
    )
