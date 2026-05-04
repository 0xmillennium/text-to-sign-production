"""Leakage bundle composition entry point."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.leakages.severity import (
    classify_leakage_severity,
    max_leakage_severity,
)
from text_to_sign_production.data.leakages.types import (
    LEAKAGE_RELATION_ORDER,
    LEAKAGE_RELATION_SPECS,
    LeakageBundle,
    LeakageInput,
    LeakagePairFact,
    LeakageRelation,
    LeakageSampleRef,
    LeakageSampleSummary,
    LeakageSeverity,
)

SampleKey = tuple[SampleSplit, str]
LeakageProgressCallback = Callable[..., None]


def build_leakage_bundle(
    inputs: Sequence[LeakageInput],
    *,
    progress_callback: LeakageProgressCallback | None = None,
) -> LeakageBundle:
    """Detect and compose deterministic leakage facts for a set of inputs."""
    seen: set[SampleKey] = set()
    for sample in inputs:
        key = (sample.split, sample.sample_id)
        if key in seen:
            raise ValueError(
                "Duplicate leakage input key: "
                f"split={sample.split.value!r}, sample_id={sample.sample_id!r}."
            )
        seen.add(key)
        if progress_callback is not None:
            progress_callback(advance=1, phase="duplicates", split=sample.split.value)

    sorted_inputs = sorted(inputs, key=lambda x: (x.split, x.sample_id))
    sample_by_key = {(sample.split, sample.sample_id): sample for sample in sorted_inputs}
    relations_by_pair: dict[tuple[SampleKey, SampleKey], set[LeakageRelation]] = {}

    for spec in LEAKAGE_RELATION_SPECS:
        _index_relation(
            sorted_inputs,
            spec.relation,
            _relation_key_fn(spec.input_field),
            relations_by_pair,
            progress_callback=progress_callback,
        )

    pair_facts: list[LeakagePairFact] = []
    relation_pairs = sorted(relations_by_pair)
    if progress_callback is not None:
        progress_callback(add_total=len(relation_pairs), phase="facts")
    for left_key, right_key in relation_pairs:
        pair_facts.append(
            _build_pair_fact(
                left_key,
                right_key,
                relations_by_pair[(left_key, right_key)],
                sample_by_key,
            )
        )
        if progress_callback is not None:
            progress_callback(advance=1, phase="facts", pairs=len(pair_facts))

    pair_facts.sort(
        key=lambda x: (x.left_split, x.left_sample_id, x.right_split, x.right_sample_id)
    )

    summary_pairs: dict[
        SampleKey,
        list[tuple[LeakagePairFact, SampleKey]],
    ] = {(sample.split, sample.sample_id): [] for sample in sorted_inputs}
    if progress_callback is not None:
        progress_callback(add_total=len(pair_facts), phase="summary_pairs")
    for pf in pair_facts:
        left_key = (pf.left_split, pf.left_sample_id)
        right_key = (pf.right_split, pf.right_sample_id)
        if left_key in summary_pairs:
            summary_pairs[left_key].append((pf, right_key))
        if right_key in summary_pairs:
            summary_pairs[right_key].append((pf, left_key))
        if progress_callback is not None:
            progress_callback(advance=1, phase="summary_pairs")

    sample_summaries: list[LeakageSampleSummary] = []
    if progress_callback is not None:
        progress_callback(add_total=len(sorted_inputs), phase="summaries")
    for sample in sorted_inputs:
        sample_key = (sample.split, sample.sample_id)
        matched_samples: set[SampleKey] = set()
        sample_max_severity = LeakageSeverity.NONE

        counts = {relation: 0 for relation in LEAKAGE_RELATION_ORDER}

        for pf, matched_sample_key in summary_pairs[sample_key]:
            matched_samples.add(matched_sample_key)
            sample_max_severity = max_leakage_severity(sample_max_severity, pf.severity)

            for rel in pf.relations:
                counts[rel] += 1

        has_leakage = sample_max_severity != LeakageSeverity.NONE

        sample_summaries.append(
            LeakageSampleSummary(
                sample_id=sample.sample_id,
                split=sample.split,
                has_leakage=has_leakage,
                max_severity=sample_max_severity,
                same_source_sentence_match_count=counts[LeakageRelation.SAME_SOURCE_SENTENCE],
                exact_normalized_text_match_count=counts[LeakageRelation.EXACT_NORMALIZED_TEXT],
                same_source_video_match_count=counts[LeakageRelation.SAME_SOURCE_VIDEO],
                matched_samples=tuple(
                    LeakageSampleRef(split=split, sample_id=sample_id)
                    for split, sample_id in sorted(matched_samples)
                ),
            ),
        )
        if progress_callback is not None:
            progress_callback(
                advance=1,
                phase="summaries",
                sample=sample.sample_id,
                has_leakage=has_leakage,
            )

    return LeakageBundle(
        pair_facts=tuple(pair_facts),
        sample_summaries=tuple(sample_summaries),
    )


def _index_relation(
    inputs: Sequence[LeakageInput],
    relation: LeakageRelation,
    key_fn: Callable[[LeakageInput], str],
    relations_by_pair: dict[tuple[SampleKey, SampleKey], set[LeakageRelation]],
    *,
    progress_callback: LeakageProgressCallback | None,
) -> None:
    groups: dict[str, list[LeakageInput]] = {}
    for sample in inputs:
        groups.setdefault(key_fn(sample), []).append(sample)
        if progress_callback is not None:
            progress_callback(
                advance=1,
                phase="index",
                relation=relation.value,
                split=sample.split.value,
            )

    candidate_pair_total = sum(
        len(group) * (len(group) - 1) // 2
        for group in groups.values()
        if len(group) >= 2
    )
    if progress_callback is not None:
        progress_callback(
            add_total=candidate_pair_total,
            phase="relations",
            relation=relation.value,
        )
    candidate_pairs_processed = 0
    relation_matches = 0
    for group in groups.values():
        if len(group) >= 2:
            sorted_group = sorted(group, key=lambda sample: (sample.split, sample.sample_id))
            for left, right in combinations(sorted_group, 2):
                candidate_pairs_processed += 1
                if left.split == right.split:
                    if progress_callback is not None:
                        progress_callback(
                            advance=1,
                            phase="relations",
                            relation=relation.value,
                            match_rate=_match_rate_text(
                                relation_matches,
                                candidate_pairs_processed,
                            ),
                        )
                    continue
                left_key: SampleKey = (left.split, left.sample_id)
                right_key: SampleKey = (right.split, right.sample_id)
                pair_key = (
                    (left_key, right_key) if left_key <= right_key else (right_key, left_key)
                )
                relations_by_pair.setdefault(pair_key, set()).add(relation)
                relation_matches += 1
                if progress_callback is not None:
                    progress_callback(
                        advance=1,
                        phase="relations",
                        relation=relation.value,
                        match_rate=_match_rate_text(
                            relation_matches,
                            candidate_pairs_processed,
                        ),
                    )


def _relation_key_fn(input_field: str) -> Callable[[LeakageInput], str]:
    def key_fn(sample: LeakageInput) -> str:
        return str(getattr(sample, input_field))

    return key_fn


def _match_rate_text(matches: int, processed: int) -> str:
    denominator = processed or 1
    return f"{matches / denominator * 100.0:.1f}%"


def _build_pair_fact(
    left_key: SampleKey,
    right_key: SampleKey,
    relation_set: set[LeakageRelation],
    sample_by_key: dict[SampleKey, LeakageInput],
) -> LeakagePairFact:
    left = sample_by_key[left_key]
    right = sample_by_key[right_key]
    relations = tuple(relation for relation in LEAKAGE_RELATION_ORDER if relation in relation_set)
    return LeakagePairFact(
        left_sample_id=left.sample_id,
        right_sample_id=right.sample_id,
        left_split=left.split,
        right_split=right.split,
        relations=relations,
        severity=classify_leakage_severity(relations),
    )
