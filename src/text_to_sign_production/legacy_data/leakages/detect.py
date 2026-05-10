"""Leakage bundle composition entry point."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Protocol

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.legacy_data.leakages.severity import (
    classify_leakage_severity,
    max_leakage_severity,
)
from text_to_sign_production.legacy_data.leakages.types import (
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
LeakageProgressKind = Literal["start", "advance", "finish"]
LeakageProgressPhase = Literal[
    "duplicates",
    "index_relations",
    "relations",
    "facts",
    "summary_pairs",
    "summaries",
]
RelationGroups = dict[LeakageRelation, dict[str, list[LeakageInput]]]


@dataclass(frozen=True, slots=True)
class LeakageProgressEvent:
    kind: LeakageProgressKind
    phase: LeakageProgressPhase
    total: int | None = None
    split: SampleSplit | None = None
    sample_id: str | None = None
    relation: LeakageRelation | None = None
    matches: int | None = None
    leakage: LeakageSeverity | None = None


class LeakageProgressSink(Protocol):
    def update_leakage_progress(self, event: LeakageProgressEvent) -> None: ...


def build_leakage_bundle(
    inputs: Sequence[LeakageInput],
    *,
    progress_sink: LeakageProgressSink | None = None,
) -> LeakageBundle:
    """Detect and compose deterministic leakage facts for a set of inputs."""
    seen: set[SampleKey] = set()
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(
                kind="start",
                phase="duplicates",
                total=len(inputs),
            )
        )
    for sample in inputs:
        key = (sample.split, sample.sample_id)
        if key in seen:
            raise ValueError(
                "Duplicate leakage input key: "
                f"split={sample.split.value!r}, sample_id={sample.sample_id!r}."
            )
        seen.add(key)
        if progress_sink is not None:
            progress_sink.update_leakage_progress(
                LeakageProgressEvent(
                    kind="advance",
                    phase="duplicates",
                    split=sample.split,
                    sample_id=sample.sample_id,
                )
            )
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(kind="finish", phase="duplicates")
        )

    sorted_inputs = sorted(inputs, key=lambda x: (x.split, x.sample_id))
    sample_by_key = {(sample.split, sample.sample_id): sample for sample in sorted_inputs}
    relations_by_pair: dict[tuple[SampleKey, SampleKey], set[LeakageRelation]] = {}

    relation_groups = _index_relation_groups(
        sorted_inputs,
        progress_sink=progress_sink,
    )
    _scan_relation_pairs(
        relation_groups,
        relations_by_pair,
        progress_sink=progress_sink,
    )

    pair_facts: list[LeakagePairFact] = []
    relation_pairs = sorted(relations_by_pair)
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(kind="start", phase="facts", total=len(relation_pairs))
        )
    for left_key, right_key in relation_pairs:
        pair_facts.append(
            _build_pair_fact(
                left_key,
                right_key,
                relations_by_pair[(left_key, right_key)],
                sample_by_key,
            )
        )
        if progress_sink is not None:
            progress_sink.update_leakage_progress(
                LeakageProgressEvent(kind="advance", phase="facts")
            )
    if progress_sink is not None:
        progress_sink.update_leakage_progress(LeakageProgressEvent(kind="finish", phase="facts"))

    pair_facts.sort(
        key=lambda x: (x.left_split, x.left_sample_id, x.right_split, x.right_sample_id)
    )

    summary_pairs: dict[
        SampleKey,
        list[tuple[LeakagePairFact, SampleKey]],
    ] = {(sample.split, sample.sample_id): [] for sample in sorted_inputs}
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(kind="start", phase="summary_pairs", total=len(pair_facts))
        )
    for pf in pair_facts:
        left_key = (pf.left_split, pf.left_sample_id)
        right_key = (pf.right_split, pf.right_sample_id)
        if left_key in summary_pairs:
            summary_pairs[left_key].append((pf, right_key))
        if right_key in summary_pairs:
            summary_pairs[right_key].append((pf, left_key))
        if progress_sink is not None:
            progress_sink.update_leakage_progress(
                LeakageProgressEvent(kind="advance", phase="summary_pairs")
            )
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(kind="finish", phase="summary_pairs")
        )

    sample_summaries: list[LeakageSampleSummary] = []
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(kind="start", phase="summaries", total=len(sorted_inputs))
        )
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
        if progress_sink is not None:
            progress_sink.update_leakage_progress(
                LeakageProgressEvent(
                    kind="advance",
                    phase="summaries",
                    split=sample.split,
                    sample_id=sample.sample_id,
                    leakage=sample_max_severity,
                )
            )
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(kind="finish", phase="summaries")
        )

    return LeakageBundle(
        pair_facts=tuple(pair_facts),
        sample_summaries=tuple(sample_summaries),
    )


def _index_relation_groups(
    inputs: Sequence[LeakageInput],
    *,
    progress_sink: LeakageProgressSink | None,
) -> RelationGroups:
    relation_groups: RelationGroups = {}
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(
                kind="start",
                phase="index_relations",
                total=len(inputs) * len(LEAKAGE_RELATION_SPECS),
            )
        )
    for spec in LEAKAGE_RELATION_SPECS:
        groups: dict[str, list[LeakageInput]] = {}
        key_fn = _relation_key_fn(spec.input_field)
        for sample in inputs:
            groups.setdefault(key_fn(sample), []).append(sample)
            if progress_sink is not None:
                progress_sink.update_leakage_progress(
                    LeakageProgressEvent(
                        kind="advance",
                        phase="index_relations",
                        relation=spec.relation,
                        split=sample.split,
                        sample_id=sample.sample_id,
                    )
                )
        relation_groups[spec.relation] = groups
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(kind="finish", phase="index_relations")
        )
    return relation_groups


def _scan_relation_pairs(
    relation_groups: RelationGroups,
    relations_by_pair: dict[tuple[SampleKey, SampleKey], set[LeakageRelation]],
    *,
    progress_sink: LeakageProgressSink | None,
) -> None:
    candidate_pair_total = _candidate_pair_total(relation_groups)
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(kind="start", phase="relations", total=candidate_pair_total)
        )

    relation_matches = 0
    for spec in LEAKAGE_RELATION_SPECS:
        groups = relation_groups[spec.relation]
        for group in groups.values():
            if len(group) >= 2:
                sorted_group = sorted(group, key=lambda sample: (sample.split, sample.sample_id))
                for left, right in combinations(sorted_group, 2):
                    if left.split != right.split:
                        left_key: SampleKey = (left.split, left.sample_id)
                        right_key: SampleKey = (right.split, right.sample_id)
                        pair_key = (
                            (left_key, right_key)
                            if left_key <= right_key
                            else (right_key, left_key)
                        )
                        relations_by_pair.setdefault(pair_key, set()).add(spec.relation)
                        relation_matches += 1
                    if progress_sink is not None:
                        progress_sink.update_leakage_progress(
                            LeakageProgressEvent(
                                kind="advance",
                                phase="relations",
                                relation=spec.relation,
                                matches=relation_matches,
                            )
                        )
    if progress_sink is not None:
        progress_sink.update_leakage_progress(
            LeakageProgressEvent(kind="finish", phase="relations")
        )


def _candidate_pair_total(relation_groups: RelationGroups) -> int:
    return sum(
        len(group) * (len(group) - 1) // 2
        for relation in LEAKAGE_RELATION_ORDER
        for group in relation_groups[relation].values()
        if len(group) >= 2
    )


def _relation_key_fn(input_field: str) -> Callable[[LeakageInput], str]:
    def key_fn(sample: LeakageInput) -> str:
        return str(getattr(sample, input_field))

    return key_fn


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
