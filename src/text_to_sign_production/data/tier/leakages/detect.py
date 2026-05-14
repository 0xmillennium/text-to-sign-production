"""Deterministic leakage bundle composition."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.models import PassedManifestEntry, PreparedSample
from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.data.tier.leakages.overlap import (
    build_leakage_input,
    detect_pair_relations,
)
from text_to_sign_production.data.tier.leakages.severity import (
    classify_leakage_severity,
    max_leakage_severity,
)
from text_to_sign_production.data.tier.leakages.types import (
    LEAKAGE_RELATION_ORDER,
    LeakageBundle,
    LeakageInput,
    LeakagePairFact,
    LeakageProgressSpecs,
    LeakageRelation,
    LeakageSampleRef,
    LeakageSampleSummary,
)

SampleKey = tuple[SampleSplit, str]


def build_leakage_bundle(
    samples: Sequence[PreparedSample],
    manifests: Sequence[PassedManifestEntry] = (),
    *,
    progress_session: ProgressSession | None = None,
    progress_specs: LeakageProgressSpecs | None = None,
) -> LeakageBundle:
    """Detect and compose leakage facts from PreparedSample checkpoint authority."""
    manifest_by_id = {manifest.sample_id: manifest for manifest in manifests}
    inputs = tuple(
        build_leakage_input(sample, manifest_by_id.get(sample.source.sample_id))
        for sample in samples
    )
    return build_leakage_bundle_from_inputs(
        inputs,
        progress_session=progress_session,
        progress_specs=progress_specs,
    )


def build_leakage_bundle_from_inputs(
    inputs: Sequence[LeakageInput],
    *,
    progress_session: ProgressSession | None = None,
    progress_specs: LeakageProgressSpecs | None = None,
) -> LeakageBundle:
    """Detect and compose leakage facts from explicit leakage inputs."""
    progress_specs = progress_specs or LeakageProgressSpecs()
    sorted_inputs = tuple(sorted(inputs, key=_input_sort_key))
    _check_duplicate_inputs(
        sorted_inputs,
        progress_session=progress_session,
        progress_specs=progress_specs,
    )
    pair_facts = _build_pair_facts(
        sorted_inputs,
        progress_session=progress_session,
        progress_specs=progress_specs,
    )
    sample_summaries = _build_sample_summaries(
        sorted_inputs,
        pair_facts,
        progress_session=progress_session,
        progress_specs=progress_specs,
    )
    return LeakageBundle(pair_facts=pair_facts, sample_summaries=sample_summaries)


def _check_duplicate_inputs(
    inputs: Sequence[LeakageInput],
    *,
    progress_session: ProgressSession | None,
    progress_specs: LeakageProgressSpecs,
) -> None:
    if progress_session is not None and progress_specs.duplicate_check is not None:
        with progress_session.task(progress_specs.duplicate_check, total=len(inputs)) as task:
            _check_duplicate_inputs_inner(inputs, progress_task=task)
        return
    _check_duplicate_inputs_inner(inputs, progress_task=None)


def _check_duplicate_inputs_inner(inputs: Sequence[LeakageInput], *, progress_task) -> None:
    seen: set[SampleKey] = set()
    for sample in inputs:
        key = _sample_key(sample)
        if key in seen:
            raise ValueError(
                "Duplicate leakage input key: "
                f"split={sample.split.value!r}, sample_id={sample.sample_id!r}."
            )
        seen.add(key)
        _advance(progress_task)


def _build_pair_facts(
    inputs: Sequence[LeakageInput],
    *,
    progress_session: ProgressSession | None,
    progress_specs: LeakageProgressSpecs,
) -> tuple[LeakagePairFact, ...]:
    total = len(inputs) * (len(inputs) - 1) // 2
    if progress_session is not None and progress_specs.relation_scan is not None:
        with progress_session.task(progress_specs.relation_scan, total=total) as task:
            return _build_pair_facts_inner(inputs, progress_task=task)
    return _build_pair_facts_inner(inputs, progress_task=None)


def _build_pair_facts_inner(
    inputs: Sequence[LeakageInput],
    *,
    progress_task,
) -> tuple[LeakagePairFact, ...]:
    facts: list[LeakagePairFact] = []
    pending_progress = 0
    for left, right in combinations(inputs, 2):
        relations = detect_pair_relations(left, right)
        pending_progress += 1
        if pending_progress >= 1000:
            _advance(progress_task, pending_progress)
            pending_progress = 0
        if relations is None:
            continue
        facts.append(
            LeakagePairFact(
                left_sample_id=left.sample_id,
                right_sample_id=right.sample_id,
                left_split=left.split,
                right_split=right.split,
                relations=tuple(
                    relation for relation in LEAKAGE_RELATION_ORDER if relation in relations
                ),
                severity=classify_leakage_severity(relations),
            )
        )
    if pending_progress:
        _advance(progress_task, pending_progress)
    return tuple(facts)


def _build_sample_summaries(
    inputs: Sequence[LeakageInput],
    pair_facts: tuple[LeakagePairFact, ...],
    *,
    progress_session: ProgressSession | None,
    progress_specs: LeakageProgressSpecs,
) -> tuple[LeakageSampleSummary, ...]:
    if progress_session is not None and progress_specs.summary_build is not None:
        with progress_session.task(progress_specs.summary_build, total=len(inputs)) as task:
            return _build_sample_summaries_inner(inputs, pair_facts, progress_task=task)
    return _build_sample_summaries_inner(inputs, pair_facts, progress_task=None)


def _build_sample_summaries_inner(
    inputs: Sequence[LeakageInput],
    pair_facts: tuple[LeakagePairFact, ...],
    *,
    progress_task,
) -> tuple[LeakageSampleSummary, ...]:
    pairs_by_sample: dict[SampleKey, list[tuple[LeakagePairFact, SampleKey]]] = {
        _sample_key(sample): [] for sample in inputs
    }
    for fact in pair_facts:
        left = (fact.left_split, fact.left_sample_id)
        right = (fact.right_split, fact.right_sample_id)
        pairs_by_sample[left].append((fact, right))
        pairs_by_sample[right].append((fact, left))
    summaries: list[LeakageSampleSummary] = []
    for sample in inputs:
        key = _sample_key(sample)
        pairs = pairs_by_sample[key]
        severities = tuple(fact.severity for fact, _ in pairs)
        matched = tuple(
            LeakageSampleRef(split=split, sample_id=sample_id)
            for split, sample_id in sorted(
                {other for _, other in pairs}, key=lambda item: (item[0].value, item[1])
            )
        )
        summaries.append(
            LeakageSampleSummary(
                sample_id=sample.sample_id,
                split=sample.split,
                has_leakage=bool(pairs),
                max_severity=max_leakage_severity(severities),
                same_source_sentence_match_count=_relation_count(
                    pairs,
                    LeakageRelation.SAME_SOURCE_SENTENCE,
                ),
                exact_text_match_count=_relation_count(
                    pairs,
                    LeakageRelation.EXACT_TEXT,
                ),
                same_source_video_match_count=_relation_count(
                    pairs,
                    LeakageRelation.SAME_SOURCE_VIDEO,
                ),
                matched_samples=matched,
            )
        )
        _advance(progress_task)
    return tuple(summaries)


def _relation_count(
    pairs: list[tuple[LeakagePairFact, SampleKey]],
    relation: LeakageRelation,
) -> int:
    return sum(1 for fact, _ in pairs if relation in fact.relations)


def _sample_key(sample: LeakageInput) -> SampleKey:
    return (sample.split, sample.sample_id)


def _input_sort_key(sample: LeakageInput) -> tuple[str, str]:
    return (sample.split.value, sample.sample_id)


def _advance(progress_task, count: int = 1) -> None:
    if progress_task is not None:
        progress_task.advance(count)


__all__ = ["build_leakage_bundle", "build_leakage_bundle_from_inputs"]
