"""Exact pairwise relation detection."""

from __future__ import annotations

from text_to_sign_production.data.leakages.types import LeakageInput, LeakageRelation
from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.samples.types import PassedManifestEntry


def build_leakage_input(manifest: PassedManifestEntry, metrics: MetricBundle) -> LeakageInput:
    """Build the deterministic leakage input from accepted facts."""
    return LeakageInput(
        sample_id=manifest.sample_id,
        split=manifest.split,
        source_video_id=manifest.source_video_id,
        source_sentence_id=manifest.source_sentence_id,
        source_sentence_name=manifest.source_sentence_name,
        normalized_text=metrics.text.normalized_text,
    )


def detect_pair_relations(
    left: LeakageInput, right: LeakageInput
) -> tuple[LeakageRelation, ...] | None:
    """Detect deterministic relations between two samples."""
    if left.split == right.split:
        return None

    relations: list[LeakageRelation] = []

    if left.source_sentence_id == right.source_sentence_id:
        relations.append(LeakageRelation.SAME_SOURCE_SENTENCE)

    if left.normalized_text == right.normalized_text:
        relations.append(LeakageRelation.EXACT_NORMALIZED_TEXT)

    if left.source_video_id == right.source_video_id:
        relations.append(LeakageRelation.SAME_SOURCE_VIDEO)

    if not relations:
        return None

    return tuple(relations)
