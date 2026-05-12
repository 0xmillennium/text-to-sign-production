"""Exact leakage input construction and pair-relation detection."""

from __future__ import annotations

from text_to_sign_production.core.models import PassedManifestEntry, PreparedSample
from text_to_sign_production.data.tier.leakages.types import LeakageInput, LeakageRelation


def build_leakage_input(
    sample: PreparedSample,
    manifest: PassedManifestEntry | None = None,
) -> LeakageInput:
    """Build leakage input from checkpoint/sample authority."""
    if manifest is not None:
        if (
            manifest.sample_id != sample.source.sample_id
            or manifest.split is not sample.source.split
        ):
            raise ValueError("Manifest identity must match PreparedSample leakage input.")
        if manifest.text != sample.source.text:
            raise ValueError("Manifest text must match PreparedSample.")
    return LeakageInput(
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        source_sentence_name=sample.source.source_sentence_name,
        text=sample.source.text,
    )


def build_leakage_input_from_manifest(manifest: PassedManifestEntry) -> LeakageInput:
    """Build leakage input from passed-manifest checkpoint authority."""
    return LeakageInput(
        sample_id=manifest.sample_id,
        split=manifest.split,
        source_video_id=manifest.source_video_id,
        source_sentence_id=manifest.source_sentence_id,
        source_sentence_name=manifest.source_sentence_name,
        text=manifest.text,
    )


def detect_pair_relations(
    left: LeakageInput,
    right: LeakageInput,
) -> tuple[LeakageRelation, ...] | None:
    """Detect exact deterministic cross-split leakage relations for a pair."""
    if left.sample_id == right.sample_id and left.split == right.split:
        return None
    if left.split == right.split:
        return None
    relations: list[LeakageRelation] = []
    if left.source_sentence_id == right.source_sentence_id:
        relations.append(LeakageRelation.SAME_SOURCE_SENTENCE)
    if left.text == right.text:
        relations.append(LeakageRelation.EXACT_TEXT)
    if left.source_video_id == right.source_video_id:
        relations.append(LeakageRelation.SAME_SOURCE_VIDEO)
    return tuple(relations) if relations else None


__all__ = [
    "build_leakage_input",
    "build_leakage_input_from_manifest",
    "detect_pair_relations",
]
