"""Quality facts, leakage facts, and tier assignment for processed-v1 samples."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from text_to_sign_production.core.files import sha256_file
from text_to_sign_production.core.progress import ItemProgress, ProgressReporter

from .constants import SPLITS
from .schemas import ProcessedManifestEntry
from .validate import validate_processed_sample_path

QUALITY_TIER_CONFIG_SCHEMA_VERSION = 1
QUALITY_TIERS = ("broad", "quality", "audit_low_quality", "dropped")
QUALITY_CHANNELS = ("body", "left_hand", "right_hand", "face")
_PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)
_COVERAGE_BINS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


@dataclass(frozen=True, slots=True)
class QualityTierConfig:
    """Config-driven thresholds for assigning processed samples to quality tiers."""

    path: Path
    sha256: str
    raw: Mapping[str, Any]
    minimum_sequence_length: int
    maximum_sequence_length: int | None
    minimum_text_length: int | None
    valid_frame_ratio_min: float
    coverage_min: Mapping[str, float]
    at_least_one_hand_coverage_min: float
    both_hands_preferred_coverage_min: float
    confidence_mean_min: Mapping[str, float]
    severe_out_of_bounds_ratio_max: float
    missingness_ratio_max: float


@dataclass(frozen=True, slots=True)
class SampleQualityFacts:
    """Per-sample facts used by reports, tier manifests, and Base summaries."""

    sample_id: str
    split: str
    quality_tier: str
    quality_score: float
    quality_reasons: tuple[str, ...]
    valid_frame_ratio: float
    coverage_by_channel: Mapping[str, float]
    valid_point_ratio_by_channel: Mapping[str, float]
    confidence_mean_by_channel: Mapping[str, float]
    confidence_distribution_by_channel: Mapping[str, Mapping[str, float | int | None]]
    out_of_bounds_ratio_by_channel: Mapping[str, float]
    out_of_bounds_severity_by_channel: Mapping[str, Mapping[str, float | int | None]]
    missingness_ratio_by_channel: Mapping[str, float]
    zeroed_parser_required_point_ratio: float
    sequence_length_bucket: str
    text_length_bucket: str
    normalized_text: str


def load_quality_tier_config(path: Path | str) -> QualityTierConfig:
    """Load and validate the Dataset quality-tier threshold config."""

    resolved_path = Path(path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Quality tier config root must be a mapping: {resolved_path}")
    schema_version = payload.get("schema_version")
    if schema_version != QUALITY_TIER_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported quality tier config schema_version "
            f"{schema_version!r}; expected {QUALITY_TIER_CONFIG_SCHEMA_VERSION}."
        )
    quality = _required_mapping(payload, "quality")
    coverage_min = _required_channel_float_mapping(quality, "coverage_min")
    confidence_mean_min = _required_channel_float_mapping(quality, "confidence_mean_min")
    return QualityTierConfig(
        path=resolved_path,
        sha256=sha256_file(resolved_path),
        raw=cast(Mapping[str, Any], payload),
        minimum_sequence_length=_required_non_negative_int(
            quality,
            "minimum_sequence_length",
        ),
        maximum_sequence_length=_optional_positive_int(quality, "maximum_sequence_length"),
        minimum_text_length=_optional_non_negative_int(quality, "minimum_text_length"),
        valid_frame_ratio_min=_required_ratio(quality, "valid_frame_ratio_min"),
        coverage_min=coverage_min,
        at_least_one_hand_coverage_min=_required_ratio(
            quality,
            "at_least_one_hand_coverage_min",
        ),
        both_hands_preferred_coverage_min=_required_ratio(
            quality,
            "both_hands_preferred_coverage_min",
        ),
        confidence_mean_min=confidence_mean_min,
        severe_out_of_bounds_ratio_max=_required_ratio(
            quality,
            "severe_out_of_bounds_ratio_max",
        ),
        missingness_ratio_max=_required_ratio(quality, "missingness_ratio_max"),
    )


def build_quality_outputs(
    *,
    records_by_split: Mapping[str, Sequence[ProcessedManifestEntry]],
    dropped_records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    config: QualityTierConfig,
    data_root: Path | str,
    generated_at: str,
    reporter: ProgressReporter | None = None,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    """Compute split reports, leakage reports, and tier manifest records."""

    facts_by_split: dict[str, list[SampleQualityFacts]] = {}
    tiers_by_split: dict[str, dict[str, list[dict[str, Any]]]] = {}
    quality_reports: dict[str, dict[str, Any]] = {}
    for split in splits:
        records = list(records_by_split[split])
        progress = ItemProgress(
            label=f"Quality report {split}",
            total=len(records),
            unit="samples",
            reporter=reporter or _NullReporter(),
        )
        split_facts: list[SampleQualityFacts] = []
        split_tiers: dict[str, list[dict[str, Any]]] = {tier: [] for tier in QUALITY_TIERS}
        for record in records:
            facts = sample_quality_facts(record, config=config, data_root=data_root)
            split_facts.append(facts)
            tier_record = tier_manifest_record(record, facts)
            split_tiers["broad"].append(tier_record)
            split_tiers[facts.quality_tier].append(tier_record)
            progress.advance()
        for dropped_record in dropped_records_by_split.get(split, ()):
            split_tiers["dropped"].append(dropped_tier_record(dropped_record))
        progress.finish(
            quality=len(split_tiers["quality"]), audit=len(split_tiers["audit_low_quality"])
        )
        facts_by_split[split] = split_facts
        tiers_by_split[split] = split_tiers
        quality_reports[split] = split_quality_report(
            split=split,
            records=records,
            facts=split_facts,
            dropped_records=dropped_records_by_split.get(split, ()),
            generated_at=generated_at,
            config=config,
        )
    leakage_report = build_leakage_report(
        records_by_split=records_by_split,
        facts_by_split=facts_by_split,
        generated_at=generated_at,
    )
    return {
        "quality_reports": quality_reports,
        "leakage_report": leakage_report,
        "tiers_by_split": tiers_by_split,
        "tier_counts": {
            split: {tier: len(records) for tier, records in tiers.items()}
            for split, tiers in tiers_by_split.items()
        },
        "facts_by_split": facts_by_split,
        "quality_config": {
            "path": config.path.as_posix(),
            "sha256": config.sha256,
            "schema_version": QUALITY_TIER_CONFIG_SCHEMA_VERSION,
        },
    }


def sample_quality_facts(
    record: ProcessedManifestEntry,
    *,
    config: QualityTierConfig,
    data_root: Path | str,
) -> SampleQualityFacts:
    """Compute model-facing quality facts for one processed sample."""

    sample_path = validate_processed_sample_path(
        record.sample_path,
        split=record.split,
        sample_id=record.sample_id,
        data_root=data_root,
    )
    with np.load(sample_path, allow_pickle=False) as sample:
        coverage_by_channel: dict[str, float] = {}
        valid_point_ratio_by_channel: dict[str, float] = {}
        confidence_mean_by_channel: dict[str, float] = {}
        confidence_distribution_by_channel: dict[str, Mapping[str, float | int | None]] = {}
        out_of_bounds_ratio_by_channel: dict[str, float] = {}
        out_of_bounds_severity_by_channel: dict[str, Mapping[str, float | int | None]] = {}
        missingness_ratio_by_channel: dict[str, float] = {}
        for channel in QUALITY_CHANNELS:
            coords = np.asarray(sample[channel], dtype=np.float32)
            confidence = np.asarray(sample[f"{channel}_confidence"], dtype=np.float32)
            available = np.isfinite(confidence) & (confidence > 0.0)
            frame_available = np.any(available, axis=1) if confidence.size else np.asarray([])
            total_points = int(confidence.size)
            valid_points = int(np.count_nonzero(available))
            coverage_by_channel[channel] = _safe_ratio(
                int(np.count_nonzero(frame_available)), record.num_frames
            )
            valid_point_ratio = _safe_ratio(valid_points, total_points)
            valid_point_ratio_by_channel[channel] = valid_point_ratio
            missingness_ratio_by_channel[channel] = 1.0 - valid_point_ratio
            confidence_values = confidence[available]
            confidence_mean_by_channel[channel] = (
                float(np.mean(confidence_values)) if confidence_values.size else 0.0
            )
            confidence_distribution_by_channel[channel] = percentiles(confidence_values)

            oob_mask = _out_of_bounds_mask(coords) & available
            oob_count = int(np.count_nonzero(oob_mask))
            out_of_bounds_ratio_by_channel[channel] = _safe_ratio(oob_count, valid_points)
            severity = _out_of_bounds_severity(coords)
            out_of_bounds_severity_by_channel[channel] = percentiles(severity[oob_mask])

    valid_frame_ratio = _safe_ratio(record.frame_valid_count, record.num_frames)
    zeroed_ratio = _safe_ratio(
        record.frames_with_any_zeroed_required_joint,
        record.num_frames,
    )
    reasons, score = _quality_reasons_and_score(
        record=record,
        config=config,
        valid_frame_ratio=valid_frame_ratio,
        coverage_by_channel=coverage_by_channel,
        confidence_mean_by_channel=confidence_mean_by_channel,
        out_of_bounds_ratio_by_channel=out_of_bounds_ratio_by_channel,
        missingness_ratio_by_channel=missingness_ratio_by_channel,
    )
    tier = "quality" if not _blocking_reasons(reasons) else "audit_low_quality"
    return SampleQualityFacts(
        sample_id=record.sample_id,
        split=record.split,
        quality_tier=tier,
        quality_score=round(score, 6),
        quality_reasons=tuple(reasons),
        valid_frame_ratio=round(valid_frame_ratio, 6),
        coverage_by_channel=_round_mapping(coverage_by_channel),
        valid_point_ratio_by_channel=_round_mapping(valid_point_ratio_by_channel),
        confidence_mean_by_channel=_round_mapping(confidence_mean_by_channel),
        confidence_distribution_by_channel=confidence_distribution_by_channel,
        out_of_bounds_ratio_by_channel=_round_mapping(out_of_bounds_ratio_by_channel),
        out_of_bounds_severity_by_channel=out_of_bounds_severity_by_channel,
        missingness_ratio_by_channel=_round_mapping(missingness_ratio_by_channel),
        zeroed_parser_required_point_ratio=round(zeroed_ratio, 6),
        sequence_length_bucket=sequence_length_bucket(record.num_frames),
        text_length_bucket=text_length_bucket(len(record.text)),
        normalized_text=normalize_text(record.text),
    )


def tier_manifest_record(
    record: ProcessedManifestEntry,
    facts: SampleQualityFacts,
) -> dict[str, Any]:
    """Return one tier manifest record with original processed fields plus quality facts."""

    return {
        **record.to_record(),
        "quality_tier": facts.quality_tier,
        "quality_score": facts.quality_score,
        "quality_reasons": list(facts.quality_reasons),
        "valid_frame_ratio": facts.valid_frame_ratio,
        "valid_point_ratio": dict(facts.valid_point_ratio_by_channel),
        "coverage_body": facts.coverage_by_channel["body"],
        "coverage_left_hand": facts.coverage_by_channel["left_hand"],
        "coverage_right_hand": facts.coverage_by_channel["right_hand"],
        "coverage_face": facts.coverage_by_channel["face"],
        "confidence_mean_by_channel": dict(facts.confidence_mean_by_channel),
        "out_of_bounds_ratio_by_channel": dict(facts.out_of_bounds_ratio_by_channel),
        "missingness_ratio_by_channel": dict(facts.missingness_ratio_by_channel),
        "sequence_length_bucket": facts.sequence_length_bucket,
        "text_length_bucket": facts.text_length_bucket,
        "normalized_text": facts.normalized_text,
    }


def dropped_tier_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one dropped tier manifest row from filtering facts."""

    return {
        **dict(record),
        "quality_tier": "dropped",
        "quality_score": 0.0,
        "quality_reasons": list(record.get("drop_reasons", ())),
    }


def split_quality_report(
    *,
    split: str,
    records: Sequence[ProcessedManifestEntry],
    facts: Sequence[SampleQualityFacts],
    dropped_records: Sequence[Mapping[str, Any]],
    generated_at: str,
    config: QualityTierConfig,
) -> dict[str, Any]:
    """Build the expanded machine-readable quality report for one split."""

    text_lengths = [len(record.text) for record in records]
    normalized_text_lengths = [len(fact.normalized_text) for fact in facts]
    sequence_lengths = [record.num_frames for record in records]
    tier_counts = Counter(fact.quality_tier for fact in facts)
    channel_payload: dict[str, Any] = {}
    for channel in QUALITY_CHANNELS:
        channel_payload[channel] = {
            "valid_frame_ratio_distribution": percentiles(
                fact.coverage_by_channel[channel] for fact in facts
            ),
            "valid_point_ratio_distribution": percentiles(
                fact.valid_point_ratio_by_channel[channel] for fact in facts
            ),
            "confidence_distribution": percentiles(
                fact.confidence_mean_by_channel[channel] for fact in facts
            ),
            "coverage_histogram": histogram(
                (fact.coverage_by_channel[channel] for fact in facts),
                bins=_COVERAGE_BINS,
            ),
            "out_of_bounds_ratio_distribution": percentiles(
                fact.out_of_bounds_ratio_by_channel[channel] for fact in facts
            ),
            "out_of_bounds_severity_distribution": _merge_percentile_distributions(
                fact.out_of_bounds_severity_by_channel[channel] for fact in facts
            ),
            "missingness_distribution": percentiles(
                fact.missingness_ratio_by_channel[channel] for fact in facts
            ),
        }
    normalized_text_counts = Counter(fact.normalized_text for fact in facts if fact.normalized_text)
    duplicated_normalized_text = {
        text: count for text, count in normalized_text_counts.items() if count > 1
    }
    return {
        "generated_at": generated_at,
        "split": split,
        "sample_counts": {
            "broad": len(records),
            "quality": tier_counts["quality"],
            "audit_low_quality": tier_counts["audit_low_quality"],
            "dropped": len(dropped_records),
        },
        "quality_config": {
            "path": config.path.as_posix(),
            "sha256": config.sha256,
        },
        "sequence_length_percentiles": percentiles(sequence_lengths),
        "text_length_percentiles": percentiles(text_lengths),
        "normalized_text_length_percentiles": percentiles(normalized_text_lengths),
        "token_length": {
            "available_without_model_download": False,
            "policy": "not computed unless a local tokenizer is available without download",
            "whitespace_token_length_percentiles": percentiles(
                len(record.text.split()) for record in records
            ),
        },
        "channels": channel_payload,
        "face_coverage_histogram": channel_payload["face"]["coverage_histogram"],
        "left_hand_coverage_histogram": channel_payload["left_hand"]["coverage_histogram"],
        "right_hand_coverage_histogram": channel_payload["right_hand"]["coverage_histogram"],
        "body_coverage_histogram": channel_payload["body"]["coverage_histogram"],
        "zeroed_parser_required_point_distribution": percentiles(
            fact.zeroed_parser_required_point_ratio for fact in facts
        ),
        "kept_low_quality_examples": [
            {
                "sample_id": fact.sample_id,
                "quality_score": fact.quality_score,
                "quality_reasons": list(fact.quality_reasons),
            }
            for fact in facts
            if fact.quality_tier == "audit_low_quality"
        ][:25],
        "dropped_examples": [
            {
                "sample_id": str(record.get("sample_id", "")),
                "drop_reasons": list(record.get("drop_reasons", ())),
            }
            for record in dropped_records
        ][:25],
        "text_normalization": {
            "empty_normalized_text_count": sum(1 for fact in facts if not fact.normalized_text),
            "short_text_count": sum(1 for record in records if len(record.text.strip()) < 2),
            "duplicated_normalized_text_count": len(duplicated_normalized_text),
            "duplicated_normalized_text_examples": dict(
                sorted(duplicated_normalized_text.items())[:25]
            ),
        },
    }


def build_leakage_report(
    *,
    records_by_split: Mapping[str, Sequence[ProcessedManifestEntry]],
    facts_by_split: Mapping[str, Sequence[SampleQualityFacts]],
    generated_at: str,
) -> dict[str, Any]:
    """Build global cross-split leakage and distribution-comparison facts."""

    overlap_fields = {
        "sample_id": lambda record, _fact: record.sample_id,
        "source_video_id": lambda record, _fact: record.source_video_id,
        "source_sentence_id": lambda record, _fact: record.source_sentence_id,
        "source_sentence_name": lambda record, _fact: record.source_sentence_name,
        "normalized_transcript": lambda _record, fact: fact.normalized_text,
    }
    overlaps: dict[str, Any] = {}
    blocking = False
    for field, extractor in overlap_fields.items():
        field_overlaps = _cross_split_overlaps(
            records_by_split=records_by_split,
            facts_by_split=facts_by_split,
            extractor=extractor,
        )
        overlaps[field] = field_overlaps
        if any(item["overlap_count"] > 0 for item in field_overlaps):
            blocking = True
    near_duplicate_groups = _near_duplicate_normalized_text_groups(facts_by_split)
    if near_duplicate_groups:
        blocking = True
    return {
        "generated_at": generated_at,
        "blocking_for_complete": blocking,
        "overlaps": overlaps,
        "near_duplicate_normalized_transcript_groups": near_duplicate_groups[:100],
        "distribution_comparison": _distribution_comparison(
            records_by_split=records_by_split,
            facts_by_split=facts_by_split,
        ),
    }


def normalize_text(value: str) -> str:
    """Return a conservative normalized transcript for leakage reporting only."""

    lowered = value.casefold()
    alnum_spaced = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(alnum_spaced.split())


def percentiles(values: Iterable[float | int]) -> dict[str, float | int | None]:
    """Return fixed percentile keys for machine-readable reports."""

    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, **{f"p{p}": None for p in _PERCENTILES}}
    result: dict[str, float | int | None] = {"count": int(array.size)}
    computed = np.percentile(array, _PERCENTILES)
    for percentile, value in zip(_PERCENTILES, computed, strict=True):
        key = "min" if percentile == 0 else "max" if percentile == 100 else f"p{percentile}"
        result[key] = round(float(value), 6)
    return result


def histogram(values: Iterable[float | int], *, bins: Sequence[float]) -> dict[str, int]:
    """Return stable closed-right histogram counts for coverage-style ratios."""

    collected = [float(value) for value in values]
    counts = {f"<= {upper:g}": 0 for upper in bins}
    for value in collected:
        for upper in bins:
            if value <= upper:
                counts[f"<= {upper:g}"] += 1
                break
    return counts


def sequence_length_bucket(length: int) -> str:
    if length < 32:
        return "short"
    if length < 128:
        return "medium"
    return "long"


def text_length_bucket(length: int) -> str:
    if length < 32:
        return "short"
    if length < 128:
        return "medium"
    return "long"


def _quality_reasons_and_score(
    *,
    record: ProcessedManifestEntry,
    config: QualityTierConfig,
    valid_frame_ratio: float,
    coverage_by_channel: Mapping[str, float],
    confidence_mean_by_channel: Mapping[str, float],
    out_of_bounds_ratio_by_channel: Mapping[str, float],
    missingness_ratio_by_channel: Mapping[str, float],
) -> tuple[list[str], float]:
    reasons: list[str] = []
    penalties = 0.0
    if record.num_frames < config.minimum_sequence_length:
        reasons.append("blocking:sequence_too_short")
        penalties += 0.25
    if (
        config.maximum_sequence_length is not None
        and record.num_frames > config.maximum_sequence_length
    ):
        reasons.append("blocking:sequence_too_long")
        penalties += 0.15
    if (
        config.minimum_text_length is not None
        and len(record.text.strip()) < config.minimum_text_length
    ):
        reasons.append("blocking:text_too_short")
        penalties += 0.25
    if valid_frame_ratio < config.valid_frame_ratio_min:
        reasons.append("blocking:low_valid_frame_ratio")
        penalties += config.valid_frame_ratio_min - valid_frame_ratio
    for channel in QUALITY_CHANNELS:
        if coverage_by_channel[channel] < config.coverage_min[channel]:
            reasons.append(f"blocking:low_{channel}_coverage")
            penalties += config.coverage_min[channel] - coverage_by_channel[channel]
        if confidence_mean_by_channel[channel] < config.confidence_mean_min[channel]:
            reasons.append(f"blocking:low_{channel}_confidence")
            penalties += config.confidence_mean_min[channel] - confidence_mean_by_channel[channel]
        if out_of_bounds_ratio_by_channel[channel] > config.severe_out_of_bounds_ratio_max:
            reasons.append(f"blocking:high_{channel}_out_of_bounds")
            penalties += (
                out_of_bounds_ratio_by_channel[channel] - config.severe_out_of_bounds_ratio_max
            )
        if missingness_ratio_by_channel[channel] > config.missingness_ratio_max:
            reasons.append(f"blocking:high_{channel}_missingness")
            penalties += missingness_ratio_by_channel[channel] - config.missingness_ratio_max
    at_least_one_hand = max(coverage_by_channel["left_hand"], coverage_by_channel["right_hand"])
    if at_least_one_hand < config.at_least_one_hand_coverage_min:
        reasons.append("blocking:low_at_least_one_hand_coverage")
        penalties += config.at_least_one_hand_coverage_min - at_least_one_hand
    both_hands = min(coverage_by_channel["left_hand"], coverage_by_channel["right_hand"])
    if both_hands < config.both_hands_preferred_coverage_min:
        reasons.append("preferred:low_both_hands_coverage")
        penalties += 0.05
    score = max(0.0, 1.0 - penalties)
    return reasons, score


def _blocking_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    return tuple(reason for reason in reasons if reason.startswith("blocking:"))


def _out_of_bounds_mask(coords: np.ndarray) -> np.ndarray:
    finite = np.isfinite(coords).all(axis=-1)
    x = coords[..., 0]
    y = coords[..., 1]
    return cast(np.ndarray, finite & ((x < 0.0) | (x > 1.0) | (y < 0.0) | (y > 1.0)))


def _out_of_bounds_severity(coords: np.ndarray) -> np.ndarray:
    low = np.maximum(0.0, -coords)
    high = np.maximum(0.0, coords - 1.0)
    return cast(np.ndarray, np.maximum(low, high).max(axis=-1))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _round_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in values.items()}


def _merge_percentile_distributions(
    distributions: Iterable[Mapping[str, float | int | None]],
) -> dict[str, float | int | None]:
    values = [
        float(value)
        for distribution in distributions
        for key, value in distribution.items()
        if key != "count" and value is not None
    ]
    return percentiles(values)


def _cross_split_overlaps(
    *,
    records_by_split: Mapping[str, Sequence[ProcessedManifestEntry]],
    facts_by_split: Mapping[str, Sequence[SampleQualityFacts]],
    extractor: Any,
) -> list[dict[str, Any]]:
    values_by_split: dict[str, set[str]] = {}
    for split, records in records_by_split.items():
        facts_by_sample_id = {fact.sample_id: fact for fact in facts_by_split[split]}
        values_by_split[split] = {
            str(extractor(record, facts_by_sample_id[record.sample_id]))
            for record in records
            if str(extractor(record, facts_by_sample_id[record.sample_id])).strip()
        }
    reports: list[dict[str, Any]] = []
    splits = tuple(split for split in SPLITS if split in values_by_split)
    for left_index, left_split in enumerate(splits):
        for right_split in splits[left_index + 1 :]:
            overlap = sorted(values_by_split[left_split].intersection(values_by_split[right_split]))
            reports.append(
                {
                    "left_split": left_split,
                    "right_split": right_split,
                    "overlap_count": len(overlap),
                    "examples": overlap[:25],
                }
            )
    return reports


def _near_duplicate_normalized_text_groups(
    facts_by_split: Mapping[str, Sequence[SampleQualityFacts]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for split, facts in facts_by_split.items():
        for fact in facts:
            if not fact.normalized_text:
                continue
            groups.setdefault(fact.normalized_text, []).append(
                {"split": split, "sample_id": fact.sample_id}
            )
    return [
        {"normalized_text": text, "records": records}
        for text, records in sorted(groups.items())
        if len({record["split"] for record in records}) > 1
    ]


def _distribution_comparison(
    *,
    records_by_split: Mapping[str, Sequence[ProcessedManifestEntry]],
    facts_by_split: Mapping[str, Sequence[SampleQualityFacts]],
) -> dict[str, Any]:
    return {
        split: {
            "sequence_length": percentiles(record.num_frames for record in records_by_split[split]),
            "text_length": percentiles(len(record.text) for record in records_by_split[split]),
            "coverage": {
                channel: percentiles(
                    fact.coverage_by_channel[channel] for fact in facts_by_split[split]
                )
                for channel in QUALITY_CHANNELS
            },
            "confidence": {
                channel: percentiles(
                    fact.confidence_mean_by_channel[channel] for fact in facts_by_split[split]
                )
                for channel in QUALITY_CHANNELS
            },
        }
        for split in records_by_split
    }


def _required_mapping(payload: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Quality tier config field {field_name!r} must be a mapping.")
    return value


def _required_channel_float_mapping(
    payload: Mapping[str, Any],
    field_name: str,
) -> dict[str, float]:
    mapping = _required_mapping(payload, field_name)
    values: dict[str, float] = {}
    for channel in QUALITY_CHANNELS:
        values[channel] = _required_ratio(mapping, channel)
    return values


def _required_ratio(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Quality tier config field {field_name!r} must be numeric.")
    resolved = float(value)
    if resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"Quality tier config field {field_name!r} must be in [0, 1].")
    return resolved


def _required_non_negative_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Quality tier config field {field_name!r} must be an integer.")
    if value < 0:
        raise ValueError(f"Quality tier config field {field_name!r} must not be negative.")
    return value


def _optional_non_negative_int(payload: Mapping[str, Any], field_name: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Quality tier config field {field_name!r} must be an integer or null.")
    if value < 0:
        raise ValueError(f"Quality tier config field {field_name!r} must not be negative.")
    return value


def _optional_positive_int(payload: Mapping[str, Any], field_name: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Quality tier config field {field_name!r} must be an integer or null.")
    if value <= 0:
        raise ValueError(f"Quality tier config field {field_name!r} must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class _NullReporter:
    def report(self, message: str, **fields: object) -> None:
        return


__all__ = [
    "QUALITY_CHANNELS",
    "QUALITY_TIERS",
    "QualityTierConfig",
    "SampleQualityFacts",
    "build_leakage_report",
    "build_quality_outputs",
    "dropped_tier_record",
    "load_quality_tier_config",
    "normalize_text",
    "sample_quality_facts",
    "split_quality_report",
    "tier_manifest_record",
]
