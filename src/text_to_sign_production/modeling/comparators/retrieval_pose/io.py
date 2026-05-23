"""Generated-pose-compatible IO for retrieval comparator outputs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts.generated_pose import (
    GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
    GeneratedPoseSample,
)
from text_to_sign_production.modeling.artifacts.generated_pose_io import (
    write_generated_pose_manifest_jsonl,
    write_generated_pose_payload,
)
from text_to_sign_production.modeling.artifacts.generated_pose_manifest import (
    GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
    GeneratedPoseManifestEntry,
)
from text_to_sign_production.modeling.artifacts.run_metadata import (
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseProducerType,
)
from text_to_sign_production.modeling.data.bfh_schema import FULL_BFH_CHANNEL_POLICY
from text_to_sign_production.modeling.comparators.retrieval_pose.config import (
    RetrievalPoseComparatorConfig,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.errors import (
    RetrievalPoseComparatorError,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.scoring import (
    RetrievalQueryResult,
)


@dataclass(frozen=True, slots=True)
class RetrievalGeneratedPoseWriteResult:
    split: SampleSplit
    manifest_path: Path
    payload_paths: tuple[Path, ...]
    entries: tuple[GeneratedPoseManifestEntry, ...]
    success_count: int
    failure_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "payload_paths", tuple(Path(path) for path in self.payload_paths))


def write_retrieval_generated_pose_split(
    *,
    topology: ArtifactTopology,
    config: RetrievalPoseComparatorConfig,
    run_name: str,
    split: SampleSplit | str,
    results: Iterable[RetrievalQueryResult],
) -> RetrievalGeneratedPoseWriteResult:
    resolved_split = SampleSplit(split)
    materialized = tuple(results)
    payload_paths: list[Path] = []
    entries: list[GeneratedPoseManifestEntry] = []
    samples_to_write: list[tuple[Path, GeneratedPoseSample]] = []
    for result in materialized:
        if result.query_split is not resolved_split:
            raise RetrievalPoseComparatorError("retrieval result split does not match writer split.")
        if result.success:
            if result.selected_candidate is None or result.selected_bank_item is None:
                raise RetrievalPoseComparatorError("successful retrieval result is missing selected item.")
            sample = _sample_from_result(config=config, run_name=run_name, result=result)
            payload_path = topology.evaluations.generated_pose_sample_file(
                config.identity.comparator_key,
                run_name,
                resolved_split,
                result.query_sample_id,
                0,
            ).path
            try:
                payload_ref = payload_path.relative_to(topology.repo_root).as_posix()
            except ValueError as exc:
                raise RetrievalPoseComparatorError(
                    f"retrieval payload path is not repo-relative: {payload_path}"
                ) from exc
            payload_paths.append(payload_path)
            samples_to_write.append((payload_path, sample))
            entries.append(_success_entry_from_result(config, run_name, result, payload_ref))
        else:
            entries.append(_failure_entry_from_result(config, run_name, result))
    manifest_path = topology.evaluations.generated_pose_manifest(
        config.identity.comparator_key,
        run_name,
        resolved_split,
    ).path
    for payload_path, sample in samples_to_write:
        write_generated_pose_payload(payload_path, sample)
    write_generated_pose_manifest_jsonl(manifest_path, entries, split=resolved_split)
    return RetrievalGeneratedPoseWriteResult(
        split=resolved_split,
        manifest_path=manifest_path,
        payload_paths=tuple(payload_paths),
        entries=tuple(entries),
        success_count=len(payload_paths),
        failure_count=len(entries) - len(payload_paths),
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def write_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def retrieval_result_to_record(result: RetrievalQueryResult) -> dict[str, object]:
    return {
        "query_sample_id": result.query_sample_id,
        "query_split": result.query_split.value,
        "success": result.success,
        "failure_reason": result.failure_reason,
        "blocked_count": result.blocked_count,
        "blocked_reasons": list(result.blocked_reasons),
        "selected_candidate": (
            None if result.selected_candidate is None else _candidate_to_record(result.selected_candidate)
        ),
        "top_k_safe_candidates": [
            _candidate_to_record(candidate) for candidate in result.top_k_safe_candidates
        ],
    }


def _sample_from_result(
    *,
    config: RetrievalPoseComparatorConfig,
    run_name: str,
    result: RetrievalQueryResult,
) -> GeneratedPoseSample:
    item = result.selected_bank_item
    if item is None:
        raise RetrievalPoseComparatorError("successful retrieval result is missing selected item.")
    return GeneratedPoseSample(
        schema_version=GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
        producer_type=GeneratedPoseProducerType.COMPARATOR,
        producer_key=config.identity.comparator_key,
        canonical_id=config.identity.canonical_id,
        phase_number=config.identity.phase_number,
        research_role=config.identity.research_role,
        run_name=run_name,
        split=result.query_split,
        sample_id=result.query_sample_id,
        text=result.query_text,
        source_video_id=result.source_video_id,
        source_sentence_id=result.source_sentence_id,
        source_sentence_name=result.source_sentence_name,
        reference_payload_ref=result.reference_payload_ref,
        generation_index=0,
        num_candidates_for_sample=config.generation.candidates_per_query,
        generation_mode=GeneratedPoseGenerationMode.RETRIEVAL,
        length_policy=GeneratedPoseLengthPolicy.RETRIEVED_LENGTH,
        channel_policy=FULL_BFH_CHANNEL_POLICY,
        confidence_policy=GeneratedPoseConfidencePolicy.RETRIEVED_CONFIDENCE,
        seed=None,
        failure_reason=None,
        pose=item.pose,
    )


def _success_entry_from_result(
    config: RetrievalPoseComparatorConfig,
    run_name: str,
    result: RetrievalQueryResult,
    payload_ref: str,
) -> GeneratedPoseManifestEntry:
    candidate = result.selected_candidate
    if candidate is None:
        raise RetrievalPoseComparatorError("successful retrieval result is missing selected candidate.")
    return GeneratedPoseManifestEntry(
        schema_version=GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
        producer_type=GeneratedPoseProducerType.COMPARATOR,
        producer_key=config.identity.comparator_key,
        canonical_id=config.identity.canonical_id,
        phase_number=config.identity.phase_number,
        research_role=config.identity.research_role,
        run_name=run_name,
        split=result.query_split,
        sample_id=result.query_sample_id,
        source_video_id=result.source_video_id,
        source_sentence_id=result.source_sentence_id,
        source_sentence_name=result.source_sentence_name,
        text=result.query_text,
        reference_payload_ref=result.reference_payload_ref,
        generated_payload_ref=payload_ref,
        generation_index=0,
        num_candidates_for_sample=config.generation.candidates_per_query,
        generation_mode=GeneratedPoseGenerationMode.RETRIEVAL,
        length_policy=config.generation.length_policy,
        channel_policy=FULL_BFH_CHANNEL_POLICY,
        confidence_policy=config.generation.confidence_policy,
        frame_count=candidate.frame_count,
        valid_frame_count=candidate.valid_frame_count,
        seed=None,
        failure_reason=None,
        retrieval_source_split=candidate.retrieval_source_split,
        retrieved_sample_id=candidate.retrieved_sample_id,
        retrieval_score=float(candidate.score),
        leakage_checked=True,
    )


def _failure_entry_from_result(
    config: RetrievalPoseComparatorConfig,
    run_name: str,
    result: RetrievalQueryResult,
) -> GeneratedPoseManifestEntry:
    return GeneratedPoseManifestEntry(
        schema_version=GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
        producer_type=GeneratedPoseProducerType.COMPARATOR,
        producer_key=config.identity.comparator_key,
        canonical_id=config.identity.canonical_id,
        phase_number=config.identity.phase_number,
        research_role=config.identity.research_role,
        run_name=run_name,
        split=result.query_split,
        sample_id=result.query_sample_id,
        source_video_id=result.source_video_id,
        source_sentence_id=result.source_sentence_id,
        source_sentence_name=result.source_sentence_name,
        text=result.query_text,
        reference_payload_ref=result.reference_payload_ref,
        generated_payload_ref=None,
        generation_index=0,
        num_candidates_for_sample=config.generation.candidates_per_query,
        generation_mode=GeneratedPoseGenerationMode.RETRIEVAL,
        length_policy=config.generation.length_policy,
        channel_policy=FULL_BFH_CHANNEL_POLICY,
        confidence_policy=config.generation.confidence_policy,
        frame_count=None,
        valid_frame_count=None,
        seed=None,
        failure_reason=result.failure_reason,
        retrieval_source_split=None,
        retrieved_sample_id=None,
        retrieval_score=None,
        leakage_checked=True,
    )


def _candidate_to_record(candidate: Any) -> dict[str, object]:
    return {
        "rank": candidate.rank,
        "retrieved_sample_id": candidate.retrieved_sample_id,
        "retrieval_source_split": candidate.retrieval_source_split.value,
        "score": candidate.score,
        "blocked_by_leakage": candidate.blocked_by_leakage,
        "leakage_reasons": list(candidate.leakage_reasons),
        "source_sentence_name": candidate.source_sentence_name,
        "frame_count": candidate.frame_count,
        "valid_frame_count": candidate.valid_frame_count,
    }


__all__ = [
    "RetrievalGeneratedPoseWriteResult",
    "retrieval_result_to_record",
    "write_json",
    "write_jsonl",
    "write_retrieval_generated_pose_split",
]
