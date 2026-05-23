"""Library runner for the leakage-safe whole-pose retrieval comparator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.text_encoder import build_text_encoder
from text_to_sign_production.modeling.comparators.retrieval_pose.bank import build_retrieval_bank
from text_to_sign_production.modeling.comparators.retrieval_pose.config import (
    RetrievalPoseComparatorConfig,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.errors import (
    RetrievalPoseComparatorError,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.io import (
    retrieval_result_to_record,
    write_json,
    write_jsonl,
    write_retrieval_generated_pose_split,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.leakage import (
    assert_bank_query_split_policy,
    summarize_leakage,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.reports import (
    write_retrieval_reports,
)
from text_to_sign_production.modeling.comparators.retrieval_pose.scoring import (
    RetrievalQuery,
    RetrievalQueryResult,
    score_retrieval_query,
)
from text_to_sign_production.modeling.data import ModelingManifestFamily, load_manifest_samples
from text_to_sign_production.modeling.registry.comparators import require_comparator_spec
from text_to_sign_production.modeling.research import ComparatorKey


@dataclass(frozen=True, slots=True)
class RetrievalComparatorRunResult:
    run_name: str
    comparator_key: str
    manifest_paths: tuple[Path, ...]
    payload_root_paths: tuple[Path, ...]
    retrieval_results_path: Path
    leakage_report_path: Path
    report_paths: tuple[Path, ...]
    records_count: int
    failure_count: int
    safe_retrieval_count: int


def run_retrieval_pose_comparator(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    config: RetrievalPoseComparatorConfig,
    run_name: str,
) -> RetrievalComparatorRunResult:
    spec = require_comparator_spec(ComparatorKey.RETRIEVAL_POSE)
    if spec.primary_model_allowed or spec.canonical_id != config.identity.canonical_id:
        raise RetrievalPoseComparatorError("retrieval comparator registry spec does not match config.")
    if (
        config.data.manifest_family is not None
        and config.data.manifest_family.family_id != manifest_family.family_id
    ):
        raise RetrievalPoseComparatorError(
            "retrieval config manifest_family="
            f"{config.data.manifest_family.family_id!r} does not match runner "
            f"manifest_family={manifest_family.family_id!r}."
        )
    text_encoder = build_text_encoder(config.text_encoder)
    bank = build_retrieval_bank(
        topology=topology,
        manifest_family=manifest_family,
        split=config.data.bank_split,
        text_encoder=text_encoder,
        max_bank_size=config.data.max_bank_size,
    )
    all_results: list[RetrievalQueryResult] = []
    manifest_paths: list[Path] = []
    payload_roots: list[Path] = []
    for query_split in config.data.query_splits:
        assert_bank_query_split_policy(
            bank_split=bank.split,
            query_split=query_split,
            policy=config.leakage,
        )
        split_results = _score_split(
            topology=topology,
            manifest_family=manifest_family,
            query_split=query_split,
            config=config,
            text_encoder=text_encoder,
            bank=bank,
        )
        written = write_retrieval_generated_pose_split(
            topology=topology,
            config=config,
            run_name=run_name,
            split=query_split,
            results=split_results,
        )
        all_results.extend(split_results)
        manifest_paths.append(written.manifest_path)
        payload_roots.append(
            topology.evaluations.generated_pose_samples_root(
                config.identity.comparator_key,
                run_name,
                query_split,
            ).path
        )
    if not all_results or not any(result.success for result in all_results):
        raise RetrievalPoseComparatorError("all retrieval comparator queries failed.")

    run_root = topology.evaluations.generated_pose_run_root(
        config.identity.comparator_key,
        run_name,
    ).path
    retrieval_results_path = run_root / "retrieval_results.jsonl"
    leakage_report_path = run_root / "leakage_report.json"
    config_snapshot_path = run_root / "config_snapshot.json"
    write_jsonl(retrieval_results_path, [retrieval_result_to_record(result) for result in all_results])
    leakage_summary = summarize_leakage(all_results)
    write_json(
        leakage_report_path,
        {
            "blocked_candidates": leakage_summary.blocked_candidates,
            "blocked_by_reason": leakage_summary.blocked_by_reason,
            "queries_without_safe_candidate": leakage_summary.queries_without_safe_candidate,
            "leakage_checked": leakage_summary.leakage_checked,
            "allow_same_split": config.leakage.allow_same_split,
        },
    )
    write_json(config_snapshot_path, dict(config.raw_config))
    report_paths = (
        write_retrieval_reports(
            report_root=topology.reports.modeling_root
            / "comparators"
            / config.reports.output_dir_name
            / run_name,
            config=config,
            results=tuple(all_results),
        )
        if config.reports.enabled
        else ()
    )
    success_count = sum(1 for result in all_results if result.success)
    return RetrievalComparatorRunResult(
        run_name=run_name,
        comparator_key=config.identity.comparator_key,
        manifest_paths=tuple(manifest_paths),
        payload_root_paths=tuple(payload_roots),
        retrieval_results_path=retrieval_results_path,
        leakage_report_path=leakage_report_path,
        report_paths=report_paths,
        records_count=len(all_results),
        failure_count=len(all_results) - success_count,
        safe_retrieval_count=success_count,
    )


def _score_split(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    query_split: SampleSplit,
    config: RetrievalPoseComparatorConfig,
    text_encoder: object,
    bank: object,
) -> tuple[RetrievalQueryResult, ...]:
    samples = load_manifest_samples(
        topology,
        manifest_family,
        query_split,
        limit=config.data.max_query_samples,
    )
    outputs = text_encoder.encode_batch(
        [sample.text for sample in samples],
        sentence_names=[sample.source_sentence_name for sample in samples],
    )
    results: list[RetrievalQueryResult] = []
    for sample, output in zip(samples, outputs, strict=True):
        query = RetrievalQuery(
            sample_id=sample.sample_id,
            split=sample.split,
            text=sample.text,
            source_video_id=sample.source_video_id,
            source_sentence_id=sample.source_sentence_id,
            source_sentence_name=sample.source_sentence_name,
            reference_payload_ref=sample.payload_ref,
            embedding=np.asarray(output.embedding, dtype=np.float32),
        )
        results.append(
            score_retrieval_query(
                query=query,
                bank=bank,
                scoring=config.retrieval,
                leakage_policy=config.leakage,
            )
        )
    return tuple(results)


__all__ = ["RetrievalComparatorRunResult", "run_retrieval_pose_comparator"]
