"""Post-generation semantic-consistency artifact integration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.modeling.artifacts import (
    GeneratedPoseManifestEntry,
    GeneratedPoseSample,
    load_generated_pose_payload,
    read_generated_pose_manifest_jsonl,
)
from text_to_sign_production.modeling.backbones.bfh_vectorization import vectorize_bfh_pose_arrays
from text_to_sign_production.modeling.candidates.results import ModelStageArtifactRef
from text_to_sign_production.modeling.data.manifest_families import ModelingManifestFamily
from text_to_sign_production.modeling.data.prepared_sample_loader import ModelingManifestSample
from text_to_sign_production.modeling.objectives.semantic_consistency.attachments import (
    build_semantic_ablation_plan,
    build_semantic_attachment_policy,
    decide_semantic_attachment,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.ablation import (
    SEMANTIC_ABLATION_RUN_REFERENCE_SCHEMA_VERSION,
    SEMANTIC_VALIDATION_PROTOCOL,
    SemanticAblationRunReference,
    evaluate_semantic_ablation_readiness,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticConsistencyObjectiveConfig,
    load_semantic_consistency_config,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.embeddings import (
    SemanticTextEmbeddingRecord,
    compute_text_embedding_record,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.io import (
    write_semantic_ablation_plan_json,
    write_semantic_ablation_readiness_json,
    write_semantic_alignment_aggregate_json,
    write_semantic_alignment_results_jsonl,
    write_semantic_attachment_decision_json,
    write_semantic_objective_config_json,
    write_semantic_pose_embeddings_jsonl,
    write_semantic_text_embeddings_jsonl,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.losses import (
    SemanticAlignmentResult,
    compute_semantic_alignment_result,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.pose_features import (
    SEMANTIC_POSE_EMBEDDING_SCHEMA_VERSION,
    SemanticPoseEmbeddingRecord,
    build_pose_feature_spec,
    compute_pose_embedding_record,
    semantic_pose_spec_hash,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.records import (
    SemanticGeneratedCandidatePolicy,
    aggregate_semantic_alignment_results,
    single_candidate_policy,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.reports import (
    semantic_ablation_plan_report,
    semantic_ablation_readiness_report,
    semantic_alignment_summary_report,
    semantic_attachment_decision_report,
    semantic_metric_limitations_report,
    semantic_risk_controls_report,
)
from text_to_sign_production.modeling.research import ModelKey, ModelSpec, ObjectiveSpec


@dataclass(frozen=True, slots=True)
class SemanticObjectiveArtifactBundle:
    """Files and provider-compatible refs written for one validation output surface."""

    config_path: Path
    config_snapshot_path: Path
    attachment_decision_path: Path
    ablation_plan_path: Path
    ablation_readiness_path: Path
    text_embeddings_path: Path
    pose_embeddings_path: Path
    alignment_results_path: Path
    alignment_aggregate_path: Path
    report_paths: tuple[Path, ...]
    artifacts: tuple[ModelStageArtifactRef, ...]
    candidate_policy: SemanticGeneratedCandidatePolicy
    generated_pose_manifest_path: Path
    generated_payload_root: Path
    config_snapshot_sha256: str
    ready_for_comparison: bool
    required_baseline_missing: bool
    readiness_issues: tuple[str, ...]
    records_count: int
    skipped_count: int


def write_semantic_objective_artifacts_for_validation_outputs(
    *,
    model_key: ModelKey,
    run_name: str,
    split: SampleSplit,
    objective_config: SemanticConsistencyObjectiveConfig,
    model_spec: ModelSpec,
    objective_spec: ObjectiveSpec,
    manifest_family: ModelingManifestFamily,
    config_snapshot_path: Path,
    generated_pose_manifest_path: Path,
    generated_payload_root: Path,
    source_manifest_samples: Sequence[ModelingManifestSample],
    output_root: Path,
    run_metadata_path: Path | None = None,
    baseline_run: SemanticAblationRunReference | None = None,
) -> SemanticObjectiveArtifactBundle:
    """Write deterministic proxy evaluation artifacts from final generated poses."""

    model = ModelKey(model_key)
    resolved_split = SampleSplit(split)
    if model_spec.key is not model:
        raise SemanticConsistencyError("semantic integration model_spec does not match model_key.")
    if not isinstance(manifest_family, ModelingManifestFamily):
        raise SemanticConsistencyError("manifest_family must be a ModelingManifestFamily.")
    if not isinstance(objective_config, SemanticConsistencyObjectiveConfig):
        raise SemanticConsistencyError(
            "objective_config must be a SemanticConsistencyObjectiveConfig."
        )
    snapshot_path = Path(config_snapshot_path)
    if not snapshot_path.is_file():
        raise SemanticConsistencyError(
            "semantic objective runtime config snapshot is missing; restore and verify runtime "
            f"before semantic processing: {snapshot_path}"
        )
    if objective_config != load_semantic_consistency_config(snapshot_path):
        raise SemanticConsistencyError(
            "semantic objective config must be loaded from the declared runtime snapshot; "
            "do not process source or mismatched objective configuration."
        )
    candidate_policy = single_candidate_policy()
    policy = build_semantic_attachment_policy(objective_config)
    decision = decide_semantic_attachment(
        model_spec=model_spec,
        objective_spec=objective_spec,
        policy=policy,
    )
    if not decision.attach_allowed:
        raise SemanticConsistencyError(
            "semantic_consistency cannot be attached for this model: "
            + "; ".join(decision.blocking_issues)
        )
    plan = build_semantic_ablation_plan(
        model_key=model,
        baseline_run_name=None if baseline_run is None else baseline_run.run_name,
        objective_run_name=run_name,
        config=objective_config,
    )
    objective_run = SemanticAblationRunReference(
        schema_version=SEMANTIC_ABLATION_RUN_REFERENCE_SCHEMA_VERSION,
        run_name=run_name,
        model_key=model,
        manifest_family=manifest_family.family_id,
        auxiliary_objectives=(objective_config.identity.objective_key,),
        evaluated_splits=(SampleSplit.VAL,),
        validation_protocol=SEMANTIC_VALIDATION_PROTOCOL,
        run_metadata_path=run_metadata_path,
    )
    readiness = evaluate_semantic_ablation_readiness(
        objective_run=objective_run,
        baseline_run=baseline_run,
        config=objective_config,
    )
    sources = _source_samples_by_key(
        source_manifest_samples,
        manifest_family=manifest_family,
        split=resolved_split,
    )
    manifest_path = Path(generated_pose_manifest_path)
    payload_root = Path(generated_payload_root)
    if not manifest_path.resolve(strict=False).is_relative_to(
        payload_root.resolve(strict=False)
    ):
        raise SemanticConsistencyError(
            "semantic objective generated payload root must contain the final validation "
            f"manifest path; root={payload_root}, manifest={manifest_path}"
        )
    try:
        generated_entries = read_generated_pose_manifest_jsonl(
            manifest_path,
            expected_split=resolved_split,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SemanticConsistencyError(
            f"semantic objective requires a readable final validation generated-pose manifest: "
            f"{manifest_path}: {exc}"
        ) from exc
    if not generated_entries:
        raise SemanticConsistencyError(
            "semantic objective final validation generated-pose manifest contains no records."
        )
    text_records: list[SemanticTextEmbeddingRecord] = []
    pose_records: list[SemanticPoseEmbeddingRecord] = []
    alignments: list[SemanticAlignmentResult] = []
    seen_generated: set[tuple[SampleSplit, str]] = set()
    for entry in generated_entries:
        _validate_generated_entry(
            entry,
            model_key=model,
            run_name=run_name,
            split=resolved_split,
            seen=seen_generated,
            candidate_policy=candidate_policy,
        )
        key = (entry.split, entry.sample_id)
        source = sources.get(key)
        if source is None:
            raise SemanticConsistencyError(
                "semantic objective cannot pair generated output with source text: "
                f"missing source sample split={entry.split.value!r}, "
                f"sample_id={entry.sample_id!r}."
            )
        if entry.text != source.text or entry.source_sentence_name != source.source_sentence_name:
            raise SemanticConsistencyError(
                f"semantic objective source/generated identity mismatch for sample "
                f"{entry.sample_id!r}; text and sentence identity must match."
            )
        text_record = compute_text_embedding_record(
            sample_id=source.sample_id,
            source_sentence_name=source.source_sentence_name,
            text=source.text,
            config=objective_config.text_embedding,
        )
        pose_record = _pose_embedding_for_entry(
            entry,
            source=source,
            generated_payload_root=payload_root,
            candidate_policy=candidate_policy,
            objective_config=objective_config,
        )
        text_records.append(text_record)
        pose_records.append(pose_record)
        alignments.append(
            compute_semantic_alignment_result(
                sample_id=entry.sample_id,
                split=resolved_split,
                model_key=model,
                text_embedding=text_record,
                pose_embedding=pose_record,
                config=objective_config,
            )
        )
    aggregate = aggregate_semantic_alignment_results(
        alignments,
        model_key=model,
        split=resolved_split,
    )
    root = Path(output_root)
    paths = {
        "config": root / "semantic_objective_config.json",
        "decision": root / "attachment_decision.json",
        "ablation": root / "ablation_plan.json",
        "readiness": root / "semantic_ablation_readiness.json",
        "text": root / "text_embeddings.jsonl",
        "pose": root / "pose_embeddings.jsonl",
        "results": root / "alignment_results.jsonl",
        "aggregate": root / "alignment_aggregate.json",
    }
    config_snapshot_sha256 = sha256_file(snapshot_path)
    write_semantic_objective_config_json(
        paths["config"],
        objective_config,
        config_snapshot_path=snapshot_path,
        config_snapshot_sha256=config_snapshot_sha256,
    )
    write_semantic_attachment_decision_json(paths["decision"], decision)
    write_semantic_ablation_plan_json(paths["ablation"], plan)
    write_semantic_ablation_readiness_json(paths["readiness"], readiness)
    write_semantic_text_embeddings_jsonl(paths["text"], text_records)
    write_semantic_pose_embeddings_jsonl(paths["pose"], pose_records)
    write_semantic_alignment_results_jsonl(paths["results"], alignments)
    write_semantic_alignment_aggregate_json(paths["aggregate"], aggregate)
    report_payloads = (
        (
            root / "attachment_decision.md",
            "Semantic Attachment Decision",
            semantic_attachment_decision_report(decision),
        ),
        (
            root / "ablation_plan.md",
            "Semantic Ablation Plan",
            semantic_ablation_plan_report(plan),
        ),
        (
            root / "ablation_readiness.md",
            "Semantic Ablation Readiness",
            semantic_ablation_readiness_report(readiness),
        ),
        (
            root / "alignment_summary.md",
            "Semantic Alignment Summary",
            semantic_alignment_summary_report(
                aggregate,
                generated_payload_root=payload_root,
                payload_ref_policy=(
                    "split_root_relative_with_current_generated_pose_tail_compatibility"
                ),
            ),
        ),
        (
            root / "metric_limitations.md",
            "Semantic Metric Limitations",
            semantic_metric_limitations_report(),
        ),
        (
            root / "risk_controls.md",
            "Semantic Risk Controls",
            semantic_risk_controls_report(
                generated_payload_root=payload_root,
                payload_ref_policy=(
                    "split_root_relative_with_current_generated_pose_tail_compatibility"
                ),
            ),
        ),
    )
    for path, title, payload in report_payloads:
        _write_report_markdown(path, title=title, payload=payload)
    report_paths = tuple(path for path, _, _ in report_payloads)
    artifact_data = (
        ("semantic_objective_config", paths["config"], "model_semantic_objective_config"),
        ("semantic_attachment_decision", paths["decision"], "model_semantic_attachment_decision"),
        ("semantic_ablation_plan", paths["ablation"], "model_semantic_ablation_plan"),
        ("semantic_ablation_readiness", paths["readiness"], "model_semantic_ablation_readiness"),
        ("semantic_text_embeddings", paths["text"], "model_semantic_text_embeddings"),
        ("semantic_pose_embeddings", paths["pose"], "model_semantic_pose_embeddings"),
        ("semantic_alignment_results", paths["results"], "model_semantic_alignment_results"),
        ("semantic_alignment_aggregate", paths["aggregate"], "model_semantic_alignment_aggregate"),
        *((path.stem, path, "model_semantic_report") for path in report_paths),
    )
    artifacts = tuple(
        ModelStageArtifactRef(
            role=role,
            path=path,
            kind=kind,
            metadata={
                "objective_key": "semantic_consistency",
                "model_key": model.value,
                "split": resolved_split.value,
                "proxy_only": True,
                "candidate_policy": candidate_policy.policy,
            },
        )
        for role, path, kind in artifact_data
    )
    return SemanticObjectiveArtifactBundle(
        config_path=paths["config"],
        config_snapshot_path=snapshot_path,
        attachment_decision_path=paths["decision"],
        ablation_plan_path=paths["ablation"],
        ablation_readiness_path=paths["readiness"],
        text_embeddings_path=paths["text"],
        pose_embeddings_path=paths["pose"],
        alignment_results_path=paths["results"],
        alignment_aggregate_path=paths["aggregate"],
        report_paths=report_paths,
        artifacts=artifacts,
        candidate_policy=candidate_policy,
        generated_pose_manifest_path=manifest_path,
        generated_payload_root=payload_root,
        config_snapshot_sha256=config_snapshot_sha256,
        ready_for_comparison=readiness.ready_for_comparison,
        required_baseline_missing=readiness.required_baseline_missing,
        readiness_issues=readiness.issues,
        records_count=aggregate.records_count,
        skipped_count=aggregate.skipped_count,
    )


def _source_samples_by_key(
    samples: Sequence[ModelingManifestSample],
    *,
    manifest_family: ModelingManifestFamily,
    split: SampleSplit,
) -> dict[tuple[SampleSplit, str], ModelingManifestSample]:
    sources: dict[tuple[SampleSplit, str], ModelingManifestSample] = {}
    for sample in samples:
        if not isinstance(sample, ModelingManifestSample):
            raise SemanticConsistencyError(
                "source_manifest_samples must contain ModelingManifestSample values."
            )
        if sample.manifest_family != manifest_family:
            raise SemanticConsistencyError(
                "semantic objective source sample manifest family does not match request."
            )
        if sample.split is not split:
            raise SemanticConsistencyError(
                "semantic objective source_manifest_samples cannot contain cross-split rows."
            )
        key = (sample.split, sample.sample_id)
        if key in sources:
            raise SemanticConsistencyError(
                f"semantic objective source_manifest_samples contains duplicate sample "
                f"{sample.sample_id!r} in split {sample.split.value!r}."
            )
        sources[key] = sample
    return sources


def _validate_generated_entry(
    entry: GeneratedPoseManifestEntry,
    *,
    model_key: ModelKey,
    run_name: str,
    split: SampleSplit,
    seen: set[tuple[SampleSplit, str]],
    candidate_policy: SemanticGeneratedCandidatePolicy,
) -> None:
    if entry.producer_key != model_key.value or entry.run_name != run_name:
        raise SemanticConsistencyError(
            "semantic objective generated manifest producer/run does not match model run."
        )
    if entry.split is not split:
        raise SemanticConsistencyError("semantic objective cannot pair cross-split generated rows.")
    if entry.generation_index != candidate_policy.required_generation_index:
        raise SemanticConsistencyError(
            "semantic objective candidate policy is single_candidate_only: "
            "generation_index must be 0. Multi-candidate semantic aggregation requires "
            "a future explicit policy."
        )
    if entry.num_candidates_for_sample > candidate_policy.max_candidates_per_sample:
        raise SemanticConsistencyError(
            "semantic objective candidate policy is single_candidate_only: "
            "num_candidates_for_sample must be 1. Multi-candidate semantic aggregation "
            "requires a future explicit policy."
        )
    key = (entry.split, entry.sample_id)
    if key in seen:
        raise SemanticConsistencyError(
            "semantic objective requires one generated validation candidate per sample; "
            f"duplicate generated sample {entry.sample_id!r} was found."
        )
    seen.add(key)


def _pose_embedding_for_entry(
    entry: GeneratedPoseManifestEntry,
    *,
    source: ModelingManifestSample,
    generated_payload_root: Path,
    candidate_policy: SemanticGeneratedCandidatePolicy,
    objective_config: SemanticConsistencyObjectiveConfig,
) -> SemanticPoseEmbeddingRecord:
    spec = build_pose_feature_spec(objective_config.pose_embedding)
    if entry.failure_reason is not None:
        return SemanticPoseEmbeddingRecord(
            schema_version=SEMANTIC_POSE_EMBEDDING_SCHEMA_VERSION,
            sample_id=entry.sample_id,
            split=entry.split,
            generation_index=entry.generation_index,
            candidate_policy=candidate_policy.policy,
            source_sentence_name=source.source_sentence_name,
            backend=spec.backend,
            embedding_dim=spec.embedding_dim,
            coordinate_mode=spec.coordinate_mode,
            channel_groups=spec.channel_groups,
            include_velocity_statistics=spec.include_velocity_statistics,
            include_validity_statistics=spec.include_validity_statistics,
            projection_seed=spec.projection_seed,
            embedding=None,
            valid_observation_count=0,
            skipped=True,
            reason=f"generated_failed: {entry.failure_reason}",
            proxy_only=spec.proxy_only,
            spec_hash=semantic_pose_spec_hash(spec),
        )
    generated = _load_generated_payload(entry, generated_payload_root=generated_payload_root)
    return compute_pose_embedding_record(
        sample_id=entry.sample_id,
        split=entry.split,
        generation_index=entry.generation_index,
        candidate_policy=candidate_policy.policy,
        source_sentence_name=source.source_sentence_name,
        vectorized=vectorize_bfh_pose_arrays(generated.pose, sample_id=entry.sample_id),
        config=objective_config.pose_embedding,
    )


def _load_generated_payload(
    entry: GeneratedPoseManifestEntry,
    *,
    generated_payload_root: Path,
) -> GeneratedPoseSample:
    if entry.generated_payload_ref is None:
        raise SemanticConsistencyError(
            f"successful generated sample {entry.sample_id!r} is missing generated_payload_ref."
        )
    path = resolve_generated_pose_payload_path(
        payload_ref=entry.generated_payload_ref,
        generated_payload_root=generated_payload_root,
    )
    try:
        sample = load_generated_pose_payload(path)
    except (OSError, ValueError) as exc:
        raise SemanticConsistencyError(
            f"semantic objective generated-pose payload is invalid: {path}: {exc}"
        ) from exc
    if (
        sample.sample_id != entry.sample_id
        or sample.split is not entry.split
        or sample.run_name != entry.run_name
        or sample.producer_key != entry.producer_key
        or sample.generation_index != entry.generation_index
    ):
        raise SemanticConsistencyError(
            f"semantic objective generated payload identity does not match manifest for "
            f"sample {entry.sample_id!r}."
        )
    return sample


def resolve_generated_pose_payload_path(
    *,
    payload_ref: str,
    generated_payload_root: Path,
) -> Path:
    """Resolve one manifest payload only within its declared generated-output root."""

    if not isinstance(payload_ref, str) or not payload_ref.strip():
        raise SemanticConsistencyError("generated_payload_ref must be a non-empty relative path.")
    relative = Path(payload_ref)
    if relative.is_absolute():
        raise SemanticConsistencyError("generated_payload_ref must be relative, not absolute.")
    if ".." in relative.parts:
        raise SemanticConsistencyError("generated_payload_ref must not contain parent traversal.")
    root = Path(generated_payload_root).resolve(strict=False)
    if not root.is_dir():
        raise SemanticConsistencyError(
            f"generated payload root must be a materialized runtime directory: {root}"
        )
    try:
        path = _resolve_payload_ref_under_root(root, relative)
    except FileNotFoundError as exc:
        raise SemanticConsistencyError(
            "generated-pose payload referenced by final validation manifest is missing under "
            f"the declared root {root}: {payload_ref!r}."
        ) from exc
    if not path.is_relative_to(root):
        raise SemanticConsistencyError(
            "generated_payload_ref resolves outside the declared runtime generated payload root; "
            "remove symlink or traversal escapes."
        )
    if not path.is_file():
        raise SemanticConsistencyError(f"generated-pose payload path is not a file: {path}")
    return path


def _resolve_payload_ref_under_root(root: Path, relative: Path) -> Path:
    direct = root / relative
    if direct.exists():
        return direct.resolve(strict=True)
    root_tail = _generated_root_tail(root)
    parts = relative.parts
    for index in range(0, len(parts) - len(root_tail) + 1):
        if parts[index:index + len(root_tail)] == root_tail:
            narrowed_relative = Path(*parts[index + len(root_tail):])
            if not narrowed_relative.parts:
                break
            narrowed = root / narrowed_relative
            if narrowed.exists():
                return narrowed.resolve(strict=True)
    return direct.resolve(strict=True)


def _generated_root_tail(root: Path) -> tuple[str, ...]:
    parts = root.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "generated_pose":
            return tuple(parts[index:])
    return tuple(parts[-3:])


def _write_report_markdown(
    path: Path,
    *,
    title: str,
    payload: dict[str, object],
) -> None:
    caveats = payload.get("caveats")
    if not isinstance(caveats, list):
        raise SemanticConsistencyError("semantic report payload must include caveats.")
    lines = [
        f"# {title}",
        "",
        "- Objective: `semantic_consistency`",
        "- Processing mode: `post-generation proxy evaluation`",
        "",
        "## Facts",
        "",
        *(
            f"- {key}: `{_report_value(value)}`"
            for key, value in payload.items()
            if key != "caveats"
        ),
        "",
        "## Limitations And Controls",
        "",
        *(f"- {str(caveat)}" for caveat in caveats),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _report_value(value: object) -> str:
    if isinstance(value, list):
        return "none" if not value else "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={item}" for key, item in value.items())
    return str(value)


__all__ = [
    "SemanticObjectiveArtifactBundle",
    "resolve_generated_pose_payload_path",
    "write_semantic_objective_artifacts_for_validation_outputs",
]
