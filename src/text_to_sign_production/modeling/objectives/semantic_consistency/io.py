"""Strict deterministic IO for semantic-consistency foundation artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.objectives.semantic_consistency.attachments import (
    SEMANTIC_ABLATION_PLAN_SCHEMA_VERSION,
    SEMANTIC_ATTACHMENT_DECISION_SCHEMA_VERSION,
    SemanticAblationPlan,
    SemanticObjectiveAttachmentDecision,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.ablation import (
    SEMANTIC_ABLATION_READINESS_SCHEMA_VERSION,
    SemanticAblationReadinessResult,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SEMANTIC_OBJECTIVE_CONFIG_SCHEMA_VERSION,
    SemanticConsistencyObjectiveConfig,
    semantic_consistency_config_from_mapping,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.embeddings import (
    SEMANTIC_TEXT_EMBEDDING_SCHEMA_VERSION,
    SemanticTextEmbeddingRecord,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.losses import (
    SEMANTIC_ALIGNMENT_RESULT_SCHEMA_VERSION,
    SemanticAlignmentResult,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.pose_features import (
    SEMANTIC_POSE_EMBEDDING_SCHEMA_VERSION,
    SemanticPoseEmbeddingRecord,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.records import (
    SEMANTIC_ALIGNMENT_AGGREGATE_SCHEMA_VERSION,
    SemanticAlignmentAggregate,
)
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey

SEMANTIC_OBJECTIVE_CONFIG_ARTIFACT_SCHEMA_VERSION = "t2sp-semantic-objective-config-artifact-v1"

_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "identity",
        "attachment",
        "text_embedding",
        "pose_embedding",
        "alignment",
        "ablation",
        "reports",
    }
)
_CONFIG_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "config",
        "snapshot_identity",
    }
)
_CONFIG_SNAPSHOT_IDENTITY_KEYS = frozenset(
    {
        "config_source",
        "config_snapshot_path",
        "config_snapshot_sha256",
    }
)
_DECISION_KEYS = frozenset(
    {
        "schema_version",
        "model_key",
        "objective_key",
        "attach_allowed",
        "requires_ablation",
        "reasons",
        "blocking_issues",
    }
)
_ABLATION_KEYS = frozenset(
    {
        "schema_version",
        "model_key",
        "objective_key",
        "baseline_run_name",
        "objective_run_name",
        "same_manifest_family_required",
        "same_splits_required",
        "same_validation_protocol_required",
        "same_seed_required",
        "status",
        "issues",
    }
)
_TEXT_EMBEDDING_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "source_sentence_name",
        "text",
        "backend",
        "embedding_dim",
        "pooling",
        "trainable",
        "embedding",
        "proxy_only",
        "config_hash",
    }
)
_POSE_EMBEDDING_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "split",
        "generation_index",
        "candidate_policy",
        "source_sentence_name",
        "backend",
        "embedding_dim",
        "coordinate_mode",
        "channel_groups",
        "include_velocity_statistics",
        "include_validity_statistics",
        "projection_seed",
        "embedding",
        "valid_observation_count",
        "skipped",
        "reason",
        "proxy_only",
        "spec_hash",
    }
)
_ALIGNMENT_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "split",
        "generation_index",
        "candidate_policy",
        "model_key",
        "objective_key",
        "metric",
        "cosine_similarity",
        "cosine_distance",
        "loss_weight",
        "loss_value",
        "skipped",
        "reason",
        "proxy_only",
    }
)
_AGGREGATE_KEYS = frozenset(
    {
        "schema_version",
        "objective_key",
        "model_key",
        "split",
        "candidate_policy",
        "records_count",
        "skipped_count",
        "mean_cosine_similarity",
        "mean_cosine_distance",
        "mean_loss_value",
        "caveats",
    }
)
_READINESS_KEYS = frozenset(
    {
        "schema_version",
        "objective_key",
        "model_key",
        "objective_run_name",
        "baseline_run_name",
        "ready_for_comparison",
        "required_baseline_missing",
        "same_model_required",
        "same_manifest_family_required",
        "same_splits_required",
        "same_validation_protocol_required",
        "issues",
        "caveats",
    }
)


def write_semantic_objective_config_json(
    path: Path,
    config: SemanticConsistencyObjectiveConfig,
    *,
    config_snapshot_path: Path,
    config_snapshot_sha256: str,
) -> None:
    if not isinstance(config, SemanticConsistencyObjectiveConfig):
        raise SemanticConsistencyError("config must be a SemanticConsistencyObjectiveConfig.")
    snapshot_path = Path(config_snapshot_path)
    snapshot_hash = _text(config_snapshot_sha256, "config_snapshot_sha256")
    _write_json(
        path,
        {
            "schema_version": SEMANTIC_OBJECTIVE_CONFIG_ARTIFACT_SCHEMA_VERSION,
            "config": {
                "schema_version": SEMANTIC_OBJECTIVE_CONFIG_SCHEMA_VERSION,
                **config.to_dict(),
            },
            "snapshot_identity": {
                "config_source": "runtime_snapshot",
                "config_snapshot_path": str(snapshot_path),
                "config_snapshot_sha256": snapshot_hash,
            },
        },
    )


def read_semantic_objective_config_json(path: Path) -> SemanticConsistencyObjectiveConfig:
    record = _read_json(path, "semantic objective config")
    if set(record) == _CONFIG_ARTIFACT_KEYS:
        _schema(
            record["schema_version"],
            SEMANTIC_OBJECTIVE_CONFIG_ARTIFACT_SCHEMA_VERSION,
            "semantic objective config artifact",
        )
        config_record = _mapping(record["config"], "semantic objective config artifact config")
        snapshot_identity = _mapping(
            record["snapshot_identity"],
            "semantic objective config snapshot_identity",
        )
        _keys(
            snapshot_identity,
            _CONFIG_SNAPSHOT_IDENTITY_KEYS,
            "semantic objective config snapshot_identity",
        )
        if snapshot_identity["config_source"] != "runtime_snapshot":
            raise SemanticConsistencyError(
                "semantic objective config snapshot_identity.config_source must be "
                "'runtime_snapshot'."
            )
        _text(snapshot_identity["config_snapshot_path"], "config_snapshot_path")
        _text(snapshot_identity["config_snapshot_sha256"], "config_snapshot_sha256")
        record = config_record
    expected_config_keys = (
        _CONFIG_KEYS | {"training_objective"}
        if "training_objective" in record
        else _CONFIG_KEYS
    )
    _keys(record, expected_config_keys, "semantic objective config")
    _schema(
        record["schema_version"],
        SEMANTIC_OBJECTIVE_CONFIG_SCHEMA_VERSION,
        "semantic objective config",
    )
    return semantic_consistency_config_from_mapping(
        {key: record[key] for key in expected_config_keys if key != "schema_version"}
    )


def write_semantic_attachment_decision_json(
    path: Path,
    decision: SemanticObjectiveAttachmentDecision,
) -> None:
    _instance(decision, SemanticObjectiveAttachmentDecision, "decision")
    _write_json(path, decision.to_dict())


def read_semantic_attachment_decision_json(path: Path) -> SemanticObjectiveAttachmentDecision:
    record = _read_json(path, "semantic attachment decision")
    _keys(record, _DECISION_KEYS, "semantic attachment decision")
    _schema(
        record["schema_version"],
        SEMANTIC_ATTACHMENT_DECISION_SCHEMA_VERSION,
        "semantic attachment decision",
    )
    return SemanticObjectiveAttachmentDecision(
        schema_version=_text(record["schema_version"], "schema_version"),
        model_key=_model(record["model_key"]),
        objective_key=_objective(record["objective_key"]),
        attach_allowed=_bool(record["attach_allowed"], "attach_allowed"),
        requires_ablation=_bool(record["requires_ablation"], "requires_ablation"),
        reasons=_text_tuple(record["reasons"], "reasons"),
        blocking_issues=_text_tuple(record["blocking_issues"], "blocking_issues"),
    )


def write_semantic_ablation_plan_json(path: Path, plan: SemanticAblationPlan) -> None:
    _instance(plan, SemanticAblationPlan, "plan")
    _write_json(path, plan.to_dict())


def read_semantic_ablation_plan_json(path: Path) -> SemanticAblationPlan:
    record = _read_json(path, "semantic ablation plan")
    _keys(record, _ABLATION_KEYS, "semantic ablation plan")
    _schema(record["schema_version"], SEMANTIC_ABLATION_PLAN_SCHEMA_VERSION, "semantic ablation plan")
    return SemanticAblationPlan(
        schema_version=_text(record["schema_version"], "schema_version"),
        model_key=_model(record["model_key"]),
        objective_key=_objective(record["objective_key"]),
        baseline_run_name=_optional_text(record["baseline_run_name"], "baseline_run_name"),
        objective_run_name=_optional_text(record["objective_run_name"], "objective_run_name"),
        same_manifest_family_required=_bool(
            record["same_manifest_family_required"], "same_manifest_family_required"
        ),
        same_splits_required=_bool(record["same_splits_required"], "same_splits_required"),
        same_validation_protocol_required=_bool(
            record["same_validation_protocol_required"], "same_validation_protocol_required"
        ),
        same_seed_required=_bool(record["same_seed_required"], "same_seed_required"),
        status=_text(record["status"], "status"),
        issues=_text_tuple(record["issues"], "issues"),
    )


def write_semantic_text_embeddings_jsonl(
    path: Path,
    records: Iterable[SemanticTextEmbeddingRecord],
) -> None:
    _write_jsonl(path, (_typed_dict(record, SemanticTextEmbeddingRecord) for record in records))


def read_semantic_text_embeddings_jsonl(path: Path) -> tuple[SemanticTextEmbeddingRecord, ...]:
    records: list[SemanticTextEmbeddingRecord] = []
    for line, record in _read_jsonl(path):
        _keys(record, _TEXT_EMBEDDING_KEYS, f"semantic text embedding line {line}")
        _schema(
            record["schema_version"],
            SEMANTIC_TEXT_EMBEDDING_SCHEMA_VERSION,
            f"semantic text embedding line {line}",
        )
        records.append(
            SemanticTextEmbeddingRecord(
                schema_version=_text(record["schema_version"], "schema_version"),
                sample_id=_text(record["sample_id"], "sample_id"),
                source_sentence_name=_text(record["source_sentence_name"], "source_sentence_name"),
                text=_text(record["text"], "text"),
                backend=_text(record["backend"], "backend"),
                embedding_dim=_int(record["embedding_dim"], "embedding_dim"),
                pooling=_text(record["pooling"], "pooling"),
                trainable=_bool(record["trainable"], "trainable"),
                embedding=_array(record["embedding"], "embedding"),
                proxy_only=_bool(record["proxy_only"], "proxy_only"),
                config_hash=_text(record["config_hash"], "config_hash"),
            )
        )
    return tuple(records)


def write_semantic_pose_embeddings_jsonl(
    path: Path,
    records: Iterable[SemanticPoseEmbeddingRecord],
) -> None:
    _write_jsonl(path, (_typed_dict(record, SemanticPoseEmbeddingRecord) for record in records))


def read_semantic_pose_embeddings_jsonl(path: Path) -> tuple[SemanticPoseEmbeddingRecord, ...]:
    records: list[SemanticPoseEmbeddingRecord] = []
    for line, record in _read_jsonl(path):
        _keys(record, _POSE_EMBEDDING_KEYS, f"semantic pose embedding line {line}")
        _schema(
            record["schema_version"],
            SEMANTIC_POSE_EMBEDDING_SCHEMA_VERSION,
            f"semantic pose embedding line {line}",
        )
        records.append(
            SemanticPoseEmbeddingRecord(
                schema_version=_text(record["schema_version"], "schema_version"),
                sample_id=_text(record["sample_id"], "sample_id"),
                split=_split(record["split"]),
                generation_index=_int(record["generation_index"], "generation_index"),
                candidate_policy=_text(record["candidate_policy"], "candidate_policy"),
                source_sentence_name=_text(record["source_sentence_name"], "source_sentence_name"),
                backend=_text(record["backend"], "backend"),
                embedding_dim=_int(record["embedding_dim"], "embedding_dim"),
                coordinate_mode=_text(record["coordinate_mode"], "coordinate_mode"),
                channel_groups=tuple(
                    _text(channel, "channel_groups")
                    for channel in _sequence(record["channel_groups"], "channel_groups")
                ),
                include_velocity_statistics=_bool(
                    record["include_velocity_statistics"], "include_velocity_statistics"
                ),
                include_validity_statistics=_bool(
                    record["include_validity_statistics"], "include_validity_statistics"
                ),
                projection_seed=_int(record["projection_seed"], "projection_seed"),
                embedding=_optional_array(record["embedding"], "embedding"),
                valid_observation_count=_int(
                    record["valid_observation_count"], "valid_observation_count"
                ),
                skipped=_bool(record["skipped"], "skipped"),
                reason=_optional_text(record["reason"], "reason"),
                proxy_only=_bool(record["proxy_only"], "proxy_only"),
                spec_hash=_text(record["spec_hash"], "spec_hash"),
            )
        )
    return tuple(records)


def write_semantic_alignment_results_jsonl(
    path: Path,
    records: Iterable[SemanticAlignmentResult],
) -> None:
    _write_jsonl(path, (_typed_dict(record, SemanticAlignmentResult) for record in records))


def read_semantic_alignment_results_jsonl(path: Path) -> tuple[SemanticAlignmentResult, ...]:
    records: list[SemanticAlignmentResult] = []
    for line, record in _read_jsonl(path):
        _keys(record, _ALIGNMENT_KEYS, f"semantic alignment result line {line}")
        _schema(
            record["schema_version"],
            SEMANTIC_ALIGNMENT_RESULT_SCHEMA_VERSION,
            f"semantic alignment result line {line}",
        )
        records.append(
            SemanticAlignmentResult(
                schema_version=_text(record["schema_version"], "schema_version"),
                sample_id=_text(record["sample_id"], "sample_id"),
                split=_split(record["split"]),
                generation_index=_int(record["generation_index"], "generation_index"),
                candidate_policy=_text(record["candidate_policy"], "candidate_policy"),
                model_key=_model(record["model_key"]),
                objective_key=_objective(record["objective_key"]),
                metric=_text(record["metric"], "metric"),
                cosine_similarity=_optional_float(record["cosine_similarity"], "cosine_similarity"),
                cosine_distance=_optional_float(record["cosine_distance"], "cosine_distance"),
                loss_weight=_float(record["loss_weight"], "loss_weight"),
                loss_value=_optional_float(record["loss_value"], "loss_value"),
                skipped=_bool(record["skipped"], "skipped"),
                reason=_optional_text(record["reason"], "reason"),
                proxy_only=_bool(record["proxy_only"], "proxy_only"),
            )
        )
    return tuple(records)


def write_semantic_alignment_aggregate_json(
    path: Path,
    aggregate: SemanticAlignmentAggregate,
) -> None:
    _instance(aggregate, SemanticAlignmentAggregate, "aggregate")
    _write_json(path, aggregate.to_dict())


def write_semantic_ablation_readiness_json(
    path: Path,
    readiness: SemanticAblationReadinessResult,
) -> None:
    _instance(readiness, SemanticAblationReadinessResult, "readiness")
    _write_json(path, readiness.to_dict())


def read_semantic_ablation_readiness_json(path: Path) -> SemanticAblationReadinessResult:
    record = _read_json(path, "semantic ablation readiness")
    _keys(record, _READINESS_KEYS, "semantic ablation readiness")
    _schema(
        record["schema_version"],
        SEMANTIC_ABLATION_READINESS_SCHEMA_VERSION,
        "semantic ablation readiness",
    )
    return SemanticAblationReadinessResult(
        schema_version=_text(record["schema_version"], "schema_version"),
        objective_key=_objective(record["objective_key"]),
        model_key=_model(record["model_key"]),
        objective_run_name=_text(record["objective_run_name"], "objective_run_name"),
        baseline_run_name=_optional_text(record["baseline_run_name"], "baseline_run_name"),
        ready_for_comparison=_bool(record["ready_for_comparison"], "ready_for_comparison"),
        required_baseline_missing=_bool(
            record["required_baseline_missing"], "required_baseline_missing"
        ),
        same_model_required=_bool(record["same_model_required"], "same_model_required"),
        same_manifest_family_required=_bool(
            record["same_manifest_family_required"], "same_manifest_family_required"
        ),
        same_splits_required=_bool(record["same_splits_required"], "same_splits_required"),
        same_validation_protocol_required=_bool(
            record["same_validation_protocol_required"], "same_validation_protocol_required"
        ),
        issues=_text_tuple(record["issues"], "issues"),
        caveats=_text_tuple(record["caveats"], "caveats"),
    )


def read_semantic_alignment_aggregate_json(path: Path) -> SemanticAlignmentAggregate:
    record = _read_json(path, "semantic alignment aggregate")
    _keys(record, _AGGREGATE_KEYS, "semantic alignment aggregate")
    _schema(
        record["schema_version"],
        SEMANTIC_ALIGNMENT_AGGREGATE_SCHEMA_VERSION,
        "semantic alignment aggregate",
    )
    return SemanticAlignmentAggregate(
        schema_version=_text(record["schema_version"], "schema_version"),
        objective_key=_objective(record["objective_key"]),
        model_key=_model(record["model_key"]),
        split=_split(record["split"]),
        candidate_policy=_text(record["candidate_policy"], "candidate_policy"),
        records_count=_int(record["records_count"], "records_count"),
        skipped_count=_int(record["skipped_count"], "skipped_count"),
        mean_cosine_similarity=_optional_float(
            record["mean_cosine_similarity"], "mean_cosine_similarity"
        ),
        mean_cosine_distance=_optional_float(
            record["mean_cosine_distance"], "mean_cosine_distance"
        ),
        mean_loss_value=_optional_float(record["mean_loss_value"], "mean_loss_value"),
        caveats=_text_tuple(record["caveats"], "caveats"),
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_dumps(record))
            handle.write("\n")


def _dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticConsistencyError(f"{label} JSON could not be read: {exc}") from exc
    return _mapping(value, label)


def _read_jsonl(path: Path) -> tuple[tuple[int, Mapping[str, object]], ...]:
    rows: list[tuple[int, Mapping[str, object]]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SemanticConsistencyError(
                        f"semantic JSONL is malformed at line {line_number}: {exc}"
                    ) from exc
                rows.append((line_number, _mapping(value, f"JSONL line {line_number}")))
    except (OSError, UnicodeDecodeError) as exc:
        raise SemanticConsistencyError(f"semantic JSONL could not be read: {exc}") from exc
    return tuple(rows)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SemanticConsistencyError(f"{label} must be a JSON object with string keys.")
    return value


def _keys(record: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    if set(record) != expected:
        raise SemanticConsistencyError(
            f"{label} keys mismatch: expected={sorted(expected)}, observed={sorted(record)}."
        )


def _schema(value: object, expected: str, label: str) -> None:
    if value != expected:
        raise SemanticConsistencyError(
            f"{label} schema_version is unsupported: expected {expected!r}, observed {value!r}."
        )


def _typed_dict(value: object, expected: type) -> dict[str, object]:
    _instance(value, expected, "record")
    return value.to_dict()


def _instance(value: object, expected: type, label: str) -> None:
    if not isinstance(value, expected):
        raise SemanticConsistencyError(f"{label} must be a {expected.__name__}.")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticConsistencyError(f"{name} must be non-empty.")
    return value


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise SemanticConsistencyError(f"{name} must be a boolean.")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SemanticConsistencyError(f"{name} must be an integer.")
    return value


def _float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise SemanticConsistencyError(f"{name} must be a number.")
    return float(value)


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise SemanticConsistencyError(f"{name} must be a JSON list.")
    return tuple(value)


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise SemanticConsistencyError(f"{name} must be a number or null.")
    return float(value)


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SemanticConsistencyError(f"{name} must be a list.")
    return tuple(_text(item, name) for item in value)


def _array(value: object, name: str) -> np.ndarray:
    if not isinstance(value, list):
        raise SemanticConsistencyError(f"{name} must be a JSON list.")
    try:
        return np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise SemanticConsistencyError(f"{name} must contain numeric values.") from exc


def _optional_array(value: object, name: str) -> np.ndarray | None:
    return None if value is None else _array(value, name)


def _model(value: object) -> ModelKey:
    try:
        return ModelKey(_text(value, "model_key"))
    except ValueError as exc:
        raise SemanticConsistencyError(f"unknown model_key in semantic artifact: {value!r}.") from exc


def _objective(value: object) -> ObjectiveKey:
    try:
        return ObjectiveKey(_text(value, "objective_key"))
    except ValueError as exc:
        raise SemanticConsistencyError(f"unknown objective_key in semantic artifact: {value!r}.") from exc


def _split(value: object) -> SampleSplit:
    try:
        return SampleSplit(_text(value, "split"))
    except ValueError as exc:
        raise SemanticConsistencyError(f"unknown split in semantic artifact: {value!r}.") from exc


__all__ = [
    "SEMANTIC_OBJECTIVE_CONFIG_ARTIFACT_SCHEMA_VERSION",
    "read_semantic_ablation_plan_json",
    "read_semantic_ablation_readiness_json",
    "read_semantic_alignment_aggregate_json",
    "read_semantic_alignment_results_jsonl",
    "read_semantic_attachment_decision_json",
    "read_semantic_objective_config_json",
    "read_semantic_pose_embeddings_jsonl",
    "read_semantic_text_embeddings_jsonl",
    "write_semantic_ablation_plan_json",
    "write_semantic_ablation_readiness_json",
    "write_semantic_alignment_aggregate_json",
    "write_semantic_alignment_results_jsonl",
    "write_semantic_attachment_decision_json",
    "write_semantic_objective_config_json",
    "write_semantic_pose_embeddings_jsonl",
    "write_semantic_text_embeddings_jsonl",
]
