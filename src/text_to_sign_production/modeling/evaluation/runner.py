"""Evaluation protocol runner over existing generated-pose surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import ModelingManifestFamily
from text_to_sign_production.modeling.evaluation.config import EvaluationProtocolConfig
from text_to_sign_production.modeling.evaluation.contrasts import (
    EvaluationContrastReadiness,
    evaluate_contrast_readiness,
)
from text_to_sign_production.modeling.evaluation.errors import EvaluationProtocolError
from text_to_sign_production.modeling.evaluation.inventory import (
    EvaluationSurfaceInventory,
    build_planned_run_requests,
    build_surface_inventory,
)
from text_to_sign_production.modeling.evaluation.reports import (
    EvaluationProtocolReportPaths,
    write_evaluation_protocol_reports,
)
from text_to_sign_production.modeling.validation import (
    ValidationComparisonConfig,
    ValidationEngineConfig,
    ValidationEngineDataConfig,
    ValidationEngineIdentityConfig,
    ValidationEngineReportsConfig,
    ValidationMetricSelectionConfig,
    ValidationSurfaceConfig,
    default_higher_is_better_metric_keys,
    default_lower_is_better_metric_keys,
    run_candidate_agnostic_validation,
)


@dataclass(frozen=True, slots=True)
class EvaluationProtocolRunResult:
    run_name: str
    manifest_family: ModelingManifestFamily
    protocol_config_path: Path | None
    inventory_path: Path
    planned_run_requests_path: Path
    validation_engine_result_path: Path | None
    contrast_readiness_path: Path
    claim_readiness_summary_path: Path
    report_paths: EvaluationProtocolReportPaths | None
    ready_variant_count: int
    missing_variant_count: int
    claim_ready_count: int
    inventory: EvaluationSurfaceInventory
    contrast_readiness: tuple[EvaluationContrastReadiness, ...]


def run_evaluation_protocol(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    config: EvaluationProtocolConfig,
    run_name: str,
) -> EvaluationProtocolRunResult:
    """Run protocol/readiness orchestration without generating missing surfaces."""

    if (
        config.identity.manifest_family is not None
        and config.identity.manifest_family.family_id != manifest_family.family_id
    ):
        raise EvaluationProtocolError(
            "evaluation protocol config manifest_family="
            f"{config.identity.manifest_family.family_id!r} does not match runner "
            f"manifest_family={manifest_family.family_id!r}."
        )
    if config.split is not SampleSplit.VAL:
        raise EvaluationProtocolError("evaluation protocol currently supports split='val' only.")
    report_root = topology.reports.modeling_root / config.reports.output_dir_name / run_name
    inventory = build_surface_inventory(
        topology=topology,
        config=config,
        manifest_family_id=manifest_family.family_id,
    )
    planned_requests = build_planned_run_requests(inventory)
    ready_entries = inventory.ready_entries
    if (
        config.execution.run_validation_engine
        and config.validation_engine.enabled
        and not ready_entries
        and config.execution.mode == "validate_existing"
    ):
        raise EvaluationProtocolError(
            "evaluation protocol did not find any ready generated surfaces."
        )
    validation_result = None
    validation_engine_result_path = None
    if config.execution.run_validation_engine and config.validation_engine.enabled and ready_entries:
        validation_result = run_candidate_agnostic_validation(
            topology=topology,
            manifest_family=manifest_family,
            config=_validation_engine_config(config, ready_entries),
            run_name=run_name,
        )
        if validation_result.report_paths is not None:
            validation_engine_result_path = validation_result.report_paths.comparison_json_path
    validation_comparison = None if validation_result is None else validation_result.comparison
    readiness = evaluate_contrast_readiness(
        config=config,
        inventory=inventory,
        validation_comparison=validation_comparison,
        paired_keys_by_variant_id=_paired_keys_by_variant_id(validation_result),
    )
    report_paths = (
        write_evaluation_protocol_reports(
            report_root=report_root,
            config=config,
            inventory=inventory,
            planned_run_requests=planned_requests,
            readiness=readiness,
            validation_engine_result_path=validation_engine_result_path,
        )
        if config.reports.enabled
        else None
    )
    if report_paths is None:
        inventory_path = report_root / "surface_inventory.jsonl"
        planned_path = report_root / "planned_run_requests.jsonl"
        readiness_path = report_root / "contrast_readiness.jsonl"
        claim_summary_path = report_root / "claim_readiness_summary.json"
    else:
        inventory_path = report_paths.surface_inventory_jsonl_path
        planned_path = report_paths.planned_run_requests_path
        readiness_path = report_paths.contrast_readiness_path
        claim_summary_path = report_paths.claim_readiness_summary_path
    return EvaluationProtocolRunResult(
        run_name=run_name,
        manifest_family=manifest_family,
        protocol_config_path=config.source_path,
        inventory_path=inventory_path,
        planned_run_requests_path=planned_path,
        validation_engine_result_path=validation_engine_result_path,
        contrast_readiness_path=readiness_path,
        claim_readiness_summary_path=claim_summary_path,
        report_paths=report_paths,
        ready_variant_count=len(inventory.ready_entries),
        missing_variant_count=len(inventory.missing_entries),
        claim_ready_count=sum(item.claim_ready for item in readiness),
        inventory=inventory,
        contrast_readiness=readiness,
    )


def _paired_keys_by_variant_id(validation_result: object | None) -> dict[str, set[tuple[str, int]]]:
    if validation_result is None:
        return {}
    paired: dict[str, set[tuple[str, int]]] = {}
    for surface_result in getattr(validation_result, "surface_results", ()):
        surface = getattr(surface_result, "surface")
        variant_id = surface.label
        keys: set[tuple[str, int]] = set()
        for pair in getattr(surface_result, "pairing", ()):
            if getattr(pair, "status", None) != "paired":
                continue
            key = getattr(pair, "key")
            keys.add((key.sample_id, key.generation_index))
        paired[variant_id] = keys
    return paired


def _validation_engine_config(
    config: EvaluationProtocolConfig,
    ready_entries: tuple[object, ...],
) -> ValidationEngineConfig:
    return ValidationEngineConfig(
        source_path=None,
        raw_config={},
        identity=ValidationEngineIdentityConfig(
            schema_version="t2sp-validation-engine-v1",
            manifest_family=config.identity.manifest_family,
        ),
        data=ValidationEngineDataConfig(
            split=SampleSplit.VAL,
            minimum_paired_count=1,
        ),
        surfaces=tuple(
            ValidationSurfaceConfig(
                label=entry.variant_id,
                producer_type=entry.producer_type,
                producer_key=entry.producer_key,
                run_name=entry.run_name,
                required=True,
            )
            for entry in ready_entries
        ),
        metrics=ValidationMetricSelectionConfig(),
        comparison=ValidationComparisonConfig(
            enabled=True,
            require_shared_pairing_subset=config.validation_engine.require_shared_pairing_subset,
            lower_is_better=default_lower_is_better_metric_keys(),
            higher_is_better=default_higher_is_better_metric_keys(),
        ),
        reports=ValidationEngineReportsConfig(
            enabled=True,
            output_dir_name=config.validation_engine.output_dir_name,
        ),
    )


__all__ = ["EvaluationProtocolRunResult", "run_evaluation_protocol"]
