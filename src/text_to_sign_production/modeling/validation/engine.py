"""Candidate-agnostic validation engine over generated-pose surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import (
    ModelingDataError,
    ModelingManifestFamily,
    load_manifest_sample,
    read_modeling_manifest,
    resolve_prepared_payload_path,
)
from text_to_sign_production.modeling.validation.aggregation import (
    aggregate_validation_metric_results,
)
from text_to_sign_production.modeling.validation.channel_metrics import (
    compute_validation_channel_metric_results,
)
from text_to_sign_production.modeling.validation.comparison import (
    ValidationComparisonResult,
    build_validation_comparison,
)
from text_to_sign_production.modeling.validation.config import ValidationEngineConfig
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.inputs import (
    ValidationGeneratedInput,
    ValidationReferenceInput,
)
from text_to_sign_production.modeling.validation.io import (
    write_validation_aggregate_metrics_json,
    write_validation_channel_metric_results_jsonl,
    write_validation_limitations_json,
    write_validation_metric_results_jsonl,
    write_validation_pairing_jsonl,
)
from text_to_sign_production.modeling.validation.limitations import (
    build_validation_limitations,
)
from text_to_sign_production.modeling.validation.metrics import (
    compute_validation_metric_results,
)
from text_to_sign_production.modeling.validation.pairing import build_validation_pairing
from text_to_sign_production.modeling.validation.records import (
    ValidationAggregateMetric,
    ValidationChannelMetricResult,
    ValidationLimitations,
    ValidationMetricResult,
    ValidationPairingEntry,
    ValidationPairingKey,
)
from text_to_sign_production.modeling.validation.reports import (
    ValidationEngineReportPaths,
    write_validation_engine_reports,
)
from text_to_sign_production.modeling.validation.surfaces import (
    ValidationGeneratedSurface,
    resolve_generated_surface,
)

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class ValidationSurfaceRunResult:
    surface: ValidationGeneratedSurface
    pairing: tuple[ValidationPairingEntry, ...]
    metric_results: tuple[ValidationMetricResult, ...]
    channel_metric_results: tuple[ValidationChannelMetricResult, ...]
    aggregates: tuple[ValidationAggregateMetric, ...]
    channel_aggregates: tuple[ValidationAggregateMetric, ...]
    limitations: ValidationLimitations
    output_root: Path
    pairing_path: Path
    metric_results_path: Path
    channel_metric_results_path: Path
    aggregate_metrics_path: Path
    channel_aggregate_metrics_path: Path
    limitations_path: Path


@dataclass(frozen=True, slots=True)
class ValidationEngineRunResult:
    run_name: str
    manifest_family: ModelingManifestFamily
    split: SampleSplit
    report_root: Path
    surface_results: tuple[ValidationSurfaceRunResult, ...]
    skipped_surface_reasons: tuple[str, ...]
    comparison: ValidationComparisonResult | None
    report_paths: ValidationEngineReportPaths | None


def run_candidate_agnostic_validation(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    config: ValidationEngineConfig,
    run_name: str,
) -> ValidationEngineRunResult:
    """Validate generated-pose model/comparator surfaces under one contract."""

    if config.data.split is not SampleSplit.VAL:
        raise ModelValidationError("candidate-agnostic validation currently supports split='val' only.")
    if (
        config.identity.manifest_family is not None
        and config.identity.manifest_family.family_id != manifest_family.family_id
    ):
        raise ModelValidationError(
            "validation engine config manifest_family="
            f"{config.identity.manifest_family.family_id!r} does not match runner "
            f"manifest_family={manifest_family.family_id!r}."
        )
    try:
        reference_manifest = read_modeling_manifest(topology, manifest_family, SampleSplit.VAL)
        reference_inputs = tuple(
            ValidationReferenceInput(entry, resolve_prepared_payload_path(topology, entry))
            for entry in reference_manifest.entries
        )
    except (FileNotFoundError, OSError, ModelingDataError, ValueError) as exc:
        raise ModelValidationError(f"validation reference manifest is invalid: {exc}") from exc

    report_root = topology.reports.modeling_root / config.reports.output_dir_name / run_name
    surface_results: list[ValidationSurfaceRunResult] = []
    skipped: list[str] = []
    for surface_config in config.surfaces:
        surface_run_name = surface_config.run_name or run_name
        surface = resolve_generated_surface(
            topology=topology,
            surface_config=surface_config,
            split=SampleSplit.VAL,
            run_name_override=surface_run_name,
        )
        if surface is None:
            skipped.append(
                f"optional validation surface {surface_config.label!r} is missing generated-pose "
                f"manifest for run_name={surface_run_name!r}."
            )
            continue
        result = _validate_surface(
            topology=topology,
            manifest_family=manifest_family,
            reference_manifest_path=reference_manifest.manifest_path,
            reference_inputs=reference_inputs,
            surface=surface,
            report_root=report_root,
            minimum_paired_count=config.data.minimum_paired_count,
            compute_full_metrics=config.metrics.full_bfh_metrics,
            compute_channel_metrics=config.metrics.channel_metrics,
        )
        surface_results.append(result)
    if not surface_results:
        raise ModelValidationError("candidate-agnostic validation did not validate any generated surfaces.")

    comparison = (
        build_validation_comparison(
            surface_results=tuple(surface_results),
            config=config.comparison,
        )
        if config.comparison.enabled
        else None
    )
    report_paths = (
        write_validation_engine_reports(
            report_root=report_root,
            surface_results=tuple(surface_results),
            comparison=comparison,
        )
        if config.reports.enabled
        else None
    )
    return ValidationEngineRunResult(
        run_name=run_name,
        manifest_family=manifest_family,
        split=SampleSplit.VAL,
        report_root=report_root,
        surface_results=tuple(surface_results),
        skipped_surface_reasons=tuple(skipped),
        comparison=comparison,
        report_paths=report_paths,
    )


def _validate_surface(
    *,
    topology: ArtifactTopology,
    manifest_family: ModelingManifestFamily,
    reference_manifest_path: Path,
    reference_inputs: tuple[ValidationReferenceInput, ...],
    surface: ValidationGeneratedSurface,
    report_root: Path,
    minimum_paired_count: int,
    compute_full_metrics: bool,
    compute_channel_metrics: bool,
) -> ValidationSurfaceRunResult:
    generated_inputs = tuple(
        ValidationGeneratedInput(sample.entry, sample.payload_path) for sample in surface.samples
    )
    pairing = build_validation_pairing(
        split=SampleSplit.VAL,
        reference_entries=reference_inputs,
        generated_entries=generated_inputs,
    )
    limitations = build_validation_limitations(split=SampleSplit.VAL, pairing_entries=pairing)
    if limitations.paired_count < minimum_paired_count:
        raise ModelValidationError(
            f"validation surface {surface.label!r} has paired_count={limitations.paired_count}, "
            f"below minimum_paired_count={minimum_paired_count}."
        )
    reference_by_key = {
        ValidationPairingKey(SampleSplit.VAL, value.entry.sample_id, 0): value
        for value in reference_inputs
    }
    generated_by_key = {
        ValidationPairingKey(
            SampleSplit.VAL,
            sample.entry.sample_id,
            sample.entry.generation_index,
        ): sample
        for sample in surface.samples
    }
    metric_results: list[ValidationMetricResult] = []
    channel_metric_results: list[ValidationChannelMetricResult] = []
    for pair in pairing:
        if pair.status != "paired":
            continue
        reference_input = reference_by_key[pair.key]
        generated = generated_by_key[pair.key]
        if generated.sample is None:
            raise ModelValidationError(
                f"paired generated validation payload was not loaded: {pair.key.sample_id!r}."
            )
        try:
            reference = load_manifest_sample(
                topology,
                manifest_family,
                reference_manifest_path,
                reference_input.entry,
            )
        except (FileNotFoundError, OSError, ModelingDataError, ValueError) as exc:
            raise ModelValidationError(
                f"validation reference payload is invalid for sample_id={pair.key.sample_id!r}: {exc}"
            ) from exc
        if compute_full_metrics:
            metric_results.extend(
                compute_validation_metric_results(
                    key=pair.key,
                    reference=reference.pose,
                    generated=generated.sample.pose,
                )
            )
        if compute_channel_metrics:
            channel_metric_results.extend(
                compute_validation_channel_metric_results(
                    key=pair.key,
                    reference=reference.pose,
                    generated=generated.sample.pose,
                )
            )
    missing_count = (
        limitations.generated_missing_count
        + limitations.reference_missing_count
        + limitations.failed_generated_count
        + limitations.identity_mismatch_count
    )
    issue_count = (
        sum(len(pair.issues) for pair in pairing)
        + sum(len(result.issues) for result in metric_results)
        + sum(len(result.issues) for result in channel_metric_results)
    )
    aggregates = aggregate_validation_metric_results(
        metric_results,
        split=SampleSplit.VAL,
        missing_count=missing_count,
        issue_count=issue_count,
    )
    channel_aggregates = aggregate_validation_metric_results(
        channel_metric_results,
        split=SampleSplit.VAL,
        missing_count=missing_count,
        issue_count=issue_count,
    )
    output_root = report_root / "surfaces" / _safe_token(surface.label)
    pairing_path = output_root / "pairing.jsonl"
    metric_results_path = output_root / "metric_results.jsonl"
    channel_metric_results_path = output_root / "channel_metric_results.jsonl"
    aggregate_metrics_path = output_root / "aggregate_metrics.json"
    channel_aggregate_metrics_path = output_root / "channel_aggregate_metrics.json"
    limitations_path = output_root / "limitations.json"
    write_validation_pairing_jsonl(pairing_path, pairing)
    write_validation_metric_results_jsonl(metric_results_path, metric_results)
    write_validation_channel_metric_results_jsonl(
        channel_metric_results_path,
        channel_metric_results,
    )
    write_validation_aggregate_metrics_json(aggregate_metrics_path, aggregates)
    write_validation_aggregate_metrics_json(channel_aggregate_metrics_path, channel_aggregates)
    write_validation_limitations_json(limitations_path, limitations)
    return ValidationSurfaceRunResult(
        surface=surface,
        pairing=pairing,
        metric_results=tuple(metric_results),
        channel_metric_results=tuple(channel_metric_results),
        aggregates=aggregates,
        channel_aggregates=channel_aggregates,
        limitations=limitations,
        output_root=output_root,
        pairing_path=pairing_path,
        metric_results_path=metric_results_path,
        channel_metric_results_path=channel_metric_results_path,
        aggregate_metrics_path=aggregate_metrics_path,
        channel_aggregate_metrics_path=channel_aggregate_metrics_path,
        limitations_path=limitations_path,
    )


def _safe_token(value: str) -> str:
    if not _SAFE_TOKEN.fullmatch(value):
        raise ModelValidationError(f"validation surface label is not a safe path token: {value!r}.")
    return value


__all__ = [
    "ValidationEngineRunResult",
    "ValidationSurfaceRunResult",
    "run_candidate_agnostic_validation",
]
