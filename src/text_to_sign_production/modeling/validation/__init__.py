"""Validation-split comparison contracts for model-generated pose outputs."""

from text_to_sign_production.modeling.validation.aggregation import (
    aggregate_validation_metric_results,
)
from text_to_sign_production.modeling.validation.channel_metrics import (
    compute_validation_channel_metric_results,
)
from text_to_sign_production.modeling.validation.comparison import (
    ValidationComparisonResult,
    ValidationMetricComparisonRow,
    ValidationSharedPairingKey,
    ValidationSurfaceAggregateSummary,
    build_validation_comparison,
)
from text_to_sign_production.modeling.validation.config import (
    VALIDATION_ENGINE_SCHEMA_VERSION,
    ValidationComparisonConfig,
    ValidationEngineConfig,
    ValidationEngineDataConfig,
    ValidationEngineIdentityConfig,
    ValidationEngineReportsConfig,
    ValidationMetricSelectionConfig,
    ValidationSurfaceConfig,
    default_higher_is_better_metric_keys,
    default_lower_is_better_metric_keys,
    load_validation_engine_config,
)
from text_to_sign_production.modeling.validation.engine import (
    ValidationEngineRunResult,
    ValidationSurfaceRunResult,
    run_candidate_agnostic_validation,
)
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.identifiers import (
    ValidationChannelMetricKey,
    ValidationMetricKey,
)
from text_to_sign_production.modeling.validation.inputs import (
    ValidationGeneratedInput,
    ValidationReferenceInput,
)
from text_to_sign_production.modeling.validation.io import (
    read_validation_aggregate_metrics_json,
    read_validation_channel_metric_results_jsonl,
    read_validation_limitations_json,
    read_validation_metric_results_jsonl,
    read_validation_pairing_jsonl,
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
    VALIDATION_SCHEMA_VERSION,
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

__all__ = [
    "VALIDATION_SCHEMA_VERSION",
    "VALIDATION_ENGINE_SCHEMA_VERSION",
    "ModelValidationError",
    "ValidationAggregateMetric",
    "ValidationChannelMetricKey",
    "ValidationChannelMetricResult",
    "ValidationComparisonConfig",
    "ValidationComparisonResult",
    "ValidationEngineConfig",
    "ValidationEngineDataConfig",
    "ValidationEngineIdentityConfig",
    "ValidationEngineReportPaths",
    "ValidationEngineReportsConfig",
    "ValidationEngineRunResult",
    "ValidationGeneratedInput",
    "ValidationGeneratedSurface",
    "ValidationLimitations",
    "ValidationMetricKey",
    "ValidationMetricComparisonRow",
    "ValidationMetricResult",
    "ValidationMetricSelectionConfig",
    "ValidationPairingEntry",
    "ValidationPairingKey",
    "ValidationReferenceInput",
    "ValidationSharedPairingKey",
    "ValidationSurfaceAggregateSummary",
    "ValidationSurfaceConfig",
    "ValidationSurfaceRunResult",
    "aggregate_validation_metric_results",
    "build_validation_limitations",
    "build_validation_pairing",
    "build_validation_comparison",
    "compute_validation_channel_metric_results",
    "compute_validation_metric_results",
    "default_higher_is_better_metric_keys",
    "default_lower_is_better_metric_keys",
    "load_validation_engine_config",
    "read_validation_aggregate_metrics_json",
    "read_validation_channel_metric_results_jsonl",
    "read_validation_limitations_json",
    "read_validation_metric_results_jsonl",
    "read_validation_pairing_jsonl",
    "resolve_generated_surface",
    "run_candidate_agnostic_validation",
    "write_validation_engine_reports",
    "write_validation_aggregate_metrics_json",
    "write_validation_channel_metric_results_jsonl",
    "write_validation_limitations_json",
    "write_validation_metric_results_jsonl",
    "write_validation_pairing_jsonl",
]
