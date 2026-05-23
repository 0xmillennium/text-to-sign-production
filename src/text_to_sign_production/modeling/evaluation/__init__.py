"""Evaluation protocol orchestration over generated-pose validation surfaces."""

from text_to_sign_production.modeling.evaluation.config import (
    EVALUATION_PROTOCOL_SCHEMA_VERSION,
    EvaluationContrastConfig,
    EvaluationProtocolConfig,
    EvaluationProtocolExecutionConfig,
    EvaluationProtocolIdentityConfig,
    EvaluationProtocolReportsConfig,
    EvaluationValidationEngineConfig,
    EvaluationVariantConfig,
    load_evaluation_protocol_config,
)
from text_to_sign_production.modeling.evaluation.contrasts import (
    EvaluationContrastReadiness,
    evaluate_contrast_readiness,
)
from text_to_sign_production.modeling.evaluation.errors import EvaluationProtocolError
from text_to_sign_production.modeling.evaluation.inventory import (
    EvaluationSurfaceInventory,
    EvaluationSurfaceInventoryEntry,
    build_planned_run_request,
    build_planned_run_requests,
    build_surface_inventory,
)
from text_to_sign_production.modeling.evaluation.runner import (
    EvaluationProtocolRunResult,
    run_evaluation_protocol,
)
from text_to_sign_production.modeling.evaluation.variants import EvaluationVariant

__all__ = [
    "EVALUATION_PROTOCOL_SCHEMA_VERSION",
    "EvaluationContrastConfig",
    "EvaluationContrastReadiness",
    "EvaluationProtocolConfig",
    "EvaluationProtocolError",
    "EvaluationProtocolExecutionConfig",
    "EvaluationProtocolIdentityConfig",
    "EvaluationProtocolReportsConfig",
    "EvaluationProtocolRunResult",
    "EvaluationSurfaceInventory",
    "EvaluationSurfaceInventoryEntry",
    "EvaluationValidationEngineConfig",
    "EvaluationVariant",
    "EvaluationVariantConfig",
    "build_planned_run_request",
    "build_planned_run_requests",
    "build_surface_inventory",
    "evaluate_contrast_readiness",
    "load_evaluation_protocol_config",
    "run_evaluation_protocol",
]
