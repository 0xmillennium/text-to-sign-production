"""Processing steps for the single-sample test_model workflow."""

from text_to_sign_production.workflows.test_model.processing.checkpoint import select_checkpoint
from text_to_sign_production.workflows.test_model.processing.comparison import (
    compare_reference_and_generated,
)
from text_to_sign_production.workflows.test_model.processing.evidence import collect_sample_evidence
from text_to_sign_production.workflows.test_model.processing.final import build_test_model_final_result
from text_to_sign_production.workflows.test_model.processing.inference import (
    run_test_model_sample_inference,
)
from text_to_sign_production.workflows.test_model.processing.model_run import resolve_model_run
from text_to_sign_production.workflows.test_model.processing.preflight import (
    run_test_model_preflight,
)
from text_to_sign_production.workflows.test_model.processing.reports import write_test_model_reports
from text_to_sign_production.workflows.test_model.processing.smoke import (
    build_test_model_smoke_execution_protocol,
)
from text_to_sign_production.workflows.test_model.processing.target import resolve_target_sample
from text_to_sign_production.workflows.test_model.processing.visualization import (
    render_test_model_visualization,
)

__all__ = [
    "build_test_model_final_result",
    "build_test_model_smoke_execution_protocol",
    "compare_reference_and_generated",
    "collect_sample_evidence",
    "render_test_model_visualization",
    "resolve_model_run",
    "resolve_target_sample",
    "run_test_model_preflight",
    "run_test_model_sample_inference",
    "select_checkpoint",
    "write_test_model_reports",
]
