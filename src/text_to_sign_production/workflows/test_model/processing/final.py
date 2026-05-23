"""Final result assembly for test_model."""

from __future__ import annotations

from text_to_sign_production.workflows.test_model.contracts import TestModelFinalResult


def build_test_model_final_result(**kwargs) -> TestModelFinalResult:
    return TestModelFinalResult(**kwargs)


__all__ = ["build_test_model_final_result"]
