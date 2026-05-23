"""Digest verification for test_model published outputs."""

from __future__ import annotations

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelPublishExecution,
    TestModelPublishVerification,
)


def verify_test_model_publish(
    execution: TestModelPublishExecution,
) -> TestModelPublishVerification:
    checks = []
    for target in execution.plan.targets:
        if not target.target_path.is_file():
            checks.append((target, False, "published target is missing"))
            continue
        if sha256_file(target.target_path) != target.source_sha256:
            checks.append((target, False, "published target digest mismatch"))
            continue
        checks.append((target, True, None))
    return TestModelPublishVerification(checks=tuple(checks))


__all__ = ["verify_test_model_publish"]
