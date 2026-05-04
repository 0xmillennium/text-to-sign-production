"""Leakage tier policy evaluation."""

from __future__ import annotations

from text_to_sign_production.data.leakages.severity import LEAKAGE_SEVERITY_RANK
from text_to_sign_production.data.leakages.types import (
    LeakageSampleSummary,
)
from text_to_sign_production.data.tiers.types import TierLeakageFailure, TierPolicy


def evaluate_leakage_policy(
    summary: LeakageSampleSummary,
    policy: TierPolicy,
) -> TierLeakageFailure | None:
    """Evaluate a sample leakage summary against a tier policy."""
    if (
        LEAKAGE_SEVERITY_RANK[summary.max_severity]
        <= LEAKAGE_SEVERITY_RANK[policy.max_allowed_leakage_severity]
    ):
        return None
    return TierLeakageFailure(
        reason_code="max_allowed_leakage_severity_exceeded",
        actual_max_severity=summary.max_severity,
        allowed_max_severity=policy.max_allowed_leakage_severity,
        matched_samples=summary.matched_samples,
    )
