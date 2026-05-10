"""Samples-stage report bundle construction."""

from __future__ import annotations

from collections.abc import Sequence

from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    GateDecisionBundle,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.data.gate.reports.sections import (
    build_checkpoint_integrity_section,
    build_gate_outcomes_section,
    build_matching_outcomes_section,
    build_pose_health_section,
    build_source_coverage_section,
)
from text_to_sign_production.data.gate.reports.types import SamplesReportBundle
from text_to_sign_production.data.gate.reports.validate import validate_samples_report_bundle


def build_samples_report_bundle(
    *,
    schema_version: str,
    payloads: Sequence[PreparedSample],
    passed_entries: Sequence[PassedManifestEntry],
    dropped_entries: Sequence[DroppedManifestEntry],
    gate_bundles: Sequence[GateDecisionBundle],
) -> SamplesReportBundle:
    """Build the root samples-stage report projection bundle."""
    bundle = SamplesReportBundle(
        schema_version=schema_version,
        source_coverage=build_source_coverage_section(payloads),
        matching_outcomes=build_matching_outcomes_section(passed_entries, dropped_entries),
        pose_health=build_pose_health_section(payloads),
        gate_outcomes=build_gate_outcomes_section(gate_bundles),
        checkpoint_integrity=build_checkpoint_integrity_section(
            payloads,
            passed_entries,
            dropped_entries,
        ),
    )
    issues = validate_samples_report_bundle(bundle)
    if issues:
        raise ValueError(f"Invalid samples report bundle: {issues}")
    return bundle


__all__ = ["build_samples_report_bundle"]
