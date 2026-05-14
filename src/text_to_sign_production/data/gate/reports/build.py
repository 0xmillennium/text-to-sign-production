"""Gate-stage report bundle construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    DroppedSample,
    GateDecisionBundle,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.data.gate.reports.sections import (
    build_checkpoint_integrity_section,
    build_dropped_sample_payload_section,
    build_gate_outcomes_section,
    build_manifest_outcomes_section,
    build_pose_health_section,
    build_source_coverage_section,
)
from text_to_sign_production.data.gate.reports.types import GateReportBundle
from text_to_sign_production.data.gate.reports.validate import validate_gate_report_bundle


def build_gate_report_bundle(
    *,
    schema_version: str,
    payloads: Sequence[PreparedSample],
    passed_entries: Sequence[PassedManifestEntry],
    dropped_entries: Sequence[DroppedManifestEntry],
    gate_bundles: Sequence[GateDecisionBundle],
    dropped_sample_payloads_by_ref: Mapping[str, DroppedSample],
) -> GateReportBundle:
    """Build the root gate-stage report projection bundle."""
    bundle = GateReportBundle(
        schema_version=schema_version,
        source_coverage=build_source_coverage_section(payloads),
        manifest_outcomes=build_manifest_outcomes_section(passed_entries, dropped_entries),
        pose_health=build_pose_health_section(payloads),
        gate_outcomes=build_gate_outcomes_section(gate_bundles),
        checkpoint_integrity=build_checkpoint_integrity_section(
            payloads,
            passed_entries,
            dropped_entries,
        ),
        dropped_sample_payloads=build_dropped_sample_payload_section(
            dropped_entries,
            dropped_sample_payloads_by_ref=dropped_sample_payloads_by_ref,
        ),
    )
    issues = validate_gate_report_bundle(bundle)
    if issues:
        raise ValueError(f"Invalid gate report bundle: {issues}")
    return bundle


__all__ = ["build_gate_report_bundle"]
