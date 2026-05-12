"""Gate-stage report bundle construction."""

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
    build_dropped_debug_payload_section,
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
    materialize_dropped_debug_payloads: bool,
    dropped_debug_payload_written_count: int,
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
        dropped_debug_payloads=build_dropped_debug_payload_section(
            dropped_entries,
            materialize_dropped_debug_payloads=materialize_dropped_debug_payloads,
            dropped_debug_payload_written_count=dropped_debug_payload_written_count,
        ),
    )
    issues = validate_gate_report_bundle(bundle)
    if issues:
        raise ValueError(f"Invalid gate report bundle: {issues}")
    return bundle


__all__ = ["build_gate_report_bundle"]
