"""Table builders for PreparedSample-based quality reports."""

from __future__ import annotations

from text_to_sign_production.core.models import GateDecisionBundle
from text_to_sign_production.data.tier.families import QualityMetricBundle
from text_to_sign_production.data.tier.policies import TierDecisionBundle
from text_to_sign_production.data.tier.reports.types import (
    GateReportRow,
    MetricReportRow,
    QualityReportTables,
    TierReportRow,
)


def build_quality_report_tables(
    metrics: QualityMetricBundle,
    tiers: TierDecisionBundle,
    gates: GateDecisionBundle | None,
) -> QualityReportTables:
    """Build machine-readable report tables."""
    return QualityReportTables(
        metric_rows=_metric_rows(metrics),
        gate_rows=_gate_rows(gates),
        tier_rows=tuple(
            TierReportRow(
                family=decision.family,
                status=decision.status,
                best_supported_tier=decision.best_supported_tier,
                supported_tiers=decision.supported_tiers,
                issue_count=len(decision.issues),
            )
            for decision in tiers.family_decisions
        ),
    )


def _metric_rows(metrics: QualityMetricBundle) -> tuple[MetricReportRow, ...]:
    return (
        MetricReportRow(
            "oob",
            "body_out_of_bounds_frame_ratio",
            metrics.oob.body_out_of_bounds_frame_ratio,
        ),
        MetricReportRow(
            "oob",
            "left_hand_out_of_bounds_frame_ratio",
            metrics.oob.left_hand_out_of_bounds_frame_ratio,
        ),
        MetricReportRow(
            "oob",
            "right_hand_out_of_bounds_frame_ratio",
            metrics.oob.right_hand_out_of_bounds_frame_ratio,
        ),
        MetricReportRow(
            "oob",
            "face_out_of_bounds_frame_ratio",
            metrics.oob.face_out_of_bounds_frame_ratio,
        ),
        MetricReportRow(
            "oob",
            "max_channel_out_of_bounds_frame_ratio",
            metrics.oob.max_channel_out_of_bounds_frame_ratio,
        ),
        MetricReportRow(
            "upper_body_support",
            "active_span_upper_body_available_frame_ratio",
            metrics.upper_body_support.active_span_upper_body_available_frame_ratio,
        ),
        MetricReportRow(
            "upper_body_support",
            "active_span_upper_body_landmark_coverage_ratio",
            metrics.upper_body_support.active_span_upper_body_landmark_coverage_ratio,
        ),
        MetricReportRow(
            "upper_body_support",
            "max_active_span_upper_body_dropout_run_ratio",
            metrics.upper_body_support.max_active_span_upper_body_dropout_run_ratio,
        ),
        MetricReportRow(
            "manual_visibility",
            "active_span_any_hand_available_frame_ratio",
            metrics.manual_visibility.active_span_any_hand_available_frame_ratio,
        ),
        MetricReportRow(
            "manual_visibility",
            "active_span_both_hands_unavailable_frame_ratio",
            metrics.manual_visibility.active_span_both_hands_unavailable_frame_ratio,
        ),
        MetricReportRow(
            "manual_visibility",
            "max_active_span_any_hand_dropout_run_ratio",
            metrics.manual_visibility.max_active_span_any_hand_dropout_run_ratio,
        ),
        MetricReportRow(
            "non_manual_visibility",
            "active_span_face_available_frame_ratio",
            metrics.non_manual_visibility.active_span_face_available_frame_ratio,
        ),
        MetricReportRow(
            "non_manual_visibility",
            "max_active_span_face_unavailable_run_ratio",
            metrics.non_manual_visibility.max_active_span_face_unavailable_run_ratio,
        ),
        MetricReportRow(
            "confidence",
            "active_span_body_mean_confidence",
            metrics.confidence.active_span_body_mean_confidence,
        ),
        MetricReportRow(
            "confidence",
            "active_span_hand_mean_confidence",
            metrics.confidence.active_span_hand_mean_confidence,
        ),
        MetricReportRow(
            "confidence",
            "active_span_face_mean_confidence",
            metrics.confidence.active_span_face_mean_confidence,
        ),
        MetricReportRow(
            "kinematic_naturalness",
            "comparable_transition_abrupt_ratio",
            metrics.kinematic_naturalness.comparable_transition_abrupt_ratio,
        ),
        MetricReportRow(
            "kinematic_naturalness",
            "comparable_transition_discontinuity_ratio",
            metrics.kinematic_naturalness.comparable_transition_discontinuity_ratio,
        ),
        MetricReportRow(
            "kinematic_naturalness",
            "active_span_frozen_frame_ratio",
            metrics.kinematic_naturalness.active_span_frozen_frame_ratio,
        ),
        MetricReportRow(
            "tracking_quality",
            "tracked_target_missing_frame_ratio",
            metrics.tracking_quality.tracked_target_missing_frame_ratio,
        ),
        MetricReportRow(
            "tracking_quality",
            "person_tracking_continuity_break_ratio",
            metrics.tracking_quality.person_tracking_continuity_break_ratio,
        ),
        MetricReportRow(
            "tracking_quality",
            "person_tracking_reanchor_ratio",
            metrics.tracking_quality.person_tracking_reanchor_ratio,
        ),
        MetricReportRow(
            "manual_detail",
            "active_span_representative_hand_landmark_coverage_ratio",
            metrics.manual_detail.active_span_representative_hand_landmark_coverage_ratio,
        ),
        MetricReportRow(
            "manual_detail",
            "active_span_representative_hand_fingertip_coverage_ratio",
            metrics.manual_detail.active_span_representative_hand_fingertip_coverage_ratio,
        ),
        MetricReportRow(
            "manual_detail",
            "active_span_representative_hand_distal_chain_coverage_ratio",
            metrics.manual_detail.active_span_representative_hand_distal_chain_coverage_ratio,
        ),
        MetricReportRow(
            "manual_detail",
            "max_active_span_representative_hand_detail_dropout_run_ratio",
            metrics.manual_detail.max_active_span_representative_hand_detail_dropout_run_ratio,
        ),
        MetricReportRow(
            "non_manual_quality",
            "active_span_face_landmark_coverage_ratio",
            metrics.non_manual_quality.active_span_face_landmark_coverage_ratio,
        ),
        MetricReportRow(
            "non_manual_quality",
            "active_span_upper_face_landmark_coverage_ratio",
            metrics.non_manual_quality.active_span_upper_face_landmark_coverage_ratio,
        ),
        MetricReportRow(
            "non_manual_quality",
            "active_span_lower_face_landmark_coverage_ratio",
            metrics.non_manual_quality.active_span_lower_face_landmark_coverage_ratio,
        ),
        MetricReportRow(
            "non_manual_quality",
            "max_active_span_face_detail_dropout_run_ratio",
            metrics.non_manual_quality.max_active_span_face_detail_dropout_run_ratio,
        ),
        MetricReportRow(
            "non_manual_quality",
            "active_span_manual_face_overlap_ratio",
            metrics.non_manual_quality.active_span_manual_face_overlap_ratio,
        ),
        MetricReportRow(
            "geometry",
            "active_span_upper_body_bone_length_outlier_frame_ratio",
            metrics.geometry.active_span_upper_body_bone_length_outlier_frame_ratio,
        ),
        MetricReportRow(
            "geometry",
            "active_span_representative_hand_bone_length_outlier_frame_ratio",
            metrics.geometry.active_span_representative_hand_bone_length_outlier_frame_ratio,
        ),
        MetricReportRow(
            "geometry",
            "active_span_cross_channel_scale_outlier_frame_ratio",
            metrics.geometry.active_span_cross_channel_scale_outlier_frame_ratio,
        ),
        MetricReportRow("text", "token_count", metrics.text.token_count),
        MetricReportRow("text", "character_count", metrics.text.character_count),
        MetricReportRow(
            "text",
            "non_whitespace_character_count",
            metrics.text.non_whitespace_character_count,
        ),
        MetricReportRow("text", "frames_per_token", metrics.text.frames_per_token),
        MetricReportRow(
            "text",
            "frames_per_character",
            metrics.text.frames_per_character,
        ),
        MetricReportRow(
            "text",
            "frames_per_non_whitespace_character",
            metrics.text.frames_per_non_whitespace_character,
        ),
        MetricReportRow("length", "frame_count", metrics.length.frame_count),
        MetricReportRow("length", "duration_seconds", metrics.length.duration_seconds),
        MetricReportRow(
            "length",
            "frames_per_second_observed",
            metrics.length.frames_per_second_observed,
        ),
    )


def _gate_rows(gates: GateDecisionBundle | None) -> tuple[GateReportRow, ...]:
    if gates is None:
        return ()
    return tuple(
        GateReportRow(
            gate_name=decision.gate,
            status=decision.status,
            issue_count=len(decision.issue_codes),
        )
        for decision in gates.decisions
    )


__all__ = ["build_quality_report_tables"]
