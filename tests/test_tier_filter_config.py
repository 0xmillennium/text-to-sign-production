from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
def test_semantic_tier_thresholds_are_expected() -> None:
    config = yaml.safe_load(Path("configs/data/filters.yaml").read_text(encoding="utf-8"))
    families = config["families"]

    assert families["oob"]["clean"]["max_out_of_bounds_ratio"] == 0.15
    assert (
        families["upper_body_support"]["clean"][
            "min_active_span_upper_body_support_landmark_coverage_ratio"
        ]
        == 0.85
    )
    assert (
        families["manual_visibility"]["clean"][
            "min_active_span_any_hand_available_frame_ratio"
        ]
        == 0.80
    )
    assert (
        families["manual_visibility"]["clean"][
            "max_active_span_any_hand_unavailable_run_ratio"
        ]
        == 0.25
    )
    assert (
        families["non_manual_visibility"]["clean"][
            "min_active_span_face_available_frame_ratio"
        ]
        == 0.85
    )
    assert (
        families["non_manual_visibility"]["clean"][
            "max_active_span_face_unavailable_run_ratio"
        ]
        == 0.20
    )
    assert (
        families["confidence"]["tight"][
            "min_active_span_any_hand_observed_mean_confidence"
        ]
        == 0.45
    )
    assert (
        families["kinematic_naturalness"]["clean"][
            "max_comparable_transition_discontinuity_ratio"
        ]
        == 0.03
    )
    assert families["tracking_quality"]["clean"]["max_tracked_target_missing_frame_ratio"] == 0.12
    assert (
        families["tracking_quality"]["clean"][
            "max_person_tracking_continuity_break_ratio"
        ]
        == 0.18
    )
    assert (
        families["tracking_quality"]["clean"]["max_person_tracking_reanchor_ratio"]
        == 0.18
    )
    assert (
        families["manual_detail"]["clean"][
            "min_active_span_representative_hand_landmark_coverage_ratio"
        ]
        == 0.80
    )
    assert (
        families["manual_detail"]["clean"][
            "min_active_span_representative_hand_fingertip_coverage_ratio"
        ]
        == 0.80
    )
    assert (
        families["manual_detail"]["clean"][
            "min_active_span_representative_hand_distal_chain_coverage_ratio"
        ]
        == 0.80
    )
    assert (
        families["manual_detail"]["clean"][
            "max_active_span_representative_hand_detail_dropout_run_ratio"
        ]
        == 0.25
    )
    assert (
        families["non_manual_quality"]["clean"][
            "min_active_span_face_landmark_coverage_ratio"
        ]
        == 0.85
    )
    assert (
        families["non_manual_quality"]["clean"][
            "min_active_span_upper_face_landmark_coverage_ratio"
        ]
        == 0.85
    )
    assert (
        families["non_manual_quality"]["clean"][
            "min_active_span_lower_face_landmark_coverage_ratio"
        ]
        == 0.85
    )
    assert (
        families["non_manual_quality"]["clean"][
            "max_active_span_face_detail_dropout_run_ratio"
        ]
        == 0.20
    )
    assert (
        families["non_manual_quality"]["clean"][
            "min_active_span_face_available_given_manual_frame_ratio"
        ]
        == 0.85
    )
    assert (
        families["geometry"]["clean"][
            "max_active_span_cross_channel_scale_outlier_frame_ratio"
        ]
        == 0.06
    )
    assert (
        families["geometry"]["tight"][
            "max_active_span_cross_channel_scale_outlier_frame_ratio"
        ]
        == 0.04
    )


@pytest.mark.unit
def test_filter_thresholds_are_monotonic_by_tier_semantics() -> None:
    config = yaml.safe_load(Path("configs/data/filters.yaml").read_text(encoding="utf-8"))

    for family_name, family in config["families"].items():
        loose = family["loose"]
        clean = family["clean"]
        tight = family["tight"]
        assert loose.keys() == clean.keys() == tight.keys(), family_name
        for metric_key in loose:
            loose_value = loose[metric_key]
            clean_value = clean[metric_key]
            tight_value = tight[metric_key]
            if metric_key.startswith("min_"):
                assert loose_value <= clean_value <= tight_value, (
                    family_name,
                    metric_key,
                    loose_value,
                    clean_value,
                    tight_value,
                )
            elif metric_key.startswith("max_"):
                assert loose_value >= clean_value >= tight_value, (
                    family_name,
                    metric_key,
                    loose_value,
                    clean_value,
                    tight_value,
                )
            else:
                pytest.fail(f"unreviewed threshold key pattern: {family_name}.{metric_key}")
