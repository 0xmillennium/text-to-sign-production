"""Upper-body/signing-relevant support metrics."""

from __future__ import annotations

from typing import Final

import numpy as np

from text_to_sign_production.data.metrics.types import UpperBodySupportMetrics
from text_to_sign_production.data.samples.types import ProcessedSamplePayload

UPPER_BODY_SUPPORT_LANDMARKS: Final[dict[int, str]] = {
    0: "nose",
    1: "neck",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
    8: "mid_hip_torso_anchor",
}
UPPER_BODY_SUPPORT_LANDMARK_INDICES: Final[tuple[int, ...]] = tuple(
    UPPER_BODY_SUPPORT_LANDMARKS
)


def compute_upper_body_support_metrics(payload: ProcessedSamplePayload) -> UpperBodySupportMetrics:
    """Compute binding support over explicit upper-body/signing-relevant landmarks."""
    body_conf = np.asarray(payload.pose.body.confidence)
    if body_conf.size == 0:
        raise ValueError("Cannot compute upper-body support for empty body confidence.")
    return UpperBodySupportMetrics(
        upper_body_support_landmark_coverage_ratio=float(
            np.count_nonzero(body_conf[:, UPPER_BODY_SUPPORT_LANDMARK_INDICES] > 0.0)
            / body_conf[:, UPPER_BODY_SUPPORT_LANDMARK_INDICES].size
        )
    )
