"""Metric bundle composition entry point."""

from __future__ import annotations

import math

import numpy as np

from text_to_sign_production.legacy_data.metrics.coherence import (
    compute_temporal_coherence_metrics,
)
from text_to_sign_production.legacy_data.metrics.confidence import compute_confidence_metrics
from text_to_sign_production.legacy_data.metrics.coverage import compute_coverage_metrics
from text_to_sign_production.legacy_data.metrics.face import compute_face_metrics
from text_to_sign_production.legacy_data.metrics.geometry import compute_geometry_metrics
from text_to_sign_production.legacy_data.metrics.hand import compute_hand_metrics
from text_to_sign_production.legacy_data.metrics.length import compute_length_metrics
from text_to_sign_production.legacy_data.metrics.manual_detail import compute_manual_detail_metrics
from text_to_sign_production.legacy_data.metrics.non_manual_quality import (
    compute_non_manual_quality_metrics,
)
from text_to_sign_production.legacy_data.metrics.oob import compute_oob_metrics
from text_to_sign_production.legacy_data.metrics.span import (
    compute_active_signing_span_metrics,
)
from text_to_sign_production.legacy_data.metrics.support import (
    compute_upper_body_support_metrics,
)
from text_to_sign_production.legacy_data.metrics.text import compute_text_metrics
from text_to_sign_production.legacy_data.metrics.tracking import compute_tracking_quality_metrics
from text_to_sign_production.legacy_data.metrics.types import MetricBundle
from text_to_sign_production.legacy_data.metrics.valid import compute_valid_metrics
from text_to_sign_production.legacy_data.pose.schema import CANONICAL_POSE_CHANNELS
from text_to_sign_production.legacy_data.samples.types import (
    PassedManifestEntry,
    ProcessedSamplePayload,
)


def build_metric_bundle(
    payload: ProcessedSamplePayload, manifest: PassedManifestEntry
) -> MetricBundle:
    """Compute and compose all metrics for a sample."""
    if payload.num_frames <= 0:
        raise ValueError(
            f"num_frames must be > 0 for a valid passed sample, got {payload.num_frames}"
        )

    if payload.sample_id != manifest.sample_id:
        raise ValueError(f"sample_id mismatch: {payload.sample_id} != {manifest.sample_id}")
    if payload.split != manifest.split:
        raise ValueError(f"split mismatch: {payload.split} != {manifest.split}")
    if payload.text != manifest.text:
        raise ValueError(f"text mismatch: {payload.text} != {manifest.text}")
    if payload.num_frames != manifest.num_frames:
        raise ValueError(f"num_frames mismatch: {payload.num_frames} != {manifest.num_frames}")
    if not _fps_values_match(payload.fps, manifest.fps):
        raise ValueError(f"fps mismatch: {payload.fps} != {manifest.fps}")

    channels = {
        "body": payload.pose.body,
        "left_hand": payload.pose.left_hand,
        "right_hand": payload.pose.right_hand,
        "face": payload.pose.face,
    }

    for channel in CANONICAL_POSE_CHANNELS:
        payload_conf = np.asarray(channels[channel].confidence)
        payload_nonzero = int(np.count_nonzero(np.any(payload_conf > 0.0, axis=1)))
        if channel not in manifest.frame_quality.channel_nonzero_frames:
            raise ValueError(
                f"Manifest frame_quality.channel_nonzero_frames missing canonical channel "
                f"{channel!r}"
            )
        manifest_nonzero = manifest.frame_quality.channel_nonzero_frames[channel]

        if payload_nonzero != manifest_nonzero:
            raise ValueError(
                f"Channel '{channel}' nonzero frame count mismatch: "
                f"payload={payload_nonzero}, manifest={manifest_nonzero}"
            )

    active_signing_span = compute_active_signing_span_metrics(payload)
    oob = compute_oob_metrics(payload, manifest)
    upper_body_support = compute_upper_body_support_metrics(payload, active_signing_span)
    coverage = compute_coverage_metrics(payload, manifest)
    manual_visibility = compute_hand_metrics(payload, manifest, active_signing_span)
    non_manual_visibility = compute_face_metrics(payload, manifest, active_signing_span)
    valid = compute_valid_metrics(payload, manifest)
    confidence = compute_confidence_metrics(payload, active_signing_span)
    kinematic_naturalness = compute_temporal_coherence_metrics(payload, active_signing_span)
    tracking_quality = compute_tracking_quality_metrics(manifest)
    manual_detail = compute_manual_detail_metrics(payload, active_signing_span)
    non_manual_quality = compute_non_manual_quality_metrics(payload, active_signing_span)
    geometry = compute_geometry_metrics(payload, active_signing_span)
    text = compute_text_metrics(payload)
    length = compute_length_metrics(payload, text)

    return MetricBundle(
        sample_id=payload.sample_id,
        split=payload.split,
        active_signing_span=active_signing_span,
        oob=oob,
        upper_body_support=upper_body_support,
        coverage=coverage,
        manual_visibility=manual_visibility,
        non_manual_visibility=non_manual_visibility,
        valid=valid,
        confidence=confidence,
        kinematic_naturalness=kinematic_naturalness,
        tracking_quality=tracking_quality,
        manual_detail=manual_detail,
        non_manual_quality=non_manual_quality,
        geometry=geometry,
        text=text,
        length=length,
    )


def _fps_values_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
