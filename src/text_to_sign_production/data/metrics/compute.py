"""Metric bundle composition entry point."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.metrics.confidence import compute_confidence_metrics
from text_to_sign_production.data.metrics.face import compute_face_metrics
from text_to_sign_production.data.metrics.hand import compute_hand_metrics
from text_to_sign_production.data.metrics.length import compute_length_metrics
from text_to_sign_production.data.metrics.oob import compute_oob_metrics
from text_to_sign_production.data.metrics.text import compute_text_metrics
from text_to_sign_production.data.metrics.types import MetricBundle
from text_to_sign_production.data.metrics.valid import compute_valid_metrics
from text_to_sign_production.data.samples.schema import CANONICAL_POSE_CHANNELS
from text_to_sign_production.data.samples.types import PassedManifestEntry, ProcessedSamplePayload


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
    if payload.fps != manifest.fps:
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
        manifest_nonzero = manifest.frame_quality.channel_nonzero_frames.get(channel, 0)

        if payload_nonzero != manifest_nonzero:
            raise ValueError(
                f"Channel '{channel}' nonzero frame count mismatch: "
                f"payload={payload_nonzero}, manifest={manifest_nonzero}"
            )

    oob = compute_oob_metrics(payload, manifest)
    hand = compute_hand_metrics(payload, manifest)
    face = compute_face_metrics(payload, manifest)
    valid = compute_valid_metrics(payload, manifest)
    confidence = compute_confidence_metrics(payload)
    text = compute_text_metrics(payload)
    length = compute_length_metrics(payload, text)

    return MetricBundle(
        sample_id=payload.sample_id,
        split=payload.split,
        oob=oob,
        hand=hand,
        face=face,
        valid=valid,
        confidence=confidence,
        text=text,
        length=length,
    )
