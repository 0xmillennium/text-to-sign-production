"""Controlled metric identifiers for validation-split pose comparison."""

from __future__ import annotations

from enum import StrEnum


class ValidationMetricKey(StrEnum):
    MASKED_L1_MEAN = "masked_l1_mean"
    MASKED_L2_MEAN = "masked_l2_mean"
    VELOCITY_L1_MEAN = "velocity_l1_mean"
    VELOCITY_L2_MEAN = "velocity_l2_mean"
    SEQUENCE_LENGTH_ABSOLUTE_ERROR = "sequence_length_absolute_error"
    VALID_JOINT_COVERAGE = "valid_joint_coverage"


class ValidationChannelMetricKey(StrEnum):
    CHANNEL_MASKED_L1_MEAN = "channel_masked_l1_mean"
    CHANNEL_MASKED_L2_MEAN = "channel_masked_l2_mean"
    CHANNEL_VELOCITY_L1_MEAN = "channel_velocity_l1_mean"
    CHANNEL_VELOCITY_L2_MEAN = "channel_velocity_l2_mean"
    CHANNEL_VALID_JOINT_COVERAGE = "channel_valid_joint_coverage"


__all__ = ["ValidationChannelMetricKey", "ValidationMetricKey"]
