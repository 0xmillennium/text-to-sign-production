import torch

from text_to_sign_production.modeling.training.precision import (
    autocast_context,
    resolve_precision_policy,
)


def test_auto_resolves_to_fp32_on_cpu() -> None:
    policy = resolve_precision_policy("auto", torch.device("cpu"))
    assert policy.precision_policy == "fp32"
    assert policy.resolved_policy == "fp32"
    assert policy.autocast_enabled is False


def test_bf16_policy_is_disabled_on_cpu() -> None:
    policy = resolve_precision_policy("bf16", torch.device("cpu"))
    assert policy.precision_policy == "fp32"
    assert policy.requested_policy == "bf16"
    assert policy.autocast_enabled is False
    assert policy.to_metadata()["precision_applied"] is False


def test_autocast_context_is_noop_on_cpu() -> None:
    policy = resolve_precision_policy("auto", torch.device("cpu"))
    with autocast_context(policy):
        value = torch.tensor([1.0])
    assert value.dtype == torch.float32


def test_precision_metadata_distinguishes_requested_resolved_and_applied() -> None:
    policy = resolve_precision_policy("bf16", torch.device("cpu"))
    metadata = policy.to_metadata(precision_applied=True)

    assert metadata["requested_precision_policy"] == "bf16"
    assert metadata["resolved_precision_policy"] == "fp32"
    assert metadata["autocast_enabled"] is False
    assert metadata["autocast_dtype"] is None
    assert metadata["precision_applied"] is False
