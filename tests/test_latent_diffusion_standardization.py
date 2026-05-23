from test_bfh_standardization_sparse_fallback import _pose

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_standardization import (
    compute_bfh_standardization_stats,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    _standardization_metadata,
)


def test_latent_diffusion_sparse_stats_expose_metadata() -> None:
    stats = compute_bfh_standardization_stats(
        [_pose((PoseChannel.BODY,))],
        missing_observation_policy="channel_fallback",
    )
    metadata = _standardization_metadata(stats)
    assert metadata["standardization_missing_observation_policy"] == "channel_fallback"
    assert metadata["standardization_zero_observation_coordinate_count"] > 0
    assert "standardization_channel_fallback_coordinate_count" in metadata
    assert "standardization_global_fallback_coordinate_count" in metadata
