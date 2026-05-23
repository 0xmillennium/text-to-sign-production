from __future__ import annotations

from text_to_sign_production.modeling.data_surfaces.telemetry import (
    cache_read_dominates_runtime,
)


def test_cache_read_dominance_blocks_readiness_when_over_half_stage_time() -> None:
    assert cache_read_dominates_runtime(cache_read_seconds=6.0, stage_elapsed_seconds=10.0)
    assert not cache_read_dominates_runtime(cache_read_seconds=4.0, stage_elapsed_seconds=10.0)
