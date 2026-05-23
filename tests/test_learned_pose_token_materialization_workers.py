from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_materialization_worker_not_applied_metadata_is_removed() -> None:
    source = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/learned_pose_token/provider.py"
    ).read_text(encoding="utf-8")

    assert "materialization_workers_requested" not in source
    assert "materialization_workers_applied" not in source
    assert "materialization_workers_reason" not in source
