from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_text_to_token_progress_has_known_epoch_batch_total() -> None:
    source = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/learned_pose_token/provider.py"
    ).read_text(encoding="utf-8")

    assert "text_to_token_train_epoch" in source
    assert "total=len(train_batches)" in source
