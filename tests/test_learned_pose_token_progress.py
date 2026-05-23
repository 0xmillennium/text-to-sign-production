from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_learned_pose_token_provider_declares_expected_progress_tasks() -> None:
    provider = Path(
        "src/text_to_sign_production/modeling/candidates/learned_pose_token/provider.py"
    ).read_text(encoding="utf-8")
    trainer = Path(
        "src/text_to_sign_production/modeling/candidates/learned_pose_token/trainer.py"
    ).read_text(encoding="utf-8")
    source = provider + trainer

    for operation in (
        "materialize_train",
        "materialize_val",
        "tokenizer_train_epoch_",
        "text_to_token_train_epoch_",
        "predict_and_decode_token_samples",
        "write_decoded_intermediate_poses",
        "write_final_generated_pose",
    ):
        assert operation in source
