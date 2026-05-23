from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_articulator_aware_provider_declares_expected_progress_tasks() -> None:
    provider = Path(
        "src/text_to_sign_production/modeling/candidates/articulator_aware/provider.py"
    ).read_text(encoding="utf-8")
    trainer = Path(
        "src/text_to_sign_production/modeling/candidates/articulator_aware/trainer.py"
    ).read_text(encoding="utf-8")
    source = provider + trainer

    for operation in (
        "materialize_train_sources",
        "materialize_validation_sources",
        "build_frame_training_samples",
        "train_epoch_",
        "evaluate_train_sources",
        "evaluate_validation_sources",
        "generate_validation_poses",
        "write_final_generated_pose",
    ):
        assert operation in source
