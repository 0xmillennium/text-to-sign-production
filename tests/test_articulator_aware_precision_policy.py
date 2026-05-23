from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_articulator_training_paths_use_shared_autocast_context() -> None:
    trainer_source = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/articulator_aware/trainer.py"
    ).read_text(encoding="utf-8")

    assert "with autocast_context(precision_policy):" in trainer_source
    assert "with autocast_context(resolved_precision):" in trainer_source
    assert "precision_policy_name: str = \"auto\"" in trainer_source


def test_articulator_provider_reports_precision_metadata() -> None:
    provider_source = (
        ROOT
        / "src/text_to_sign_production/modeling/candidates/articulator_aware/provider.py"
    ).read_text(encoding="utf-8")

    assert "precision_policy_name=_compute_precision_policy(context)" in provider_source
    assert "**_precision_metadata(context, config, precision_applied=True)" in provider_source
    assert "generation_precision_applied" in provider_source
