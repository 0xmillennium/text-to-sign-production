"""Smoke tests for the Sprint 3 modeling scaffold."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_modeling_scaffold_packages_are_importable() -> None:
    package_names = (
        "text_to_sign_production.modeling",
        "text_to_sign_production.modeling.backbones",
        "text_to_sign_production.modeling.config",
        "text_to_sign_production.modeling.data",
        "text_to_sign_production.modeling.inference",
        "text_to_sign_production.modeling.models",
        "text_to_sign_production.modeling.training",
    )

    for package_name in package_names:
        package = import_module(package_name)
        assert package.__name__ == package_name


def test_later_baseline_placeholder_scripts_fail_cleanly(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate_baseline as evaluate_baseline_script
    import scripts.export_qualitative_panel as export_qualitative_panel_script

    script_modules = (
        evaluate_baseline_script,
        export_qualitative_panel_script,
    )

    for script_module in script_modules:
        monkeypatch.setattr(sys, "argv", [f"{script_module.__name__}.py"])
        assert script_module.main() == 1
        captured = capsys.readouterr()
        assert "not implemented yet" in captured.err
        assert "Phase 1 placeholder" in captured.err


def test_train_baseline_script_calls_training_runner(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import scripts.train_baseline as train_baseline_script

    config_path = tmp_path / "baseline.yaml"
    output_dir = tmp_path / "output"
    seen_args: dict[str, object] = {}

    def fake_run_baseline_training(
        config: Path,
        *,
        checkpoint_output_dir: Path | None = None,
    ) -> SimpleNamespace:
        seen_args["config"] = config
        seen_args["checkpoint_output_dir"] = checkpoint_output_dir
        return SimpleNamespace(
            summary_path=output_dir / "run_summary.json",
            last_checkpoint_path=output_dir / "last.pt",
            best_checkpoint_path=output_dir / "best.pt",
            final_train_loss=1.0,
            final_validation_loss=2.0,
            final_validation_metric=3.0,
            best_validation_loss=2.0,
            best_epoch=1,
        )

    monkeypatch.setattr(
        train_baseline_script,
        "run_baseline_training",
        fake_run_baseline_training,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_baseline.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert train_baseline_script.main() == 0
    captured = capsys.readouterr()
    assert seen_args == {
        "config": config_path,
        "checkpoint_output_dir": output_dir,
    }
    assert f"Run summary: {output_dir / 'run_summary.json'}" in captured.out
    assert f"Last checkpoint: {output_dir / 'last.pt'}" in captured.out
    assert f"Best checkpoint: {output_dir / 'best.pt'}" in captured.out
