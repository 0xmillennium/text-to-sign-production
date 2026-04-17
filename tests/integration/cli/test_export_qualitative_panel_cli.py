"""Qualitative-panel export CLI wrapper tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.export_qualitative_panel as export_qualitative_panel_script
from text_to_sign_production.modeling.inference.qualitative import (
    DEFAULT_QUALITATIVE_PANEL_SIZE,
)

pytestmark = pytest.mark.integration


def test_export_qualitative_panel_cli_calls_export_runner(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "baseline.yaml"
    checkpoint_path = tmp_path / "best.pt"
    output_dir = tmp_path / "exports"
    panel_definition_path = tmp_path / "panel.json"
    run_summary_path = tmp_path / "run_summary.json"
    seen_args: dict[str, object] = {}

    def fake_run_qualitative_export(
        config: Path,
        *,
        output_dir: Path,
        checkpoint_path: Path | None = None,
        panel_definition_path: Path | None = None,
        run_summary_path: Path | None = None,
        panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE,
    ) -> SimpleNamespace:
        seen_args.update(
            {
                "config": config,
                "output_dir": output_dir,
                "checkpoint_path": checkpoint_path,
                "panel_definition_path": panel_definition_path,
                "run_summary_path": run_summary_path,
                "panel_size": panel_size,
            }
        )
        return SimpleNamespace(
            panel_definition_path=output_dir / "panel_definition.json",
            records_path=output_dir / "records.jsonl",
            panel_summary_path=output_dir / "panel_summary.json",
            evidence_bundle_path=output_dir / "baseline_evidence_bundle.json",
            checkpoint_path=checkpoint_path,
            sample_count=3,
        )

    monkeypatch.setattr(
        export_qualitative_panel_script,
        "run_qualitative_export",
        fake_run_qualitative_export,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_qualitative_panel.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--panel-definition",
            str(panel_definition_path),
            "--run-summary",
            str(run_summary_path),
            "--panel-size",
            "3",
        ],
    )

    assert export_qualitative_panel_script.main() == 0
    captured = capsys.readouterr()
    assert seen_args == {
        "config": config_path,
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "panel_definition_path": panel_definition_path,
        "run_summary_path": run_summary_path,
        "panel_size": 3,
    }
    assert f"Panel definition: {output_dir / 'panel_definition.json'}" in captured.out
    assert "Sample count: 3" in captured.out


def test_export_qualitative_panel_cli_uses_shared_default_panel_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_panel_size: list[int] = []

    def fake_run_qualitative_export(
        config: Path,
        *,
        output_dir: Path,
        checkpoint_path: Path | None = None,
        panel_definition_path: Path | None = None,
        run_summary_path: Path | None = None,
        panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE,
    ) -> SimpleNamespace:
        seen_panel_size.append(panel_size)
        return SimpleNamespace(
            panel_definition_path=output_dir / "panel_definition.json",
            records_path=output_dir / "records.jsonl",
            panel_summary_path=output_dir / "panel_summary.json",
            evidence_bundle_path=output_dir / "baseline_evidence_bundle.json",
            checkpoint_path=None,
            sample_count=1,
        )

    monkeypatch.setattr(
        export_qualitative_panel_script,
        "run_qualitative_export",
        fake_run_qualitative_export,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_qualitative_panel.py",
            "--config",
            str(tmp_path / "baseline.yaml"),
            "--output-dir",
            str(tmp_path / "exports"),
        ],
    )

    assert export_qualitative_panel_script.main() == 0
    assert seen_panel_size == [DEFAULT_QUALITATIVE_PANEL_SIZE]


def test_export_qualitative_panel_cli_reports_direct_errors(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_qualitative_export(
        config: Path,
        *,
        output_dir: Path,
        checkpoint_path: Path | None = None,
        panel_definition_path: Path | None = None,
        run_summary_path: Path | None = None,
        panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE,
    ) -> None:
        raise FileNotFoundError("Baseline checkpoint does not exist: missing.pt")

    monkeypatch.setattr(
        export_qualitative_panel_script,
        "run_qualitative_export",
        fake_run_qualitative_export,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_qualitative_panel.py",
            "--config",
            str(tmp_path / "baseline.yaml"),
            "--output-dir",
            str(tmp_path / "exports"),
        ],
    )

    assert export_qualitative_panel_script.main() == 1
    captured = capsys.readouterr()
    assert "error: Baseline checkpoint does not exist: missing.pt" in captured.err
