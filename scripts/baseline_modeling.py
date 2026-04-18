"""Run the Sprint 3 baseline-modeling stage workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH  # noqa: E402
from text_to_sign_production.modeling.inference.qualitative import (  # noqa: E402
    DEFAULT_QUALITATIVE_PANEL_SIZE,
)
from text_to_sign_production.workflows.baseline_modeling import (  # noqa: E402
    DEFAULT_BASELINE_RUN_NAME,
    run_baseline_modeling,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        nargs="?",
        choices=("prepare", "train", "export-panel", "package", "all"),
        default="all",
        help="Baseline-modeling workflow step to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG_PATH,
        help="Path to a Sprint 3 baseline training YAML config.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_BASELINE_RUN_NAME,
        help="Stable run directory name under the baseline-modeling artifact root.",
    )
    parser.add_argument(
        "--artifact-runs-root",
        type=Path,
        default=None,
        help=(
            "Directory that contains baseline-modeling run directories. "
            "Defaults to outputs/modeling/baseline-modeling/runs for local CLI use."
        ),
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=DEFAULT_QUALITATIVE_PANEL_SIZE,
        help="Number of validation samples to export in the qualitative panel.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_baseline_modeling(
            step=args.step,
            config_path=args.config,
            run_name=args.run_name,
            artifact_runs_root=args.artifact_runs_root,
            panel_size=args.panel_size,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Baseline Modeling run: {result.run_name}")
    print(f"Run root: {result.layout.run_root}")
    if not result.steps:
        print("steps: none")
    for step_result in result.steps:
        print(f"{step_result.step}: {step_result.action}")
        print(f"{step_result.step} output: {step_result.output_path}")
        if step_result.archive_path is not None:
            print(f"{step_result.step} archive: {step_result.archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
