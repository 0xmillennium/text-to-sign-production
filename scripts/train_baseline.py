"""Run the Sprint 3 baseline training loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG_PATH,
        help="Path to a Sprint 3 baseline training YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional checkpoint output directory override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_baseline_training(
        args.config,
        checkpoint_output_dir=args.output_dir,
    )
    print(f"Run summary: {result.summary_path}")
    print(f"Last checkpoint: {result.last_checkpoint_path}")
    if result.best_checkpoint_path is None:
        print("Best checkpoint: none")
    else:
        print(f"Best checkpoint: {result.best_checkpoint_path}")
    return 0


def run_baseline_training(
    config_path: Path,
    *,
    checkpoint_output_dir: Path | None = None,
) -> Any:
    """Import the training runner lazily so CLI import stays dependency-light."""

    from text_to_sign_production.modeling.training.train import (
        run_baseline_training as training_runner,
    )

    return training_runner(
        config_path,
        checkpoint_output_dir=checkpoint_output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
