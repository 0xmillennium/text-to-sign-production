"""Export the fixed Sprint 3 qualitative validation panel for a baseline run."""

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
from text_to_sign_production.modeling.inference.qualitative import (  # noqa: E402
    DEFAULT_QUALITATIVE_PANEL_SIZE as _DEFAULT_QUALITATIVE_PANEL_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG_PATH,
        help="Path to a Sprint 3 baseline training YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to export from. Defaults to checkpoint.output_dir/best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where qualitative panel artifacts should be written.",
    )
    parser.add_argument(
        "--panel-definition",
        type=Path,
        default=None,
        help="Optional reusable qualitative panel definition JSON file.",
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=None,
        help=(
            "Optional Phase 5 run_summary.json path. "
            "Defaults to checkpoint.output_dir/run_summary.json."
        ),
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=_DEFAULT_QUALITATIVE_PANEL_SIZE,
        help="Number of validation samples to select when generating a panel definition.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_qualitative_export(
            args.config,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            panel_definition_path=args.panel_definition,
            run_summary_path=args.run_summary,
            panel_size=args.panel_size,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Panel definition: {result.panel_definition_path}")
    print(f"Panel records: {result.records_path}")
    print(f"Panel summary: {result.panel_summary_path}")
    print(f"Evidence bundle: {result.evidence_bundle_path}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Sample count: {result.sample_count}")
    return 0


def run_qualitative_export(
    config_path: Path,
    *,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    panel_definition_path: Path | None = None,
    run_summary_path: Path | None = None,
    panel_size: int = _DEFAULT_QUALITATIVE_PANEL_SIZE,
) -> Any:
    """Import the exporter lazily so CLI import stays dependency-light."""

    from text_to_sign_production.modeling.inference.qualitative import (
        export_qualitative_panel,
    )

    return export_qualitative_panel(
        config_path,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        panel_definition_path=panel_definition_path,
        run_summary_path=run_summary_path,
        panel_size=panel_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
