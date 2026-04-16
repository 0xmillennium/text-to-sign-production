"""Run the Dataset Build stage from the terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import DEFAULT_FILTER_CONFIG_PATH, SPLITS  # noqa: E402
from text_to_sign_production.workflows.dataset_build import (  # noqa: E402
    DatasetBuildOutputMode,
    run_dataset_build,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Official How2Sign splits to build.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_FILTER_CONFIG_PATH),
        help="YAML filtering policy.",
    )
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Build manifests and samples without creating local output archives.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_mode: DatasetBuildOutputMode = "none" if args.no_package else "local_archives"
    result = run_dataset_build(
        splits=tuple(args.splits),
        filter_config_path=Path(args.config),
        output_mode=output_mode,
    )

    print(f"Dataset Build completed for splits: {', '.join(result.splits)}")
    for split in result.splits:
        raw_report = result.assumption_report["splits"][split]
        filter_report = result.filter_report["splits"][split]
        print(
            f"{split}: rows={raw_report['translation_row_count']} "
            f"matched={raw_report['matched_sample_count']} "
            f"kept={filter_report['kept_samples']} "
            f"dropped={filter_report['dropped_samples']}"
        )

    if result.output_paths:
        for path in result.output_paths:
            print(f"output: {path}")
    else:
        print("output packaging: skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
