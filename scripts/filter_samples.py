"""Apply structural filtering to normalized candidate manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import SPLITS  # noqa: E402
from text_to_sign_production.data.filtering import filter_all_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data/filter-v1.yaml",
        help="YAML filtering policy.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Official How2Sign splits to filter.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = filter_all_splits(Path(args.config), splits=tuple(args.splits))
    for split in args.splits:
        split_report = report["splits"][split]
        print(
            f"{split}: kept={split_report['kept_samples']} "
            f"dropped={split_report['dropped_samples']}"
        )
    print("filter-report: data/interim/reports/filter-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
