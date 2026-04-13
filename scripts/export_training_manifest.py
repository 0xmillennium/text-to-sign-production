"""Export final training manifests and Markdown reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import SPLITS  # noqa: E402
from text_to_sign_production.data.manifests import export_final_manifests  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assumption-report",
        default=str(PROJECT_ROOT / "data/interim/reports/assumption-report.json"),
        help="Machine-readable assumption report from prepare_raw.",
    )
    parser.add_argument(
        "--filter-report",
        default=str(PROJECT_ROOT / "data/interim/reports/filter-report.json"),
        help="Machine-readable filter report from filter_samples.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Official How2Sign splits to export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assumption_report = json.loads(Path(args.assumption_report).read_text(encoding="utf-8"))
    filter_report = json.loads(Path(args.filter_report).read_text(encoding="utf-8"))
    export_final_manifests(
        assumption_report=assumption_report,
        filter_report=filter_report,
        splits=tuple(args.splits),
    )
    print("processed manifests: data/processed/v1/manifests/")
    print("processed reports: data/processed/v1/reports/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
