"""Build raw split manifests and assumption reports for How2Sign Sprint 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import SPLITS  # noqa: E402
from text_to_sign_production.data.raw import build_raw_manifests  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Official How2Sign splits to ingest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_raw_manifests(splits=tuple(args.splits))
    for split in args.splits:
        split_report = report["splits"][split]
        print(
            f"{split}: rows={split_report['translation_row_count']} "
            f"matched={split_report['matched_sample_count']} "
            f"unmatched={split_report['unmatched_sample_count']}"
        )
    print("assumption-report: data/interim/reports/assumption-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
