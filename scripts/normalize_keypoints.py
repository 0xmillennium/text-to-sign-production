"""Normalize OpenPose keypoints and export `.npz` sample files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import SPLITS  # noqa: E402
from text_to_sign_production.data.normalize import normalize_all_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Official How2Sign splits to normalize.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    normalize_all_splits(splits=tuple(args.splits))
    for split in args.splits:
        print(f"normalized: {split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
