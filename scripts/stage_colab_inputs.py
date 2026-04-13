"""Stage the fixed Sprint 2 Colab inputs from Google Drive into canonical raw layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import SPLITS  # noqa: E402
from text_to_sign_production.ops.colab_workflow import stage_colab_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Official How2Sign splits to stage from the fixed Drive layout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    staged = stage_colab_inputs(splits=tuple(args.splits))
    for item in staged:
        print(
            f"{item['split']}: translation={item['translation_path']} "
            f"keypoints={item['keypoints_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
