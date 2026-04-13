"""Package Sprint 2 outputs and publish the archives to the fixed Drive artifact path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import SPLITS  # noqa: E402
from text_to_sign_production.ops.colab_workflow import publish_colab_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Official How2Sign splits whose archives should be published.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    published_paths = publish_colab_outputs(splits=tuple(args.splits))
    for path in published_paths:
        print(f"published: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
