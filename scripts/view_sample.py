"""Inspect a processed sample and its manifest metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import PROCESSED_MANIFESTS_ROOT, SPLITS  # noqa: E402
from text_to_sign_production.data.jsonl import iter_jsonl  # noqa: E402
from text_to_sign_production.data.utils import resolve_processed_sample_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="train",
        help="Official split whose processed manifest should be inspected.",
    )
    parser.add_argument("--sample-id", required=True, help="Canonical sample identifier.")
    return parser.parse_args()


def load_record(split: str, sample_id: str) -> dict[str, Any]:
    manifest_path = PROCESSED_MANIFESTS_ROOT / f"{split}.jsonl"
    for record in iter_jsonl(manifest_path):
        if record["sample_id"] == sample_id:
            return record
    raise KeyError(f"Sample {sample_id} not found in {manifest_path}.")


def main() -> int:
    args = parse_args()
    record = load_record(args.split, args.sample_id)
    print(json.dumps(record, indent=2, sort_keys=True))

    try:
        sample_path = resolve_processed_sample_path(record["sample_path"])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with np.load(sample_path) as sample:
        print("npz keys:", sorted(sample.files))
        for key in sorted(sample.files):
            array = sample[key]
            shape = getattr(array, "shape", ())
            dtype = getattr(array, "dtype", None)
            print(f"{key}: shape={shape} dtype={dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
