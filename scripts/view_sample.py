"""Inspect a processed sample and its manifest metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import PROCESSED_MANIFESTS_ROOT, SPLITS  # noqa: E402
from text_to_sign_production.data.jsonl import iter_jsonl  # noqa: E402
from text_to_sign_production.data.validate import validate_processed_sample_path  # noqa: E402


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
        record_sample_id = record.get("sample_id")
        if record_sample_id is None:
            raise ValueError(
                f"Processed manifest record is missing 'sample_id' in {manifest_path}."
            )
        if str(record_sample_id) == sample_id:
            return record
    raise KeyError(f"Sample {sample_id} not found in {manifest_path}.")


def main() -> int:
    args = parse_args()
    try:
        record = load_record(args.split, args.sample_id)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))

    try:
        sample_path_value = record["sample_path"]
        record_split = str(record["split"])
        record_sample_id = str(record["sample_id"])
    except KeyError as exc:
        print(
            f"error: Processed manifest record is missing required field {exc!s}.",
            file=sys.stderr,
        )
        return 1

    try:
        sample_path = validate_processed_sample_path(
            sample_path_value,
            split=record_split,
            sample_id=record_sample_id,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        with np.load(sample_path, allow_pickle=False) as sample:
            print("npz keys:", sorted(sample.files))
            for key in sorted(sample.files):
                array = sample[key]
                shape = getattr(array, "shape", ())
                dtype = getattr(array, "dtype", None)
                print(f"{key}: shape={shape} dtype={dtype}")
    except (BadZipFile, EOFError, OSError, ValueError) as exc:
        print(f"error: Processed sample file could not be read as .npz: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
