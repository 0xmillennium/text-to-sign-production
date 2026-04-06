"""Validate a raw, normalized, or processed JSONL manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.jsonl import write_json  # noqa: E402
from text_to_sign_production.data.validate import validate_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to a JSONL manifest.")
    parser.add_argument(
        "--kind",
        required=True,
        choices=("raw", "normalized", "processed"),
        help="Manifest kind to validate.",
    )
    parser.add_argument(
        "--report-path",
        help="Optional JSON output path for machine-readable validation issues.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues = validate_manifest(path=Path(args.manifest), kind=args.kind)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]

    print(f"validated {args.kind} manifest: {args.manifest}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    for issue in issues[:20]:
        print(f"- {issue.severity}: {issue.code}: {issue.message}")

    if args.report_path:
        write_json(Path(args.report_path), [issue.to_record() for issue in issues])

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
