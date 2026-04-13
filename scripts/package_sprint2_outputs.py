"""Archive Sprint 2 manifests, reports, and processed samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import ARCHIVES_RELATIVE_ROOT, SPLITS  # noqa: E402
from text_to_sign_production.ops.archive_ops import tar_supports_zstd  # noqa: E402
from text_to_sign_production.ops.colab_workflow import package_sprint2_outputs  # noqa: E402

DEFAULT_OUTPUT_DIR = ARCHIVES_RELATIVE_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Processed sample splits to archive individually.",
    )
    return parser.parse_args()


def _format_size(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _display_path(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _tar_supports_zstd() -> bool:
    return tar_supports_zstd()


def package_outputs(
    *,
    project_root: Path = PROJECT_ROOT,
    splits: tuple[str, ...] = SPLITS,
) -> list[Path]:
    return package_sprint2_outputs(
        project_root=project_root,
        splits=splits,
    )


def main() -> int:
    args = parse_args()
    archives = package_outputs(splits=tuple(args.splits))
    for archive_path in archives:
        print(
            f"created: {_display_path(archive_path, project_root=PROJECT_ROOT)} "
            f"({_format_size(archive_path.stat().st_size)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
