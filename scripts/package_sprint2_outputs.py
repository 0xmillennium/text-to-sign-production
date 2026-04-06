"""Archive Sprint 2 manifests, reports, and processed samples for transfer or storage."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from text_to_sign_production.data.constants import SPLITS  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("data/archives")
MANIFESTS_AND_REPORTS_MEMBERS = (
    Path("data/interim/raw_manifests"),
    Path("data/interim/normalized_manifests"),
    Path("data/interim/filtered_manifests"),
    Path("data/interim/reports"),
    Path("data/processed/v1/manifests"),
    Path("data/processed/v1/reports"),
)
MANIFESTS_AND_REPORTS_ARCHIVE_NAME = "sprint2_manifests_reports.tar.zst"
SAMPLE_ARCHIVE_NAME_TEMPLATE = "sprint2_samples_{split}.tar.zst"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory that will receive the generated `.tar.zst` archives.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Processed sample splits to archive individually.",
    )
    return parser.parse_args()


def _resolve_output_dir(output_dir: str, *, project_root: Path) -> Path:
    candidate = Path(output_dir)
    return candidate if candidate.is_absolute() else project_root / candidate


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


def _assert_archive_prerequisites() -> None:
    if shutil.which("tar") is None:
        raise RuntimeError("`tar` is required to package Sprint 2 outputs.")
    if shutil.which("zstd") is None:
        raise RuntimeError("`zstd` is required to create `.tar.zst` archives.")
    if not _tar_supports_zstd():
        raise RuntimeError("`tar` with `--zstd` support is required to package Sprint 2 outputs.")


def _tar_supports_zstd() -> bool:
    tar_path = shutil.which("tar")
    if tar_path is None:
        return False
    help_result = subprocess.run(
        [tar_path, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "--zstd" in (help_result.stdout + help_result.stderr)


def _assert_required_members(project_root: Path, members: tuple[Path, ...]) -> None:
    missing = [member for member in members if not (project_root / member).exists()]
    if missing:
        formatted = "\n".join(f"- {member}" for member in missing)
        raise FileNotFoundError(f"Missing required Sprint 2 outputs:\n{formatted}")


def _create_archive(
    *,
    archive_path: Path,
    members: tuple[Path, ...],
    project_root: Path,
) -> Path:
    _assert_required_members(project_root, members)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    command = [
        "tar",
        "--zstd",
        "-cf",
        str(archive_path),
        *[str(member) for member in members],
    ]
    subprocess.run(command, cwd=project_root, check=True)
    return archive_path


def package_outputs(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path | None = None,
    splits: tuple[str, ...] = SPLITS,
) -> list[Path]:
    _assert_archive_prerequisites()

    resolved_output_dir = output_dir or project_root / DEFAULT_OUTPUT_DIR
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = project_root / resolved_output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    archives = [
        _create_archive(
            archive_path=resolved_output_dir / MANIFESTS_AND_REPORTS_ARCHIVE_NAME,
            members=MANIFESTS_AND_REPORTS_MEMBERS,
            project_root=project_root,
        )
    ]
    for split in splits:
        archives.append(
            _create_archive(
                archive_path=resolved_output_dir / SAMPLE_ARCHIVE_NAME_TEMPLATE.format(split=split),
                members=(Path("data/processed/v1/samples") / split,),
                project_root=project_root,
            )
        )
    return archives


def main() -> int:
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir, project_root=PROJECT_ROOT)
    archives = package_outputs(output_dir=output_dir, splits=tuple(args.splits))
    for archive_path in archives:
        print(
            f"created: {_display_path(archive_path, project_root=PROJECT_ROOT)} "
            f"({_format_size(archive_path.stat().st_size)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
