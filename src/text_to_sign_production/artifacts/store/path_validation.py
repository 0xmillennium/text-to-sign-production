"""Path-shape validation for the physical artifact store."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path, PurePosixPath

from text_to_sign_production.core.ids import (
    SampleSplit,
    SampleStatus,
    TierMembership,
)

_SAFE_TIER_TOKEN_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


def validate_samples_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a samples-root-relative physical sample artifact path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) != 3:
        return [
            "Sample relative path must have shape "
            "passed/<split>/<filename> or dropped/<split>/<filename>."
        ]

    status, split, filename = parts
    errors: list[str] = []
    if status not in _values(SampleStatus):
        errors.append("Sample relative path status must be passed or dropped.")
    if split not in _values(SampleSplit):
        errors.append("Sample relative path split must be train, val, or test.")
    if filename in {"", ".", ".."}:
        errors.append("Sample relative path filename must be a concrete file name.")
    else:
        expected_suffix = ".npz" if status == SampleStatus.PASSED.value else ".json"
        if Path(filename).suffix != expected_suffix or not Path(filename).stem:
            errors.append(
                f"Sample relative path filename must be a concrete {expected_suffix} file name."
            )
    return errors


def validate_keypoint_archive_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative How2Sign 2D keypoint archive path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) != 4 or parts[:3] != ("assets", "how2sign", "bfh_keypoints"):
        return [
            "Keypoint archive relative path must have shape "
            "assets/how2sign/bfh_keypoints/<split>_2D_keypoints.tar.zst."
        ]

    filename = parts[3]
    errors: list[str] = []
    if not filename.endswith("_2D_keypoints.tar.zst"):
        errors.append("Keypoint archive filename must end with _2D_keypoints.tar.zst.")
        return errors

    split = filename.removesuffix("_2D_keypoints.tar.zst")
    if split not in _values(SampleSplit):
        errors.append("Keypoint archive split must be train, val, or test.")
    return errors


def validate_sample_archive_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative processed sample split archive path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) != 3 or parts[0] != "samples":
        return [
            "Sample archive relative path must have shape samples/<passed|dropped>/<split>.tar.zst."
        ]

    _, status, filename = parts
    errors: list[str] = []
    if status not in _values(SampleStatus):
        errors.append("Sample archive status must be passed or dropped.")
    if not filename.endswith(".tar.zst"):
        errors.append("Sample archive filename must end with .tar.zst.")
        return errors

    split = filename.removesuffix(".tar.zst")
    if split not in _values(SampleSplit):
        errors.append("Sample archive split must be train, val, or test.")
    return errors


def validate_sample_archive_member_path(relative_path: str | PurePosixPath) -> list[str]:
    """Validate a processed sample archive member path shape."""

    parts_errors, parts = _posix_relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) != 2:
        return ["Sample archive member path must have shape <split>/<sample_id>.<npz|json>."]

    split, filename = parts
    errors: list[str] = []
    if split not in _values(SampleSplit):
        errors.append("Sample archive member split must be train, val, or test.")
    if filename in {"", ".", ".."}:
        errors.append("Sample archive member filename must be a concrete file name.")
    elif (
        PurePosixPath(filename).suffix not in {".npz", ".json"} or not PurePosixPath(filename).stem
    ):
        errors.append("Sample archive member filename must be a concrete .npz or .json file name.")
    return errors


def validate_checkpoint_runtime_file_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative runtime model checkpoint file path shape."""

    return _validate_checkpoint_relative_path(
        relative_path,
        expected_suffix=".pt",
        label="Checkpoint runtime file",
    )


def validate_checkpoint_drive_archive_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative Drive model checkpoint archive path shape."""

    return _validate_checkpoint_relative_path(
        relative_path,
        expected_suffix=".pt.zst",
        label="Checkpoint Drive archive",
    )


def validate_manifests_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a manifests-root-relative physical manifest path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) == 3 and parts[0] == "untiered":
        _, status, filename = parts
        errors: list[str] = []
        if status not in _values(SampleStatus):
            errors.append("Untiered manifest status must be passed or dropped.")
        errors.extend(_validate_split_json_filename(filename, "Untiered manifest"))
        return errors

    if len(parts) == 4 and parts[0] == "tiered":
        _, tier, membership, filename = parts
        errors = []
        if not _SAFE_TIER_TOKEN_RE.fullmatch(tier):
            errors.append("Tiered manifest tier must be a lowercase safe token.")
        if membership not in _values(TierMembership):
            errors.append("Tiered manifest membership must be included or excluded.")
        errors.extend(_validate_split_json_filename(filename, "Tiered manifest"))
        return errors

    return [
        "Manifest relative path must have shape untiered/passed/<split>.json, "
        "untiered/dropped/<split>.json, or tiered/<tier>/<membership>/<split>.json."
    ]


def validate_model_run_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative model run artifact path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors
    safe_errors = _validate_safe_parts(parts, "Model run relative path")
    if safe_errors:
        return safe_errors
    if len(parts) < 4 or parts[0] != "models":
        return ["Model run relative path must be under models/<model_key>/<run_name>/."]
    _, model_key, run_name, family, *rest = parts
    errors: list[str] = []
    errors.extend(_validate_path_token(model_key, "Model key"))
    errors.extend(_validate_path_token(run_name, "Model run name"))
    if family in {"effective_config.json", "research_spec.json", "run_metadata.json"}:
        if rest:
            errors.append("Model run metadata files must be directly under the run root.")
        return errors
    if family == "checkpoints":
        if len(rest) != 1:
            errors.append("Model checkpoint path must include exactly one checkpoint filename.")
            return errors
        filename = rest[0]
        if not (filename.endswith(".pt") or filename.endswith(".pt.zst")):
            errors.append("Model checkpoint filename must end with .pt or .pt.zst.")
        if filename in {".pt", ".pt.zst"}:
            errors.append("Model checkpoint filename must include a concrete file stem.")
        return errors
    if family == "training":
        if tuple(rest) not in {
            ("metrics.jsonl",),
            ("summary.json",),
            ("live.log",),
        }:
            errors.append("Model training path must be metrics.jsonl, summary.json, or live.log.")
        return errors
    if family == "intermediates":
        if not rest:
            errors.append("Model intermediate path must include an intermediate name or file.")
        else:
            errors.extend(_validate_path_token(rest[0], "Model intermediate name"))
        return errors
    errors.append("Model run relative path family is not recognized.")
    return errors


def validate_generated_pose_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative generated-pose artifact path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors
    safe_errors = _validate_safe_parts(parts, "Generated-pose relative path")
    if safe_errors:
        return safe_errors
    if len(parts) >= 2 and parts[0] == "generated_pose":
        return _validate_generated_pose_diagnostic_tail(parts[1:])
    if len(parts) >= 6 and parts[:2] == ("reports", "test_model"):
        return _validate_test_model_generated_pose_path(parts)
    if len(parts) < 6 or parts[:2] != ("evaluations", "generated_pose"):
        return [
            "Generated-pose relative path must be under "
            "evaluations/generated_pose/<producer_key>/<run_name>/<split>/ "
            "or reports/test_model/<model_run>/<sample>/<execution>/generated_pose/."
        ]
    _, _, producer_key, run_name, split, *rest = parts
    errors: list[str] = []
    errors.extend(_validate_path_token(producer_key, "Generated-pose producer key"))
    errors.extend(_validate_path_token(run_name, "Generated-pose run name"))
    if split not in _values(SampleSplit):
        errors.append("Generated-pose split must be train, val, or test.")
    if tuple(rest) == ("manifest.jsonl",):
        return errors
    if len(rest) == 2 and rest[0] == "samples":
        filename = rest[1]
        match = _GENERATED_POSE_SAMPLE_RE.fullmatch(filename)
        if not match:
            errors.append(
                "Generated-pose sample filename must match {sample_id}__g{non_negative_int}.npz."
            )
        else:
            errors.extend(
                _validate_path_token(match.group("sample_id"), "Generated-pose sample id")
            )
        return errors
    errors.append(
        "Generated-pose relative path must end with manifest.jsonl or "
        "samples/{sample_id}__g{generation_index}.npz."
    )
    return errors


def _validate_test_model_generated_pose_path(parts: tuple[str, ...]) -> list[str]:
    if len(parts) < 7 or parts[:2] != ("reports", "test_model"):
        return [
            "Test-model generated-pose path must be under "
            "reports/test_model/<model_run>/<sample>/<execution>/generated_pose/."
        ]
    _, _, model_run, sample_id, execution_id, generated_pose, *rest = parts
    errors: list[str] = []
    errors.extend(_validate_path_token(model_run, "Test-model run name"))
    errors.extend(_validate_path_token(sample_id, "Test-model sample id"))
    errors.extend(_validate_path_token(execution_id, "Test-model execution id"))
    if generated_pose != "generated_pose":
        errors.append("Test-model generated-pose path must include generated_pose/.")
        return errors
    errors.extend(_validate_generated_pose_diagnostic_tail(tuple(rest)))
    return errors


def _validate_generated_pose_diagnostic_tail(parts: tuple[str, ...]) -> list[str]:
    if tuple(parts) == ("manifest.jsonl",):
        return []
    if len(parts) == 2 and parts[0] == "samples":
        filename = parts[1]
        match = _GENERATED_POSE_SAMPLE_RE.fullmatch(filename)
        if not match:
            return [
                "Generated-pose sample filename must match {sample_id}__g{non_negative_int}.npz."
            ]
        return _validate_path_token(match.group("sample_id"), "Generated-pose sample id")
    return [
        "Generated-pose diagnostic path must end with manifest.jsonl or "
        "samples/{sample_id}__g{generation_index}.npz."
    ]


def validate_modeling_report_relative_path(relative_path: str | Path) -> list[str]:
    """Validate a repo-relative model report path shape."""

    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors
    safe_errors = _validate_safe_parts(parts, "Modeling report relative path")
    if safe_errors:
        return safe_errors
    if len(parts) == 6 and parts[:2] == ("reports", "modeling") and parts[4] == "validation":
        errors = []
        errors.extend(_validate_path_token(parts[2], "Modeling report model key"))
        errors.extend(_validate_path_token(parts[3], "Modeling report run name"))
        errors.extend(_validate_validation_report_filename(parts[-1]))
        return errors
    if len(parts) != 5 or parts[:2] != ("reports", "modeling"):
        return [
            "Modeling report relative path must have shape "
            "reports/modeling/<model_key>/<run_name>/<filename> or "
            "reports/modeling/<model_key>/<run_name>/validation/<filename>."
        ]
    errors = []
    errors.extend(_validate_path_token(parts[2], "Modeling report model key"))
    errors.extend(_validate_path_token(parts[3], "Modeling report run name"))
    errors.extend(_validate_report_filename(parts[-1], "Modeling report"))
    return errors


def _values(enum_type: type[StrEnum]) -> set[str]:
    return {entry.value for entry in enum_type}


def _relative_parts(relative_path: str | Path) -> tuple[list[str], tuple[str, ...]]:
    path = Path(relative_path)
    if path.is_absolute():
        return ([f"Expected a relative path, got absolute path: {path}"], ())
    if not path.parts:
        return (["Expected a non-empty relative path."], ())
    if ".." in path.parts:
        return ([f"Relative path must not contain parent directory references: {path}"], ())
    return ([], path.parts)


def _posix_relative_parts(relative_path: str | PurePosixPath) -> tuple[list[str], tuple[str, ...]]:
    path = PurePosixPath(relative_path)
    if path.is_absolute():
        return ([f"Expected a relative path, got absolute path: {path}"], ())
    if not path.parts:
        return (["Expected a non-empty relative path."], ())
    if ".." in path.parts:
        return ([f"Relative path must not contain parent directory references: {path}"], ())
    return ([], path.parts)


def _validate_split_json_filename(filename: str, label: str) -> list[str]:
    errors: list[str] = []
    if not filename.endswith(".json"):
        errors.append(f"{label} filename must end with .json.")
        return errors

    split = filename.removesuffix(".json")
    if split not in _values(SampleSplit):
        errors.append(f"{label} filename must be train.json, val.json, or test.json.")
    return errors


def _validate_checkpoint_relative_path(
    relative_path: str | Path,
    *,
    expected_suffix: str,
    label: str,
) -> list[str]:
    parts_errors, parts = _relative_parts(relative_path)
    if parts_errors:
        return parts_errors

    if len(parts) < 2 or parts[0] != "models":
        return [f"{label} relative path must be under models/."]

    filename = parts[-1]
    if filename in {"", ".", ".."}:
        return [f"{label} filename must be a concrete file name."]
    if not filename.endswith(expected_suffix) or filename == expected_suffix:
        return [f"{label} filename must end with {expected_suffix}."]
    return []


def _validate_safe_parts(parts: tuple[str, ...], label: str) -> list[str]:
    errors: list[str] = []
    for part in parts:
        if not part.strip() or part in {".", ".."}:
            errors.append(f"{label} must not contain blank or dot path tokens.")
            break
        if "\\" in part:
            errors.append(f"{label} must not contain backslash path separators.")
            break
    return errors


def _validate_report_filename(filename: str, label: str) -> list[str]:
    if not (filename.endswith(".md") or filename.endswith(".json")):
        return [f"{label} filename must end with .md or .json."]
    if filename in {".md", ".json"}:
        return [f"{label} filename must include a concrete file stem."]
    return []


def _validate_validation_report_filename(filename: str) -> list[str]:
    if not (
        filename.endswith(".md")
        or filename.endswith(".json")
        or filename.endswith(".jsonl")
    ):
        return ["Model validation report filename must end with .md, .json, or .jsonl."]
    if filename in {".md", ".json", ".jsonl"}:
        return ["Model validation report filename must include a concrete file stem."]
    return []


def _validate_path_token(value: str, label: str) -> list[str]:
    if not value.strip() or value in {".", ".."}:
        return [f"{label} must be a non-empty concrete path token."]
    if "." in value or "/" in value or "\\" in value:
        return [f"{label} must not contain dot or path separator characters."]
    return []


_GENERATED_POSE_SAMPLE_RE = re.compile(r"(?P<sample_id>.+)__g(?:0|[1-9][0-9]*)\.npz")


__all__ = [
    "validate_checkpoint_drive_archive_relative_path",
    "validate_checkpoint_runtime_file_relative_path",
    "validate_generated_pose_relative_path",
    "validate_keypoint_archive_relative_path",
    "validate_manifests_relative_path",
    "validate_model_run_relative_path",
    "validate_modeling_report_relative_path",
    "validate_sample_archive_member_path",
    "validate_sample_archive_relative_path",
    "validate_samples_relative_path",
]
