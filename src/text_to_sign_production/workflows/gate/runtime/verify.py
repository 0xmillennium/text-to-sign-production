from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.data.gate.policies import load_gates_config
from text_to_sign_production.data.gate.sources.translations import (
    load_translation_records,
    validate_translation_columns,
)
from text_to_sign_production.workflows.foundation.execution import ReadinessLevel
from text_to_sign_production.workflows.foundation.provenance import FileProvenance
from text_to_sign_production.workflows.gate.contracts import (
    GateRuntimeAssetCheck,
    GateRuntimePlan,
    GateRuntimeVerification,
    GateSplitRuntimeInputs,
)


def verify_gate_runtime(
    plan: GateRuntimePlan,
) -> GateRuntimeVerification:
    return GateRuntimeVerification(
        checks=_build_runtime_asset_checks(plan),
        readiness_level=ReadinessLevel.FAST_READINESS,
        checked_semantics=(
            "config_digest_and_parse",
            "translation_schema_time_and_physical_identity",
            "keypoint_video_directory_presence",
            "sampled_keypoint_video_alignment",
        ),
        limitations=(
            "keypoint/video alignment is sampled to the first 10 translation rows per split",
            "frame JSON content is not exhaustively parsed during runtime readiness",
        ),
    )


def _build_runtime_asset_checks(
    plan: GateRuntimePlan,
) -> tuple[GateRuntimeAssetCheck, ...]:
    checks = [
        _gates_config_check(plan.execution_inputs.gates_config_provenance),
    ]
    for split_input in plan.execution_inputs.split_inputs:
        checks.extend(
            _split_runtime_checks(
                split_input=split_input,
            )
        )
    return tuple(checks)


def _split_runtime_checks(
    *,
    split_input: GateSplitRuntimeInputs,
) -> tuple[GateRuntimeAssetCheck, ...]:
    return (
        _translation_check(
            f"translation csv [{split_input.split}]",
            split_input.translation_csv_path,
        ),
        _directory_check(f"keypoint root [{split_input.split}]", split_input.keypoint_root),
        _directory_check(
            f"keypoint json root [{split_input.split}]",
            split_input.keypoint_json_root,
            glob_pattern="*/*.json",
        ),
        _directory_check(
            f"keypoint video root [{split_input.split}]",
            split_input.keypoint_video_root,
            glob_pattern="*.mp4",
        ),
        _keypoint_world_check(
            f"keypoint world [{split_input.split}]",
            split_input=split_input,
        ),
    )


def _directory_check(
    label: str,
    path: Path,
    *,
    glob_pattern: str | None = None,
) -> GateRuntimeAssetCheck:
    if not path.exists():
        return GateRuntimeAssetCheck(label=label, path=path, exists=False)
    if not path.is_dir():
        return GateRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message="path is not a directory",
        )
    if glob_pattern is not None and not any(path.glob(glob_pattern)):
        return GateRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message=f"directory has no files matching {glob_pattern!r}",
        )
    return GateRuntimeAssetCheck(label=label, path=path, exists=True, valid=True)


def _gates_config_check(provenance: FileProvenance) -> GateRuntimeAssetCheck:
    path = provenance.execution_path
    if not path.exists():
        return GateRuntimeAssetCheck(label=provenance.label, path=path, exists=False)
    try:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != provenance.sha256:
            return GateRuntimeAssetCheck(
                label=provenance.label,
                path=path,
                exists=True,
                valid=False,
                message=(
                    "config digest mismatch: "
                    f"expected {provenance.sha256}, observed {actual_sha256}"
                ),
            )
        load_gates_config(path)
    except (OSError, ValueError, TypeError) as exc:
        return GateRuntimeAssetCheck(
            label=provenance.label,
            path=path,
            exists=True,
            valid=False,
            message=str(exc),
        )
    return GateRuntimeAssetCheck(label=provenance.label, path=path, exists=True, valid=True)


def _translation_check(
    label: str,
    path: Path,
) -> GateRuntimeAssetCheck:
    if not path.exists():
        return GateRuntimeAssetCheck(label=label, path=path, exists=False)
    try:
        validate_translation_columns(path)
    except (OSError, ValueError, TypeError) as exc:
        return GateRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message=str(exc),
            scope="domain",
        )
    return GateRuntimeAssetCheck(label=label, path=path, exists=True, valid=True, scope="domain")


def _keypoint_world_check(
    label: str,
    *,
    split_input: GateSplitRuntimeInputs,
) -> GateRuntimeAssetCheck:
    path = split_input.keypoint_json_root
    if not path.exists():
        return GateRuntimeAssetCheck(label=label, path=path, exists=False, scope="domain")
    try:
        translations = load_translation_records(split_input.translation_csv_path)
        if not translations:
            raise ValueError("translation file has no sample rows")
        missing_keypoints: list[str] = []
        missing_videos: list[str] = []
        empty_keypoints: list[str] = []
        for translation in translations[:10]:
            keypoint_dir = split_input.keypoint_json_root / translation.sentence_name
            video_path = split_input.keypoint_video_root / f"{translation.sentence_name}.mp4"
            if not keypoint_dir.is_dir():
                missing_keypoints.append(translation.sentence_name)
            elif not any(keypoint_dir.glob("*.json")):
                empty_keypoints.append(translation.sentence_name)
            if not video_path.is_file():
                missing_videos.append(translation.sentence_name)
        if missing_keypoints or missing_videos or empty_keypoints:
            raise ValueError(
                "sampled keypoint/video readiness failed "
                f"(missing_keypoint_dirs={missing_keypoints[:3]}, "
                f"empty_keypoint_dirs={empty_keypoints[:3]}, "
                f"missing_videos={missing_videos[:3]})"
            )
    except (OSError, ValueError, TypeError) as exc:
        return GateRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message=str(exc),
            scope="domain",
        )
    return GateRuntimeAssetCheck(label=label, path=path, exists=True, valid=True, scope="domain")
