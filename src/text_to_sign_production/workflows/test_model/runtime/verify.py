"""Runtime readiness verification for the single-sample test-model workflow."""

from __future__ import annotations

import json

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import (
    parse_modeling_manifest_family,
    read_modeling_manifest,
)
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelRestorePlan,
    TestModelRuntimeCheck,
    TestModelRuntimeVerification,
)
from text_to_sign_production.workflows.test_model.layout import TestModelLayout


def verify_test_model_runtime(
    plan: TestModelRestorePlan,
    *,
    layout: TestModelLayout,
) -> TestModelRuntimeVerification:
    checks: list[TestModelRuntimeCheck] = []
    metadata_path = _target_for_group(plan, "model_run_metadata")
    if metadata_path is not None:
        checks.append(_json_file_check("model run metadata", metadata_path, required=True))
    effective_config_path = _target_for_group(plan, "effective_config")
    if effective_config_path is not None:
        checks.append(_json_file_check("effective config", effective_config_path, required=True))
    research_spec_path = _target_for_group(plan, "research_spec")
    if research_spec_path is not None:
        checks.append(_json_file_check("research spec", research_spec_path, required=True))
    metadata = _metadata(metadata_path)
    if metadata.get("model_key") == "learned_pose_token":
        tokenizer_path = _target_for_group(plan, "learned_pose_token_tokenizer_checkpoint")
        if tokenizer_path is not None:
            checks.append(
                _file_check("learned pose-token tokenizer checkpoint", tokenizer_path, required=True)
            )
        stats_path = _target_for_group(plan, "learned_pose_token_standardization_stats")
        if stats_path is not None:
            checks.append(
                _json_file_check("learned pose-token standardization stats", stats_path, required=True)
            )
    if metadata.get("model_key") == "latent_diffusion":
        for group, label in (
            ("latent_diffusion_target_spec", "latent diffusion target spec"),
            ("latent_diffusion_standardization_stats", "latent diffusion standardization stats"),
            ("latent_diffusion_schedule", "latent diffusion schedule"),
            ("latent_diffusion_denoiser_config", "latent diffusion denoiser config"),
            (
                "latent_diffusion_length_predictor_config",
                "latent diffusion length predictor config",
            ),
        ):
            path = _target_for_group(plan, group)
            if path is not None:
                checks.append(_json_file_check(label, path, required=True))
        for group, label, schema_version in (
            (
                "latent_diffusion_denoiser_config",
                "latent diffusion denoiser config schema",
                "t2sp-latent-denoiser-config-v1",
            ),
            (
                "latent_diffusion_length_predictor_config",
                "latent diffusion length predictor config schema",
                "t2sp-latent-length-predictor-config-v1",
            ),
        ):
            path = _target_for_group(plan, group)
            if path is not None:
                checks.append(
                    _json_schema_check(
                        label,
                        path,
                        required=True,
                        schema_version=schema_version,
                    )
                )
    if metadata.get("model_key") == "articulator_aware":
        for group, label, schema_version in (
            (
                "articulator_partition_policy",
                "articulator-aware partition policy",
                "t2sp-articulator-partition-policy-v1",
            ),
            (
                "articulator_mask_strategy",
                "articulator-aware mask strategy",
                "t2sp-articulator-mask-strategy-v1",
            ),
            (
                "articulator_loss_weighting_policy",
                "articulator-aware loss weighting policy",
                "t2sp-channel-loss-weighting-v1",
            ),
            (
                "articulator_model_config",
                "articulator-aware model config",
                "t2sp-articulator-model-config-v1",
            ),
            (
                "articulator_length_predictor_config",
                "articulator-aware length predictor config",
                "t2sp-articulator-length-predictor-config-v1",
            ),
        ):
            path = _target_for_group(plan, group)
            if path is not None:
                checks.append(
                    _json_schema_check(
                        label,
                        path,
                        required=True,
                        schema_version=schema_version,
                    )
                )
    manifest_path = _target_for_group(plan, "test_manifest")
    if manifest_path is not None:
        checks.append(_file_check("test manifest", manifest_path, required=True))
    samples_root = layout.stores.runtime.samples.passed_split_dir(SampleSplit.TEST).path
    checks.append(_dir_check("test prepared samples root", samples_root, required=True))
    manifest_family_value = metadata.get("manifest_family")
    if isinstance(manifest_family_value, str):
        try:
            read_modeling_manifest(
                layout.stores.runtime,
                parse_modeling_manifest_family(manifest_family_value),
                SampleSplit.TEST,
            )
        except (OSError, ValueError, TypeError) as exc:
            checks.append(
                TestModelRuntimeCheck(
                    "test manifest schema",
                    manifest_path or layout.stores.runtime.manifests_root,
                    True,
                    manifest_path.exists() if manifest_path else False,
                    False,
                    str(exc),
                )
            )
    source_root = layout.stores.runtime.assets.keypoint_video_dir(SampleSplit.TEST).path
    checks.append(_dir_check("test source video root", source_root, required=False))
    errors = tuple(
        f"{check.label}: {check.message or 'missing or invalid'}"
        for check in checks
        if check.required and (not check.exists or not check.valid)
    )
    warnings = tuple(
        f"{check.label}: {check.message or 'missing or invalid'}"
        for check in checks
        if not check.required and (not check.exists or not check.valid)
    )
    return TestModelRuntimeVerification(checks=tuple(checks), errors=errors, warnings=warnings)


def _target_for_group(plan: TestModelRestorePlan, group: str):
    for operation in plan.operations:
        if operation.group == group:
            return operation.target
    return None


def _metadata(path) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _json_file_check(label: str, path, *, required: bool) -> TestModelRuntimeCheck:
    check = _file_check(label, path, required=required)
    if not check.exists or not check.valid:
        return check
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return TestModelRuntimeCheck(label, path, required, True, False, str(exc))
    return check


def _json_schema_check(
    label: str,
    path,
    *,
    required: bool,
    schema_version: str,
) -> TestModelRuntimeCheck:
    check = _file_check(label, path, required=required)
    if not check.exists or not check.valid:
        return check
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return TestModelRuntimeCheck(label, path, required, True, False, str(exc))
    if not isinstance(loaded, dict):
        return TestModelRuntimeCheck(label, path, required, True, False, "JSON root is not an object")
    observed = loaded.get("schema_version")
    if observed != schema_version:
        return TestModelRuntimeCheck(
            label,
            path,
            required,
            True,
            False,
            f"schema_version must be {schema_version!r}; observed {observed!r}",
        )
    return TestModelRuntimeCheck(label, path, required, True, True)


def _file_check(label: str, path, *, required: bool) -> TestModelRuntimeCheck:
    if not path.exists():
        return TestModelRuntimeCheck(label, path, required, False, False, "file does not exist")
    if not path.is_file():
        return TestModelRuntimeCheck(label, path, required, True, False, "path is not a file")
    return TestModelRuntimeCheck(label, path, required, True, True)


def _dir_check(label: str, path, *, required: bool) -> TestModelRuntimeCheck:
    if not path.exists():
        return TestModelRuntimeCheck(label, path, required, False, False, "directory does not exist")
    if not path.is_dir():
        return TestModelRuntimeCheck(label, path, required, True, False, "path is not a directory")
    return TestModelRuntimeCheck(label, path, required, True, True)


__all__ = ["verify_test_model_runtime"]
