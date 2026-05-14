from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.data.dataset.manifests import read_passed_manifest_json
from text_to_sign_production.data.dataset.payloads import load_prepared_sample_payload
from text_to_sign_production.data.dataset.validate import validate_payload_manifest_coherence
from text_to_sign_production.data.tier.policies import (
    load_tier_filters_config,
    load_tier_policies_config,
)
from text_to_sign_production.workflows.foundation.execution import ReadinessLevel
from text_to_sign_production.workflows.foundation.provenance import FileProvenance
from text_to_sign_production.workflows.tier.contracts import (
    TierRuntimeAssetCheck,
    TierRuntimePlan,
    TierRuntimeVerification,
    TierSplitRuntimeInputs,
)


def verify_tier_runtime(
    plan: TierRuntimePlan,
) -> TierRuntimeVerification:
    return TierRuntimeVerification(
        checks=_build_runtime_asset_checks(plan),
        readiness_level=ReadinessLevel.FAST_READINESS,
        checked_semantics=(
            "config_digest_and_parse",
            "passed_manifest_schema_and_identity",
            "payload_presence_for_manifest_rows",
            "sampled_payload_manifest_coherence",
        ),
        limitations=(
            "payload/manifest coherence is sampled to the first 5 rows per split",
            "payload arrays are loaded only for sampled coherence checks",
        ),
    )


def _build_runtime_asset_checks(
    plan: TierRuntimePlan,
) -> tuple[TierRuntimeAssetCheck, ...]:
    execution_inputs = plan.execution_inputs
    return (
        _filters_config_check(execution_inputs.filters_config_provenance),
        _tier_config_check(execution_inputs.tier_config_provenance),
        _runtime_directory_check("passed samples root", execution_inputs.passed_samples_root),
        *(
            check
            for split_input in execution_inputs.split_inputs
            for check in _split_runtime_checks(
                split_input,
                passed_samples_root=execution_inputs.passed_samples_root,
            )
        ),
    )


def _split_runtime_checks(
    split_input: TierSplitRuntimeInputs,
    *,
    passed_samples_root: Path,
) -> tuple[TierRuntimeAssetCheck, ...]:
    return (
        _passed_manifest_check(
            f"passed manifest [{split_input.split}]",
            split_input.passed_manifest_path,
        ),
        _runtime_directory_check(
            f"passed samples root [{split_input.split}]",
            split_input.passed_samples_split_root,
            glob_pattern="*.npz",
        ),
        _payload_world_check(
            f"passed payload world [{split_input.split}]",
            passed_manifest_path=split_input.passed_manifest_path,
            passed_samples_root=passed_samples_root,
        ),
    )


def _runtime_directory_check(
    label: str,
    path: Path,
    *,
    glob_pattern: str | None = None,
) -> TierRuntimeAssetCheck:
    if not path.exists():
        return TierRuntimeAssetCheck(label=label, path=path, exists=False)
    if not path.is_dir():
        return TierRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message="path is not a directory",
        )
    if glob_pattern is not None and not any(path.glob(glob_pattern)):
        return TierRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message=f"directory has no files matching {glob_pattern!r}",
        )
    return TierRuntimeAssetCheck(label=label, path=path, exists=True, valid=True)


def _filters_config_check(provenance: FileProvenance) -> TierRuntimeAssetCheck:
    path = provenance.execution_path
    if not path.exists():
        return TierRuntimeAssetCheck(label=provenance.label, path=path, exists=False)
    try:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != provenance.sha256:
            return TierRuntimeAssetCheck(
                label=provenance.label,
                path=path,
                exists=True,
                valid=False,
                message=(
                    "config digest mismatch: "
                    f"expected {provenance.sha256}, observed {actual_sha256}"
                ),
            )
        load_tier_filters_config(path)
    except (OSError, ValueError, TypeError) as exc:
        return TierRuntimeAssetCheck(
            label=provenance.label,
            path=path,
            exists=True,
            valid=False,
            message=str(exc),
        )
    return TierRuntimeAssetCheck(label=provenance.label, path=path, exists=True, valid=True)


def _tier_config_check(provenance: FileProvenance) -> TierRuntimeAssetCheck:
    path = provenance.execution_path
    if not path.exists():
        return TierRuntimeAssetCheck(label=provenance.label, path=path, exists=False)
    try:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != provenance.sha256:
            return TierRuntimeAssetCheck(
                label=provenance.label,
                path=path,
                exists=True,
                valid=False,
                message=(
                    "config digest mismatch: "
                    f"expected {provenance.sha256}, observed {actual_sha256}"
                ),
            )
        load_tier_policies_config(path)
    except (OSError, ValueError, TypeError) as exc:
        return TierRuntimeAssetCheck(
            label=provenance.label,
            path=path,
            exists=True,
            valid=False,
            message=str(exc),
        )
    return TierRuntimeAssetCheck(label=provenance.label, path=path, exists=True, valid=True)


def _passed_manifest_check(label: str, path: Path) -> TierRuntimeAssetCheck:
    if not path.exists():
        return TierRuntimeAssetCheck(label=label, path=path, exists=False, scope="domain")
    try:
        entries = tuple(read_passed_manifest_json(path))
        if not entries:
            raise ValueError("passed manifest has no rows")
        identities = tuple((entry.split, entry.sample_id) for entry in entries)
        if len(set(identities)) != len(identities):
            raise ValueError("passed manifest contains duplicate split/sample identities")
        if any(not entry.payload_ref.strip() for entry in entries):
            raise ValueError("passed manifest contains blank payload_ref")
    except (OSError, ValueError, TypeError) as exc:
        return TierRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message=str(exc),
            scope="domain",
        )
    return TierRuntimeAssetCheck(label=label, path=path, exists=True, valid=True, scope="domain")


def _payload_world_check(
    label: str,
    *,
    passed_manifest_path: Path,
    passed_samples_root: Path,
) -> TierRuntimeAssetCheck:
    path = passed_samples_root
    if not path.exists():
        return TierRuntimeAssetCheck(label=label, path=path, exists=False, scope="domain")
    try:
        entries = read_passed_manifest_json(passed_manifest_path)
        samples_root = passed_samples_root.parent
        missing_payloads: list[str] = []
        coherence_issues: list[str] = []
        for entry in entries:
            payload_path = samples_root / entry.payload_ref
            if not payload_path.is_file():
                missing_payloads.append(f"{entry.split.value}/{entry.sample_id}")
                continue
        for entry in entries[:5]:
            payload_path = samples_root / entry.payload_ref
            if not payload_path.is_file():
                continue
            sample = load_prepared_sample_payload(payload_path)
            issues = validate_payload_manifest_coherence(sample, entry)
            if issues:
                coherence_issues.append(f"{entry.split.value}/{entry.sample_id}: {issues}")
        if missing_payloads or coherence_issues:
            raise ValueError(
                "passed manifest/payload readiness failed "
                f"(missing_payloads={missing_payloads[:5]}, "
                f"sampled_coherence_issues={coherence_issues[:3]})"
            )
    except (OSError, ValueError, TypeError) as exc:
        return TierRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message=str(exc),
            scope="domain",
        )
    return TierRuntimeAssetCheck(label=label, path=path, exists=True, valid=True, scope="domain")
