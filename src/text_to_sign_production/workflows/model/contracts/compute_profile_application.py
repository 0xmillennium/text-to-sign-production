"""Validated compute-profile application ledger contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

SCHEMA_VERSION = "model.compute_profile_application.v1"


@dataclass(frozen=True, slots=True)
class ProviderComputeProfileApplication:
    schema_version: str
    profile_name: str
    provider_key: str
    run_mode: str

    active_requested: Mapping[str, object]
    active_applied: Mapping[str, object]
    active_not_applicable: Mapping[str, str]
    active_unsupported: Mapping[str, str]

    dataloader_requested: Mapping[str, object]
    dataloader_applied: Mapping[str, object]
    dataloader_not_applicable: Mapping[str, str]
    dataloader_unsupported: Mapping[str, str]

    candidates_requested: Mapping[str, object]
    candidates_applicable: Mapping[str, object]
    candidates_not_applicable: Mapping[str, str]
    candidates_unsupported: Mapping[str, str]

    telemetry_required_fields: tuple[str, ...]
    calibration_candidate_keys: tuple[str, ...]
    calibration_override_targets: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("compute profile application schema_version is unsupported.")
        for field_name in ("profile_name", "provider_key", "run_mode"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"compute profile application {field_name} must be text.")
        for field_name in (
            "active_requested",
            "active_applied",
            "dataloader_requested",
            "dataloader_applied",
            "candidates_requested",
            "candidates_applicable",
        ):
            object.__setattr__(self, field_name, _object_mapping(getattr(self, field_name), field_name))
        for field_name in (
            "active_not_applicable",
            "active_unsupported",
            "dataloader_not_applicable",
            "dataloader_unsupported",
            "candidates_not_applicable",
            "candidates_unsupported",
            "calibration_override_targets",
        ):
            object.__setattr__(self, field_name, _text_mapping(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "telemetry_required_fields",
            _text_tuple(self.telemetry_required_fields, "telemetry_required_fields"),
        )
        object.__setattr__(
            self,
            "calibration_candidate_keys",
            _text_tuple(self.calibration_candidate_keys, "calibration_candidate_keys"),
        )
        classify_requested_keys(
            self.active_requested,
            applied=self.active_applied,
            not_applicable=self.active_not_applicable,
            unsupported=self.active_unsupported,
            context="active",
        )
        classify_requested_keys(
            self.dataloader_requested,
            applied=self.dataloader_applied,
            not_applicable=self.dataloader_not_applicable,
            unsupported=self.dataloader_unsupported,
            context="dataloader",
        )
        classify_requested_keys(
            self.candidates_requested,
            applied=self.candidates_applicable,
            not_applicable=self.candidates_not_applicable,
            unsupported=self.candidates_unsupported,
            context="candidates",
        )
        missing_targets = set(self.calibration_candidate_keys) - set(
            self.calibration_override_targets
        )
        if missing_targets:
            raise ValueError(
                "calibration override targets are missing candidate keys: "
                f"{sorted(missing_targets)}."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "provider_key": self.provider_key,
            "run_mode": self.run_mode,
            "active_requested": dict(self.active_requested),
            "active_applied": dict(self.active_applied),
            "active_not_applicable": dict(self.active_not_applicable),
            "active_unsupported": dict(self.active_unsupported),
            "dataloader_requested": dict(self.dataloader_requested),
            "dataloader_applied": dict(self.dataloader_applied),
            "dataloader_not_applicable": dict(self.dataloader_not_applicable),
            "dataloader_unsupported": dict(self.dataloader_unsupported),
            "candidates_requested": dict(self.candidates_requested),
            "candidates_applicable": dict(self.candidates_applicable),
            "candidates_not_applicable": dict(self.candidates_not_applicable),
            "candidates_unsupported": dict(self.candidates_unsupported),
            "telemetry_required_fields": list(self.telemetry_required_fields),
            "calibration_candidate_keys": list(self.calibration_candidate_keys),
            "calibration_override_targets": dict(self.calibration_override_targets),
        }


def compute_profile_name(compute_profile: Mapping[str, object]) -> str | None:
    value = compute_profile.get("name") if isinstance(compute_profile, Mapping) else None
    return value if isinstance(value, str) and value.strip() else None


def compute_profile_dataloader_section(
    compute_profile: Mapping[str, object],
) -> Mapping[str, object]:
    value = compute_profile.get("dataloader") if isinstance(compute_profile, Mapping) else None
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        return MappingProxyType({})
    return MappingProxyType(dict(cast(Mapping[str, object], value)))


def provider_compute_profile_section(
    compute_profile: Mapping[str, object],
    *,
    provider_key: str,
) -> Mapping[str, object]:
    if not isinstance(compute_profile, Mapping):
        return MappingProxyType({})
    provider_overrides = compute_profile.get("provider_overrides")
    if not isinstance(provider_overrides, Mapping):
        return MappingProxyType({})
    provider = provider_overrides.get(provider_key)
    if not isinstance(provider, Mapping) or any(not isinstance(key, str) for key in provider):
        return MappingProxyType({})
    return MappingProxyType(dict(cast(Mapping[str, object], provider)))


def provider_active_overrides_for_run(
    compute_profile: Mapping[str, object],
    *,
    provider_key: str,
    run_mode: str,
) -> Mapping[str, object]:
    provider = provider_compute_profile_section(compute_profile, provider_key=provider_key)
    active = provider.get("active")
    if not isinstance(active, Mapping):
        return MappingProxyType({})
    selected = active.get(run_mode)
    if not isinstance(selected, Mapping) or any(not isinstance(key, str) for key in selected):
        return MappingProxyType({})
    return MappingProxyType(dict(cast(Mapping[str, object], selected)))


def provider_candidate_overrides_for_profile(
    compute_profile: Mapping[str, object],
    *,
    provider_key: str,
) -> Mapping[str, object]:
    provider = provider_compute_profile_section(compute_profile, provider_key=provider_key)
    candidates = provider.get("candidates")
    if not isinstance(candidates, Mapping) or any(
        not isinstance(key, str) for key in candidates
    ):
        return MappingProxyType({})
    return MappingProxyType(dict(cast(Mapping[str, object], candidates)))


def classify_requested_keys(
    requested: Mapping[str, object],
    *,
    applied: Mapping[str, object],
    not_applicable: Mapping[str, str],
    unsupported: Mapping[str, str],
    context: str,
) -> None:
    requested_keys = set(requested)
    applied_keys = set(applied)
    not_applicable_keys = set(not_applicable)
    unsupported_keys = set(unsupported)
    duplicate = (
        applied_keys & not_applicable_keys
        | applied_keys & unsupported_keys
        | not_applicable_keys & unsupported_keys
    )
    if duplicate:
        raise ValueError(
            f"compute profile application {context} keys have multiple classifications: "
            f"{sorted(duplicate)}."
        )
    classified = applied_keys | not_applicable_keys | unsupported_keys
    missing = requested_keys - classified
    extra = classified - requested_keys
    if missing:
        raise ValueError(
            f"compute profile application {context} keys are unclassified: {sorted(missing)}."
        )
    if extra:
        raise ValueError(
            f"compute profile application {context} classifications include unrequested keys: "
            f"{sorted(extra)}."
        )


def provider_compute_profile_application(
    *,
    compute_profile: Mapping[str, object],
    provider_key: str,
    run_mode: str,
    active_applied: Mapping[str, object],
    active_not_applicable: Mapping[str, str] | None = None,
    active_unsupported: Mapping[str, str] | None = None,
    dataloader_applied: Mapping[str, object],
    dataloader_not_applicable: Mapping[str, str] | None = None,
    dataloader_unsupported: Mapping[str, str] | None = None,
    candidates_applicable: Mapping[str, object],
    candidates_not_applicable: Mapping[str, str] | None = None,
    candidates_unsupported: Mapping[str, str] | None = None,
    telemetry_required_fields: tuple[str, ...],
    calibration_candidate_keys: tuple[str, ...],
    calibration_override_targets: Mapping[str, str],
) -> ProviderComputeProfileApplication:
    return ProviderComputeProfileApplication(
        schema_version=SCHEMA_VERSION,
        profile_name=compute_profile_name(compute_profile) or "unspecified",
        provider_key=provider_key,
        run_mode=run_mode,
        active_requested=provider_active_overrides_for_run(
            compute_profile,
            provider_key=provider_key,
            run_mode=run_mode,
        ),
        active_applied=active_applied,
        active_not_applicable=active_not_applicable or {},
        active_unsupported=active_unsupported or {},
        dataloader_requested=compute_profile_dataloader_section(compute_profile),
        dataloader_applied=dataloader_applied,
        dataloader_not_applicable=dataloader_not_applicable or {},
        dataloader_unsupported=dataloader_unsupported or {},
        candidates_requested=provider_candidate_overrides_for_profile(
            compute_profile,
            provider_key=provider_key,
        ),
        candidates_applicable=candidates_applicable,
        candidates_not_applicable=candidates_not_applicable or {},
        candidates_unsupported=candidates_unsupported or {},
        telemetry_required_fields=telemetry_required_fields,
        calibration_candidate_keys=calibration_candidate_keys,
        calibration_override_targets=calibration_override_targets,
    )


def _object_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"compute profile application {name} must be a mapping.")
    return MappingProxyType(dict(cast(Mapping[str, object], value)))


def _text_mapping(value: object, name: str) -> Mapping[str, str]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or any(not isinstance(item, str) or not item.strip() for item in value.values())
    ):
        raise ValueError(f"compute profile application {name} must map text to text.")
    return MappingProxyType(dict(cast(Mapping[str, str], value)))


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"compute profile application {name} must be a sequence.")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"compute profile application {name} must be a sequence.") from exc
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"compute profile application {name} must contain text.")
    if len(set(items)) != len(items):
        raise ValueError(f"compute profile application {name} must be unique.")
    return items


__all__ = [
    "ProviderComputeProfileApplication",
    "classify_requested_keys",
    "compute_profile_dataloader_section",
    "compute_profile_name",
    "provider_active_overrides_for_run",
    "provider_candidate_overrides_for_profile",
    "provider_compute_profile_application",
    "provider_compute_profile_section",
]
