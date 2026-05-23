"""Compute profile loading and torch runtime configuration for model workflow runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

MODEL_COMPUTE_PROFILE_SCHEMA_VERSION = "model.compute_profile.v1"
DEFAULT_MODEL_COMPUTE_PROFILE = "portable"


@dataclass(frozen=True, slots=True)
class ModelPrecisionProfile:
    policy: str

    def __post_init__(self) -> None:
        if self.policy not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("compute profile precision.policy is invalid")

    def to_dict(self) -> dict[str, object]:
        return {"policy": self.policy}


@dataclass(frozen=True, slots=True)
class ModelTorchProfile:
    allow_tf32: bool
    float32_matmul_precision: str
    compile: bool

    def __post_init__(self) -> None:
        if not isinstance(self.allow_tf32, bool):
            raise ValueError("compute profile torch.allow_tf32 must be boolean")
        if self.float32_matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError(
                "compute profile torch.float32_matmul_precision is invalid"
            )
        if not isinstance(self.compile, bool):
            raise ValueError("compute profile torch.compile must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_tf32": self.allow_tf32,
            "float32_matmul_precision": self.float32_matmul_precision,
            "compile": self.compile,
        }


@dataclass(frozen=True, slots=True)
class ModelDataloaderProfile:
    num_workers: int | None
    pin_memory: bool
    persistent_workers: bool | None
    prefetch_factor: int | None
    materialization_workers: int | None = None

    def __post_init__(self) -> None:
        for name in ("num_workers", "prefetch_factor", "materialization_workers"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"compute profile dataloader.{name} is invalid")
        if not isinstance(self.pin_memory, bool):
            raise ValueError("compute profile dataloader.pin_memory must be boolean")
        if self.persistent_workers is not None and not isinstance(self.persistent_workers, bool):
            raise ValueError(
                "compute profile dataloader.persistent_workers must be boolean or null"
            )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
        }
        if self.materialization_workers is not None:
            payload["materialization_workers"] = self.materialization_workers
        return payload


@dataclass(frozen=True, slots=True)
class ModelComputeProfile:
    schema_version: str
    name: str
    precision: ModelPrecisionProfile
    torch: ModelTorchProfile
    dataloader: ModelDataloaderProfile
    provider_overrides: Mapping[str, object]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_COMPUTE_PROFILE_SCHEMA_VERSION:
            raise ValueError("compute profile schema_version is unsupported")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("compute profile name must be non-empty")
        if not isinstance(self.precision, ModelPrecisionProfile):
            raise ValueError("compute profile precision is invalid")
        if not isinstance(self.torch, ModelTorchProfile):
            raise ValueError("compute profile torch section is invalid")
        if not isinstance(self.dataloader, ModelDataloaderProfile):
            raise ValueError("compute profile dataloader section is invalid")
        if not isinstance(self.provider_overrides, Mapping):
            raise ValueError("compute profile provider_overrides must be a mapping")
        if any(not isinstance(key, str) for key in self.provider_overrides):
            raise ValueError("compute profile provider_overrides keys must be strings")
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "provider_overrides",
            MappingProxyType(dict(self.provider_overrides)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "source_path": None if self.source_path is None else str(self.source_path),
            "precision": self.precision.to_dict(),
            "torch": self.torch.to_dict(),
            "dataloader": self.dataloader.to_dict(),
            "provider_overrides": dict(self.provider_overrides),
        }


def load_model_compute_profile(project_root: Path, name: str) -> ModelComputeProfile:
    """Load one strict model compute profile by name."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("compute_profile must be non-empty")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("compute_profile must be a safe profile name")
    path = Path(project_root) / "configs" / "modeling" / "compute_profiles" / f"{name}.yaml"
    if not path.is_file():
        if name == DEFAULT_MODEL_COMPUTE_PROFILE:
            return _portable_profile(source_path=None)
        raise ValueError(f"unknown model compute profile: {name!r}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"compute profile YAML is invalid: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("compute profile must be a YAML mapping")
    raw = cast(Mapping[str, Any], loaded)
    expected = {
        "schema_version",
        "name",
        "precision",
        "torch",
        "dataloader",
        "provider_overrides",
    }
    if set(raw) != expected:
        raise ValueError("compute profile keys do not match schema")
    profile = ModelComputeProfile(
        schema_version=_text(raw["schema_version"], "schema_version"),
        name=_text(raw["name"], "name"),
        precision=ModelPrecisionProfile(**_mapping(raw["precision"], "precision")),
        torch=ModelTorchProfile(**_mapping(raw["torch"], "torch")),
        dataloader=ModelDataloaderProfile(**_mapping(raw["dataloader"], "dataloader")),
        provider_overrides=_mapping(raw["provider_overrides"], "provider_overrides"),
        source_path=path.resolve(strict=False),
    )
    if profile.name != name:
        raise ValueError("compute profile file name and name field differ")
    return profile


def _portable_profile(*, source_path: Path | None) -> ModelComputeProfile:
    return ModelComputeProfile(
        schema_version=MODEL_COMPUTE_PROFILE_SCHEMA_VERSION,
        name=DEFAULT_MODEL_COMPUTE_PROFILE,
        precision=ModelPrecisionProfile(policy="auto"),
        torch=ModelTorchProfile(
            allow_tf32=False,
            float32_matmul_precision="highest",
            compile=False,
        ),
        dataloader=ModelDataloaderProfile(
            num_workers=None,
            pin_memory=True,
            persistent_workers=None,
            prefetch_factor=None,
            materialization_workers=None,
        ),
        provider_overrides={},
        source_path=source_path,
    )


def provider_active_overrides(
    compute_profile: Mapping[str, object],
    *,
    provider_key: str,
    run_mode: str,
) -> Mapping[str, object]:
    """Return active provider overrides for one profile/run mode."""

    provider = _provider_profile_section(compute_profile, provider_key)
    active = provider.get("active") if provider is not None else None
    if not isinstance(active, Mapping):
        return MappingProxyType({})
    selected = active.get(run_mode)
    if not isinstance(selected, Mapping) or any(not isinstance(key, str) for key in selected):
        return MappingProxyType({})
    return MappingProxyType(dict(selected))


def provider_candidate_overrides(
    compute_profile: Mapping[str, object],
    *,
    provider_key: str,
) -> Mapping[str, object]:
    """Return tuning candidate values for one provider profile section."""

    provider = _provider_profile_section(compute_profile, provider_key)
    candidates = provider.get("candidates") if provider is not None else None
    if not isinstance(candidates, Mapping) or any(not isinstance(key, str) for key in candidates):
        return MappingProxyType({})
    return MappingProxyType(dict(candidates))


def _provider_profile_section(
    compute_profile: Mapping[str, object],
    provider_key: str,
) -> Mapping[str, object] | None:
    if not isinstance(compute_profile, Mapping):
        return None
    provider_overrides = compute_profile.get("provider_overrides")
    if not isinstance(provider_overrides, Mapping):
        return None
    provider = provider_overrides.get(provider_key)
    if not isinstance(provider, Mapping) or any(not isinstance(key, str) for key in provider):
        return None
    return cast(Mapping[str, object], provider)


def apply_torch_runtime_settings(profile: ModelComputeProfile) -> None:
    """Apply guarded torch runtime switches from a compute profile."""

    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = profile.torch.allow_tf32
        torch.backends.cudnn.allow_tf32 = profile.torch.allow_tf32
    torch.set_float32_matmul_precision(profile.torch.float32_matmul_precision)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"compute profile {name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"compute profile {name} must be non-empty text")
    return value


__all__ = [
    "DEFAULT_MODEL_COMPUTE_PROFILE",
    "MODEL_COMPUTE_PROFILE_SCHEMA_VERSION",
    "ModelComputeProfile",
    "apply_torch_runtime_settings",
    "load_model_compute_profile",
    "provider_active_overrides",
    "provider_candidate_overrides",
]
