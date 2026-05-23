"""Provider-neutral model stage planning and sequential execution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.modeling.artifacts import (
    GeneratedPoseSampleArchive,
    GeneratedPoseSampleArchiveError,
    build_generated_pose_samples_archive,
)
from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelProvider,
    ModelProviderLoadedConfig,
    ModelRunRequest,
    ModelStageArtifactRef,
    ModelStageExecutionError,
    ModelStagePlan,
    ModelStageResult,
    VerificationEvidenceRef,
    VerifiedProviderFullDataPipelineCapability,
    execute_stage_plan,
    validate_model_provider,
)
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.candidates.registry import require_model_provider
from text_to_sign_production.modeling.candidates.artifacts import (
    GeneratedPosePublishPolicy,
    GeneratedPoseSurface,
    generated_pose_samples_archive_artifact_ref,
    generated_pose_samples_archive_manifest_artifact_ref,
    generated_pose_samples_archive_sha256_artifact_ref,
)
from text_to_sign_production.workflows.model.contracts import (
    ModelStageExecutionWorkflowResult,
    ModelStagePlanningResult,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.processing.calibration import (
    PreExecutionCalibrationResult,
    build_calibrated_loaded_config,
    ensure_pre_execution_calibration,
    inspect_provider_real_calibration_artifacts,
    provider_model_build_spec_hash,
    provider_config_hash,
)
from text_to_sign_production.workflows.model.progress import visible_model_progress_session


def plan_model_stages(
    provider: ModelProvider,
    request: ModelRunRequest,
    loaded_config: ModelProviderLoadedConfig,
) -> ModelStagePlanningResult:
    """Request and validate a provider stage plan."""

    try:
        validate_model_provider(provider)
        stage_plan = provider.plan_stages(request, loaded_config)
    except Exception as exc:
        raise ModelWorkflowInvariantError("model provider stage planning failed") from exc
    if not isinstance(stage_plan, ModelStagePlan):
        raise ModelWorkflowInvariantError("model provider did not return ModelStagePlan")
    if loaded_config.model_key is not provider.spec.key:
        raise ModelWorkflowInvariantError("loaded config does not match provider model key")
    if stage_plan.request != request:
        raise ModelWorkflowInvariantError("provider stage plan request does not match request")
    if stage_plan.model.key is not request.model_key:
        raise ModelWorkflowInvariantError("provider stage plan model does not match request")
    if not stage_plan.stages:
        raise ModelWorkflowInvariantError("provider stage plan must be non-empty")
    return ModelStagePlanningResult(
        provider=provider,
        loaded_config=loaded_config,
        stage_plan=stage_plan,
    )


def execute_model_stage_plan(
    *,
    provider: ModelProvider,
    request: ModelRunRequest,
    loaded_config: ModelProviderLoadedConfig,
    topology: ArtifactTopology,
    working_dir: Path,
    stage_plan: ModelStagePlan,
    calibration_root: Path | None = None,
    progress_session: ProgressSession | None = None,
) -> ModelStageExecutionWorkflowResult:
    """Execute provider stages with workflow-owned status progress."""

    if stage_plan.request != request:
        raise ModelWorkflowInvariantError("stage execution request does not match stage plan")
    progress = visible_model_progress_session(progress_session)
    pre_calibration = ensure_pre_execution_calibration(
        provider=provider,
        request=request,
        loaded_config=loaded_config,
        stage_plan=stage_plan,
        output_root=Path(calibration_root) if calibration_root is not None else working_dir,
        working_dir=working_dir,
        topology=topology,
        progress_session=progress,
    )
    if pre_calibration.blocking_issues:
        raise ModelWorkflowInvariantError(
            "model pre-execution calibration blocked provider execution: "
            f"{pre_calibration.blocking_issues}"
        )
    calibrated_loaded_config = build_calibrated_loaded_config(
        loaded_config,
        pre_calibration,
    )
    _guard_full_run_data_surface_readiness(
        request,
        loaded_config=calibrated_loaded_config,
        calibration_root=calibration_root,
        pre_calibration=pre_calibration,
    )
    progress.status(
        "model provider stage plan start",
        model_key=request.model_key.value,
        run_mode=request.run_mode.value,
        stage_count=len(stage_plan.stages),
    )
    try:
        execution = execute_stage_plan(
            provider,
            stage_plan,
            loaded_config=calibrated_loaded_config,
            topology=topology,
            working_dir=working_dir,
            progress_session=progress,
        )
    except ModelStageExecutionError as exc:
        raise ModelWorkflowInvariantError("model provider stage execution failed") from exc
    execution = _package_generated_pose_archives(
        request=request,
        execution=execution,
        progress_session=progress,
    )
    ensure_no_exact_duplicate_model_artifact_refs(execution)
    progress.status(
        "model provider stage plan done",
        model_key=request.model_key.value,
        completed_stage_count=len(execution.stages),
    )
    return ModelStageExecutionWorkflowResult(
        provider=provider,
        loaded_config=calibrated_loaded_config,
        stage_plan=stage_plan,
        execution=execution,
        pre_calibration=pre_calibration,
    )


def _guard_full_run_data_surface_readiness(
    request: ModelRunRequest,
    *,
    loaded_config: ModelProviderLoadedConfig | None = None,
    calibration_root: Path | None = None,
    pre_calibration: PreExecutionCalibrationResult | None = None,
) -> None:
    if request.run_mode.value != "full":
        return
    ensure_model_provider_registered(request.model_key)
    provider = require_model_provider(request.model_key)
    capability_for_config = getattr(provider, "full_data_pipeline_capability_for_config", None)
    capability = (
        capability_for_config(loaded_config)
        if loaded_config is not None and callable(capability_for_config)
        else getattr(provider, "full_data_pipeline_capability", None)
    )
    if capability is None:
        raise ModelWorkflowInvariantError(
            "full run blocked before provider execution: "
            f"provider_capability_unverified provider={request.model_key.value} mode=missing"
        )
    if capability.full_training_data_mode not in {"streaming_sharded", "lazy_dataloader"}:
        raise ModelWorkflowInvariantError(
            "full run blocked before provider execution: "
            f"provider_full_pipeline_not_streaming provider_key={capability.provider_key} "
            f"full_training_data_mode={capability.full_training_data_mode} "
            f"verified={capability.verified} limitations={capability.limitations}"
        )
    missing = _missing_required_full_evidence(
        provider_key=request.model_key.value,
        observed=capability.verification_evidence,
        effective_config=loaded_config.effective_config if loaded_config is not None else None,
    )
    if not capability.is_full_safe or missing:
        reason = (
            "provider_capability_unverified"
            if not capability.verified
            else "provider_limitations_present"
            if capability.limitations
            else "provider_capability_unverified"
            if missing
            else "provider_full_pipeline_not_streaming"
        )
        raise ModelWorkflowInvariantError(
            "full run blocked before provider execution: "
            f"{reason} provider_key={capability.provider_key} "
            f"full_training_data_mode={capability.full_training_data_mode} "
            f"verified={capability.verified} "
            f"limitations={capability.limitations} "
            f"missing verification evidence={missing}"
        )
    if pre_calibration is None and calibration_root is None:
        raise ModelWorkflowInvariantError(
            "full run blocked before provider execution: "
            "pre_execution_calibration_missing provider_real_calibration" "_missing"
        )
    if pre_calibration is not None and pre_calibration.required:
        required_paths = {
            "provider_real_calibration": pre_calibration.compute_calibration_path,
            "selected_overrides": pre_calibration.selected_overrides_path,
            "calibrated_effective_config": pre_calibration.calibrated_effective_config_path,
        }
        missing_runtime_artifacts = tuple(
            name for name, path in required_paths.items() if path is None or not path.is_file()
        )
        if missing_runtime_artifacts:
            raise ModelWorkflowInvariantError(
                "full run blocked before provider execution: "
                f"missing runtime calibration artifacts={missing_runtime_artifacts}"
            )
    expected_hash = (
        pre_calibration.base_effective_config_hash
        if pre_calibration is not None
        else (
            provider_config_hash(loaded_config.effective_config)
            if loaded_config is not None
            else None
        )
    )
    expected_build_hash = (
        provider_model_build_spec_hash(
            request.model_key.value,
            effective_config=loaded_config.effective_config,
        )
        if loaded_config is not None
        else None
    )
    calibration_status = inspect_provider_real_calibration_artifacts(
        calibration_root=calibration_root,
        provider_key=request.model_key.value,
        expected_provider_config_hash=expected_hash,
        expected_model_build_spec_hash=expected_build_hash,
        required=True,
    )
    runtime_capability = VerifiedProviderFullDataPipelineCapability(
        provider_key=capability.provider_key,
        model_key=request.model_key.value,
        run_name=request.run_name,
        mode=capability.full_training_data_mode,
        verified=not calibration_status.blocking_issues,
        verification_artifact_paths=tuple(
            path
            for path in (
                calibration_status.calibration_artifact_path,
                calibration_status.selected_overrides_path,
            )
            if path is not None and path.is_file()
        ),
        calibration_artifact_path=calibration_status.calibration_artifact_path
        if calibration_status.calibration_authoritative
        else None,
        selected_overrides_path=calibration_status.selected_overrides_path
        if calibration_status.selected_overrides_found
        else None,
        behavior_evidence=capability.verification_evidence,
        limitations=capability.limitations,
        verification_evidence_refs=_runtime_evidence_refs(
            calibration_artifact_path=calibration_status.calibration_artifact_path,
            selected_overrides_path=calibration_status.selected_overrides_path,
            calibrated_effective_config_path=pre_calibration.calibrated_effective_config_path
            if pre_calibration is not None
            else None,
        )
        if not calibration_status.blocking_issues
        else (),
    )
    if not runtime_capability.is_full_safe or calibration_status.blocking_issues:
        raise ModelWorkflowInvariantError(
            "full run blocked before provider execution: "
            f"{calibration_status.blocking_issues or ('provider_capability_unverified',)} "
            f"provider_key={capability.provider_key} "
            f"full_training_data_mode={capability.full_training_data_mode} "
            f"verified={runtime_capability.verified} "
            f"limitations={runtime_capability.limitations} "
            f"missing verification evidence={missing} "
            f"calibration_artifact_path={calibration_status.calibration_artifact_path} "
            f"selected_overrides_path={calibration_status.selected_overrides_path}"
        )


def _missing_required_full_evidence(
    *,
    provider_key: str,
    observed: tuple[str, ...],
    effective_config: object | None = None,
) -> tuple[str, ...]:
    required = {
        "learned_pose_token": (
            "behavior:fit_representation:surface_reader",
            "behavior:train_text_to_token:surface_reader",
            "behavior:evaluate_reconstruction:incremental_writer",
            "behavior:decode_to_pose:incremental_writer",
            "progress:known_totals",
        ),
        "latent_diffusion": _latent_diffusion_required_full_evidence(effective_config),
        "articulator_aware": (
            "behavior:frame_training:surface_reader",
            "behavior:export:incremental_writer",
            "progress:known_totals",
        ),
        "base_direct": (
            "behavior:lazy_dataloader:no_full_payload_load_before_first_batch",
            "progress:known_totals",
        ),
    }.get(provider_key, ())
    present = set(observed)
    return tuple(item for item in required if item not in present)


def _latent_diffusion_required_full_evidence(
    effective_config: object | None,
) -> tuple[str, ...]:
    required = [
        "behavior:latent_sequences:surface_reader",
        "behavior:denoiser_training:surface_reader",
        "behavior:generation:incremental_writer",
        "progress:known_totals",
    ]
    latent_autoencoder = (
        effective_config.get("latent_autoencoder")
        if isinstance(effective_config, Mapping)
        else None
    )
    if (
        isinstance(latent_autoencoder, Mapping)
        and latent_autoencoder.get("active") is True
    ):
        required.extend(
            [
                "behavior:latent_windows:surface_reader",
                "behavior:autoencoder_training:surface_reader",
            ]
        )
    return tuple(required)


def _runtime_evidence_refs(
    *,
    calibration_artifact_path: Path | None,
    selected_overrides_path: Path | None,
    calibrated_effective_config_path: Path | None = None,
) -> tuple[VerificationEvidenceRef, ...]:
    refs: list[VerificationEvidenceRef] = []
    if calibration_artifact_path is not None and calibration_artifact_path.is_file():
        sha = _sha256_file(calibration_artifact_path)
        for kind in (
            "provider_real_calibration",
            "representative_surface",
            "progress_totals",
            "bounded_writers",
        ):
            refs.append(
                VerificationEvidenceRef(
                    kind=kind,
                    path=calibration_artifact_path,
                    sha256=sha,
                    summary={"artifact": "compute_calibration", "evidence": kind},
                )
            )
    if selected_overrides_path is not None and selected_overrides_path.is_file():
        refs.append(
            VerificationEvidenceRef(
                kind="selected_overrides",
                path=selected_overrides_path,
                sha256=_sha256_file(selected_overrides_path),
                summary={"artifact": "selected_overrides"},
            )
        )
    if calibrated_effective_config_path is not None and calibrated_effective_config_path.is_file():
        refs.append(
            VerificationEvidenceRef(
                kind="calibrated_effective_config",
                path=calibrated_effective_config_path,
                sha256=_sha256_file(calibrated_effective_config_path),
                summary={"artifact": "calibrated_effective_config"},
            )
        )
    elif calibration_artifact_path is not None and calibration_artifact_path.is_file():
        refs.append(
            VerificationEvidenceRef(
                kind="calibrated_effective_config",
                path=calibration_artifact_path,
                sha256=_sha256_file(calibration_artifact_path),
                summary={"artifact": "calibrated_effective_config", "legacy_guard_root": True},
            )
        )
    return tuple(refs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_generated_pose_archives(
    *,
    request: ModelRunRequest,
    execution: ModelExecutionResult,
    progress_session: ProgressSession | None = None,
) -> ModelExecutionResult:
    stages: list[ModelStageResult] = []
    for result in execution.stages:
        archives: list[tuple[GeneratedPoseSampleArchive, str]] = []
        for surface in result.generated_pose_surfaces:
            if not isinstance(surface, GeneratedPoseSurface):
                raise ModelWorkflowInvariantError(
                    "provider stage generated_pose_surfaces contain an invalid surface."
                )
            if surface.publish_policy is GeneratedPosePublishPolicy.RUNTIME_ONLY:
                continue
            if surface.publish_policy is not GeneratedPosePublishPolicy.PERSISTENT:
                raise ModelWorkflowInvariantError(
                    f"unknown generated-pose publish policy: {surface.publish_policy!r}"
                )
            if not surface.archive_required:
                raise ModelWorkflowInvariantError(
                    "persistent generated-pose model surfaces must require archive packaging."
                )
            try:
                archive_progress = ModelProviderArchiveProgress(
                    progress_session=progress_session,
                    provider_key=surface.model_key,
                    split=surface.split.value,
                    total=_generated_manifest_entry_count(surface.manifest_path),
                )
                archives.append(
                    (
                        build_generated_pose_samples_archive(
                            manifest_path=surface.manifest_path,
                            samples_root=surface.samples_dir,
                            model_run_name=surface.model_run_name,
                            model_key=surface.model_key,
                            manifest_family=surface.manifest_family,
                            split=surface.split,
                            on_event=archive_progress.event,
                            on_member=archive_progress.member,
                        ),
                        surface.producer_stage_id,
                    )
                )
            except GeneratedPoseSampleArchiveError as exc:
                raise ModelWorkflowInvariantError(
                    "generated-pose sample archive packaging failed"
                ) from exc
        archived_split_roots = {archive.manifest_path.parent for archive, _ in archives}
        retained_artifacts = tuple(
            artifact
            for artifact in result.artifacts
            if not _is_generated_pose_sample_artifact(
                artifact,
                archived_split_roots=archived_split_roots if archives else None,
            )
        )
        archive_artifacts = tuple(
            artifact
            for archive, producer_stage_id in archives
            for artifact in _archive_artifact_refs(
                archive,
                producer_stage_id=producer_stage_id,
            )
        )
        metadata = dict(result.metadata)
        metadata["generated_pose_archives"] = {
            archive.split.value: archive.to_summary_dict() for archive, _ in archives
        }
        stages.append(
            replace(
                result,
                artifacts=(*retained_artifacts, *archive_artifacts),
                metadata=metadata,
            )
        )
    return ModelExecutionResult(
        model_key=execution.model_key,
        run_name=execution.run_name,
        stages=tuple(stages),
    )


def ensure_no_exact_duplicate_model_artifact_refs(execution: ModelExecutionResult) -> None:
    """Reject exact duplicate stage artifact references across a model execution."""

    seen: dict[tuple[Path, str, str], ModelStageArtifactRef] = {}
    for artifact in execution.artifact_refs:
        key = (
            artifact.path.resolve(strict=False),
            artifact.kind,
            _artifact_role_text(artifact.role),
        )
        if key in seen:
            previous = seen[key]
            raise ModelWorkflowInvariantError(
                "duplicate model stage artifact ref: "
                f"path={key[0]}; kind={artifact.kind!r}; role={key[2]!r}; "
                f"previous_description={previous.description!r}; "
                f"current_description={artifact.description!r}"
            )
        seen[key] = artifact


def _artifact_role_text(role: object) -> str:
    value = getattr(role, "value", None)
    if isinstance(value, str):
        return value
    return str(role)


class ModelProviderArchiveProgress:
    def __init__(
        self,
        *,
        progress_session: ProgressSession | None,
        provider_key: str,
        split: str,
        total: int,
    ) -> None:
        self._progress = None
        self._task = None
        if progress_session is None:
            return
        from text_to_sign_production.workflows.model.processing.provider_progress import (
            ModelProviderProgress,
        )

        self._progress = ModelProviderProgress(
            progress_session=progress_session,
            provider_key=provider_key,
            provider_stage_id=f"generated_pose.archive.{split}",
        )
        self._task = self._progress.task(
            operation="members",
            label=f"model archive generated pose [{split}]",
            unit="member",
            total=total,
            total_semantics="generated pose archive manifest members",
            allowed_counters=("member", "bytes"),
        )

    def event(self, event: str, payload: dict[str, object]) -> None:
        if self._progress is not None:
            self._progress.status(f"generated pose archive {event}", **payload)
        if event == "archive_done" and self._task is not None:
            self._task.close()

    def member(self, index: int, total: int, sample_id: str, size_bytes: int) -> None:
        del total
        if self._task is not None:
            self._task.advance(
                1,
                counters={"member": index, "bytes": size_bytes},
            )
            self._task.status("generated pose archive member", sample_id=sample_id)


def _generated_manifest_entry_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _is_generated_pose_sample_artifact(
    artifact: ModelStageArtifactRef,
    *,
    archived_split_roots: set[Path] | None,
) -> bool:
    if artifact.role != "generated_pose_sample" and artifact.kind != "generated_pose_sample":
        return False
    if archived_split_roots is None:
        return True
    try:
        return artifact.path.parent.parent in archived_split_roots
    except IndexError:
        return False


def _archive_artifact_refs(
    archive: GeneratedPoseSampleArchive,
    *,
    producer_stage_id: str,
) -> tuple[ModelStageArtifactRef, ...]:
    common = {
        "split": archive.split,
        "producer_stage": producer_stage_id,
        "model_key": archive.model_key,
        "member_count": archive.member_count,
        "archive_sha256": archive.archive_sha256,
    }
    return (
        generated_pose_samples_archive_artifact_ref(
            path=archive.archive_path,
            **common,
        ),
        generated_pose_samples_archive_manifest_artifact_ref(
            path=archive.archive_manifest_path,
            **common,
        ),
        generated_pose_samples_archive_sha256_artifact_ref(
            path=archive.archive_sha256_path,
            **common,
        ),
    )


__all__ = [
    "ensure_no_exact_duplicate_model_artifact_refs",
    "execute_model_stage_plan",
    "plan_model_stages",
]
