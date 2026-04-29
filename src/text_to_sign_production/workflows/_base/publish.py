"""Base workflow publish plan construction and verification."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.files import (
    ensure_dir,
    require_dir,
    require_file,
    require_non_empty_file,
    sha256_file,
    verify_output_file,
)
from text_to_sign_production.core.paths import ProjectLayout, repo_relative_path
from text_to_sign_production.modeling.inference.qualitative import (
    PREDICTION_ARTIFACTS_DIRNAME,
    REFERENCE_ARTIFACTS_DIRNAME,
)
from text_to_sign_production.modeling.training.checkpointing import load_training_checkpoint
from text_to_sign_production.workflows._base.constants import (
    BASE_CHECKPOINT_MANIFEST_FILENAME,
    BASE_PREDICTION_SAMPLES_ARCHIVE_FILENAME,
    BASE_PREDICTION_SAMPLES_OBSERVED_MEMBERS_FILENAME,
    BASE_QUALITATIVE_PANEL_SAMPLES_ARCHIVE_FILENAME,
    BASE_QUALITATIVE_PANEL_SAMPLES_OBSERVED_MEMBERS_FILENAME,
    CHECKPOINT_MANIFEST_SCHEMA_VERSION,
)
from text_to_sign_production.workflows._base.layout import _resolve_manifest_or_repo_path
from text_to_sign_production.workflows._base.types import (
    BaseCheckpointPublishSpec,
    BaseFilePublishSpec,
    BasePredictionSampleArchiveSpec,
    BasePublishedCheckpointSummary,
    BasePublishedFileSummary,
    BasePublishPlan,
    BaseQualitativeSampleArchiveSpec,
    BaseRunLayout,
    BaseWorkflowError,
    BaseWorkflowInputError,
    BaseWorkflowResult,
)
from text_to_sign_production.workflows._base.validate import _validate_split
from text_to_sign_production.workflows._shared.archives import (
    build_tar_zstd_create_from_member_list_command,
    build_tar_zstd_list_command,
)
from text_to_sign_production.workflows._shared.metadata import (
    iter_jsonl_records,
    verify_portable_json_file,
    write_json_object,
)
from text_to_sign_production.workflows._shared.shell import shell_quote as _shell_quote
from text_to_sign_production.workflows._shared.transfer import build_byte_progress_copy_command
from text_to_sign_production.workflows.commands import CommandSpec, command


def build_base_publish_plan(
    result: BaseWorkflowResult,
    *,
    drive_project_root: Path | str,
    project_root: Path | str | None = None,
) -> BasePublishPlan:
    """Build direct-file, checkpoint, and sample-archive publish specs."""

    runtime_layout = ProjectLayout(Path(project_root or result.run_layout.project_root))
    drive_layout = ProjectLayout(Path(drive_project_root))
    if result.run_layout.project_root != runtime_layout.root:
        raise BaseWorkflowInputError(
            "Published Base result does not belong to the requested project root: "
            f"{result.run_layout.project_root}"
        )
    runtime_run_root = require_dir(
        runtime_layout.base_m0_run_root(result.run_layout.run_name),
        label="Base runtime run root",
    )
    if result.run_layout.run_root != runtime_run_root:
        raise BaseWorkflowInputError(
            "Published Base result run root does not match the canonical Base M0 run root: "
            f"{result.run_layout.run_root}"
        )
    drive_run_root = drive_layout.base_m0_run_root(result.run_layout.run_name)
    return BasePublishPlan(
        run_name=result.run_layout.run_name,
        runtime_project_root=runtime_layout.root,
        drive_project_root=drive_layout.root,
        runtime_run_root=runtime_run_root,
        drive_run_root=drive_run_root,
        direct_file_specs=_base_direct_file_specs(
            result,
            runtime_layout=runtime_layout,
            drive_layout=drive_layout,
        ),
        checkpoint_specs=_base_checkpoint_publish_specs(
            result.run_layout,
            runtime_layout=runtime_layout,
            drive_layout=drive_layout,
        ),
        checkpoint_manifest_source_path=(
            result.run_layout.checkpoints_dir / BASE_CHECKPOINT_MANIFEST_FILENAME
        ),
        checkpoint_manifest_target_path=drive_run_root
        / "checkpoints"
        / BASE_CHECKPOINT_MANIFEST_FILENAME,
        prediction_sample_archive_specs=_base_prediction_sample_archive_specs(
            result,
            runtime_layout=runtime_layout,
            drive_layout=drive_layout,
        ),
        qualitative_sample_archive_spec=_base_qualitative_sample_archive_spec(
            result,
            runtime_layout=runtime_layout,
            drive_layout=drive_layout,
        ),
    )


def verify_base_file_publish(spec: BaseFilePublishSpec) -> BasePublishedFileSummary:
    """Verify one direct Base file publish."""

    target_path = verify_output_file(spec.target_path, label=spec.label)
    target_size = target_path.stat().st_size
    target_sha256 = sha256_file(target_path)
    if target_size != spec.source_size:
        raise BaseWorkflowError(
            f"{spec.label} byte count mismatch: expected "
            f"{spec.source_size}, observed {target_size}: {target_path}"
        )
    if target_sha256 != spec.source_sha256:
        raise BaseWorkflowError(
            f"{spec.label} sha256 mismatch: expected "
            f"{spec.source_sha256}, observed {target_sha256}: {target_path}"
        )
    return BasePublishedFileSummary(
        label=spec.label,
        group=spec.group,
        split=spec.split,
        source_path=spec.source_path,
        target_path=target_path,
        source_size=spec.source_size,
        target_size=target_size,
        source_sha256=spec.source_sha256,
        target_sha256=target_sha256,
    )


def verify_base_checkpoint_publish(
    spec: BaseCheckpointPublishSpec,
) -> BasePublishedCheckpointSummary:
    """Verify one compressed checkpoint publish."""

    target_path = verify_output_file(spec.target_path, label=f"Base {spec.role} checkpoint publish")
    compressed_size = target_path.stat().st_size
    compressed_sha256 = sha256_file(target_path)
    if compressed_size <= 0:
        raise BaseWorkflowError(f"Compressed Base checkpoint is empty: {target_path}")
    return BasePublishedCheckpointSummary(
        role=spec.role,
        source_path=spec.source_path,
        target_path=target_path,
        source_size=spec.source_size,
        compressed_size=compressed_size,
        source_sha256=spec.source_sha256,
        compressed_sha256=compressed_sha256,
    )


def write_base_checkpoint_manifest(plan: BasePublishPlan) -> BaseFilePublishSpec:
    """Write the runtime checkpoint manifest and return its direct publish spec."""

    checkpoint_entries: dict[str, dict[str, object]] = {}
    for spec in plan.checkpoint_specs:
        summary = verify_base_checkpoint_publish(spec)
        checkpoint_entries[spec.role] = {
            "logical_name": spec.role,
            "epoch": summary_epoch(spec.source_path),
            "metric_name": summary_metric_name(spec.source_path),
            "metric_value": summary_metric_value(spec.source_path),
            "source_runtime_path": repo_relative_path(
                spec.source_path,
                repo_root=plan.runtime_project_root,
            ),
            "published_path": repo_relative_path(
                spec.target_path,
                repo_root=plan.drive_project_root,
            ),
            "source_size": summary.source_size,
            "published_size": summary.compressed_size,
            "source_sha256": summary.source_sha256,
            "published_sha256": summary.compressed_sha256,
            "config_hash": summary_config_hash(spec.source_path),
            "model_id": summary_model_id(spec.source_path),
            "model_revision": summary_model_revision(spec.source_path),
            "run_mode": summary_run_mode(spec.source_path),
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        }
    payload = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "checkpoints": checkpoint_entries,
    }
    write_json_object(plan.checkpoint_manifest_source_path, payload)
    _verify_portable_json_file(
        plan.checkpoint_manifest_source_path,
        project_root=plan.runtime_project_root,
    )
    return _base_file_publish_spec(
        label="checkpoint manifest",
        group="checkpoints",
        split=None,
        source_path=plan.checkpoint_manifest_source_path,
        target_path=plan.checkpoint_manifest_target_path,
    )


def write_base_prediction_sample_archive_member_list(
    spec: BasePredictionSampleArchiveSpec,
) -> Path:
    """Write the expected member list for a prediction sample archive."""

    ensure_dir(spec.member_list_path.parent, label=f"{spec.split} prediction member list directory")
    spec.member_list_path.write_text("\n".join(spec.members) + "\n", encoding="utf-8")
    return verify_output_file(
        spec.member_list_path,
        label=f"{spec.split} prediction sample member list",
    )


def write_base_qualitative_sample_archive_member_list(
    spec: BaseQualitativeSampleArchiveSpec,
) -> Path:
    """Write the expected member list for the qualitative sample archive."""

    ensure_dir(spec.member_list_path.parent, label="Base qualitative member list directory")
    spec.member_list_path.write_text("\n".join(spec.members) + "\n", encoding="utf-8")
    return verify_output_file(spec.member_list_path, label="Base qualitative sample member list")


def verify_base_prediction_sample_archive_file(
    spec: BasePredictionSampleArchiveSpec,
) -> Path:
    """Verify one split prediction sample archive exists and is non-empty."""

    return verify_output_file(spec.target_archive_path, label=f"{spec.split} prediction samples")


def verify_base_prediction_sample_archive_members(
    spec: BasePredictionSampleArchiveSpec,
) -> tuple[str, ...]:
    """Verify one split prediction sample archive member list exactly."""

    return _verify_sample_archive_members(
        label=f"{spec.split} prediction samples",
        observed_member_list_path=spec.observed_member_list_path,
        expected_members=spec.members,
    )


def verify_base_qualitative_sample_archive_file(
    spec: BaseQualitativeSampleArchiveSpec,
) -> Path:
    """Verify the qualitative sample archive exists and is non-empty."""

    return verify_output_file(spec.target_archive_path, label="Base qualitative samples")


def verify_base_qualitative_sample_archive_members(
    spec: BaseQualitativeSampleArchiveSpec,
) -> tuple[str, ...]:
    """Verify the qualitative sample archive member list exactly."""

    return _verify_sample_archive_members(
        label="Base qualitative samples",
        observed_member_list_path=spec.observed_member_list_path,
        expected_members=spec.members,
    )


def _base_direct_file_specs(
    result: BaseWorkflowResult,
    *,
    runtime_layout: ProjectLayout,
    drive_layout: ProjectLayout,
) -> tuple[BaseFilePublishSpec, ...]:
    run_layout = result.run_layout
    files: list[tuple[str, str, str | None, Path]] = [
        ("config snapshot", "config", None, run_layout.config_snapshot_path),
        ("baseline config", "config", None, run_layout.baseline_config_copy_path),
        ("training metrics", "training", None, run_layout.training_metrics_path),
        ("training summary", "training", None, run_layout.training_summary_path),
        ("baseline report JSON", "reports", None, run_layout.baseline_report_json_path),
        (
            "baseline report Markdown",
            "reports",
            None,
            run_layout.baseline_report_markdown_path,
        ),
        ("failure-mode report JSON", "reports", None, run_layout.failure_modes_json_path),
        (
            "failure-mode report Markdown",
            "reports",
            None,
            run_layout.failure_modes_markdown_path,
        ),
        ("run summary", "root", None, run_layout.run_summary_path),
    ]
    for split, prediction in result.predictions.items():
        files.append(
            (f"{split} prediction manifest", "predictions", split, prediction.manifest_path)
        )
    if result.qualitative is not None:
        files.extend(
            [
                (
                    "qualitative records",
                    "qualitative",
                    None,
                    result.qualitative.records_path,
                ),
                (
                    "qualitative panel summary",
                    "qualitative",
                    None,
                    result.qualitative.panel_summary_path,
                ),
                (
                    "qualitative evidence bundle",
                    "qualitative",
                    None,
                    result.qualitative.evidence_bundle_path,
                ),
                (
                    "qualitative panel definition",
                    "qualitative",
                    None,
                    result.qualitative.panel_definition_path,
                ),
            ]
        )
    return tuple(
        _base_file_publish_spec(
            label=label,
            group=group,
            split=split,
            source_path=source_path,
            target_path=_mirrored_drive_run_path(
                source_path,
                runtime_layout=runtime_layout,
                drive_layout=drive_layout,
                run_name=run_layout.run_name,
                label=label,
            ),
        )
        for label, group, split, source_path in files
    )


def _base_file_publish_spec(
    *,
    label: str,
    group: str,
    split: str | None,
    source_path: Path,
    target_path: Path,
) -> BaseFilePublishSpec:
    verified_source_path = verify_output_file(source_path, label=f"Runtime Base {label}")
    source_size = verified_source_path.stat().st_size
    source_sha256 = sha256_file(verified_source_path)
    return BaseFilePublishSpec(
        label=label,
        group=group,
        split=split,
        source_path=verified_source_path,
        target_path=target_path,
        source_size=source_size,
        source_sha256=source_sha256,
        copy_command=_publish_file_command_spec(
            label=f"Base {label}",
            source_path=verified_source_path,
            target_path=target_path,
            expected_input_bytes=source_size,
        ),
    )


def _base_checkpoint_publish_specs(
    run_layout: BaseRunLayout,
    *,
    runtime_layout: ProjectLayout,
    drive_layout: ProjectLayout,
) -> tuple[BaseCheckpointPublishSpec, ...]:
    specs: list[BaseCheckpointPublishSpec] = []
    for role, source_path in (
        ("best", run_layout.best_checkpoint_path),
        ("last", run_layout.last_checkpoint_path),
    ):
        verified_source_path = verify_output_file(
            source_path,
            label=f"Runtime Base {role} checkpoint",
        )
        source_size = verified_source_path.stat().st_size
        source_sha256 = sha256_file(verified_source_path)
        target_path = _mirrored_drive_run_path(
            verified_source_path,
            runtime_layout=runtime_layout,
            drive_layout=drive_layout,
            run_name=run_layout.run_name,
            label=f"{role} checkpoint",
        ).with_suffix(".pt.zst")
        specs.append(
            BaseCheckpointPublishSpec(
                role=role,
                source_path=verified_source_path,
                target_path=target_path,
                source_size=source_size,
                source_sha256=source_sha256,
                compress_command=_compress_checkpoint_command_spec(
                    role=role,
                    source_path=verified_source_path,
                    target_path=target_path,
                    expected_input_bytes=source_size,
                ),
            )
        )
    return tuple(specs)


def _base_prediction_sample_archive_specs(
    result: BaseWorkflowResult,
    *,
    runtime_layout: ProjectLayout,
    drive_layout: ProjectLayout,
) -> tuple[BasePredictionSampleArchiveSpec, ...]:
    specs: list[BasePredictionSampleArchiveSpec] = []
    for split, prediction in result.predictions.items():
        validated_split = _validate_split(split)
        runtime_samples_dir = require_dir(
            result.run_layout.prediction_samples_dir(validated_split),
            label=f"Base {validated_split} prediction samples",
        )
        members = _prediction_sample_members_from_manifest(
            prediction.manifest_path,
            samples_dir=runtime_samples_dir,
            project_root=runtime_layout.root,
            split=validated_split,
        )
        member_list_path = (
            runtime_layout.base_m0_runs_root
            / "_publish"
            / result.run_layout.run_name
            / "predictions"
            / validated_split
            / "samples_members.txt"
        )
        target_archive_path = (
            drive_layout.base_m0_run_root(result.run_layout.run_name)
            / "predictions"
            / validated_split
            / BASE_PREDICTION_SAMPLES_ARCHIVE_FILENAME
        )
        observed_member_list_path = (
            target_archive_path.parent / BASE_PREDICTION_SAMPLES_OBSERVED_MEMBERS_FILENAME
        )
        specs.append(
            BasePredictionSampleArchiveSpec(
                split=validated_split,
                runtime_manifest_path=prediction.manifest_path,
                runtime_samples_dir=runtime_samples_dir,
                member_list_path=member_list_path,
                observed_member_list_path=observed_member_list_path,
                target_archive_path=target_archive_path,
                members=members,
                expected_member_count=len(members),
                archive_command=_sample_archive_create_command_spec(
                    label=f"{validated_split} Base prediction samples",
                    archive_path=target_archive_path,
                    archive_parent=target_archive_path.parent,
                    source_root=runtime_layout.root,
                    member_list_path=member_list_path,
                    expected_member_count=len(members),
                ),
                list_command=_sample_archive_list_command_spec(
                    label=f"{validated_split} Base prediction samples",
                    archive_path=target_archive_path,
                    observed_member_list_path=observed_member_list_path,
                    expected_member_count=len(members),
                ),
            )
        )
    return tuple(specs)


def _base_qualitative_sample_archive_spec(
    result: BaseWorkflowResult,
    *,
    runtime_layout: ProjectLayout,
    drive_layout: ProjectLayout,
) -> BaseQualitativeSampleArchiveSpec | None:
    if result.qualitative is None:
        return None
    runtime_qualitative_dir = require_dir(
        result.qualitative.output_dir,
        label="Base qualitative output directory",
    )
    members = _qualitative_sample_members(
        runtime_qualitative_dir,
        project_root=runtime_layout.root,
    )
    member_list_path = (
        runtime_layout.base_m0_runs_root
        / "_publish"
        / result.run_layout.run_name
        / "qualitative"
        / "panel_samples_members.txt"
    )
    target_archive_path = (
        drive_layout.base_m0_run_root(result.run_layout.run_name)
        / "qualitative"
        / BASE_QUALITATIVE_PANEL_SAMPLES_ARCHIVE_FILENAME
    )
    observed_member_list_path = (
        target_archive_path.parent / BASE_QUALITATIVE_PANEL_SAMPLES_OBSERVED_MEMBERS_FILENAME
    )
    return BaseQualitativeSampleArchiveSpec(
        runtime_qualitative_dir=runtime_qualitative_dir,
        member_list_path=member_list_path,
        observed_member_list_path=observed_member_list_path,
        target_archive_path=target_archive_path,
        members=members,
        expected_member_count=len(members),
        archive_command=_sample_archive_create_command_spec(
            label="Base qualitative samples",
            archive_path=target_archive_path,
            archive_parent=target_archive_path.parent,
            source_root=runtime_layout.root,
            member_list_path=member_list_path,
            expected_member_count=len(members),
        ),
        list_command=_sample_archive_list_command_spec(
            label="Base qualitative samples",
            archive_path=target_archive_path,
            observed_member_list_path=observed_member_list_path,
            expected_member_count=len(members),
        ),
    )


def _mirrored_drive_run_path(
    path: Path,
    *,
    runtime_layout: ProjectLayout,
    drive_layout: ProjectLayout,
    run_name: str,
    label: str,
) -> Path:
    relative_path = Path(repo_relative_path(path, repo_root=runtime_layout.root))
    expected_prefix = Path("runs") / "base" / "m0-direct-text-to-full-bfh" / run_name
    if not relative_path.is_relative_to(expected_prefix):
        raise BaseWorkflowError(
            f"{label} must be published from the canonical Base M0 run tree: {path}"
        )
    return drive_layout.root / relative_path


def _prediction_sample_members_from_manifest(
    manifest_path: Path,
    *,
    samples_dir: Path,
    project_root: Path,
    split: str,
) -> tuple[str, ...]:
    members: list[str] = []
    for record in iter_jsonl_records(manifest_path):
        sample_path = record.get("prediction_sample_path")
        if not isinstance(sample_path, str) or not sample_path.strip():
            raise BaseWorkflowError(
                f"Prediction manifest row is missing prediction_sample_path: {manifest_path}"
            )
        resolved_sample_path = _resolve_manifest_or_repo_path(
            sample_path,
            project_root=project_root,
        )
        if not resolved_sample_path.is_relative_to(samples_dir):
            raise BaseWorkflowError(
                f"{split} prediction sample is outside the split samples directory: "
                f"{resolved_sample_path}"
            )
        if resolved_sample_path.suffix != ".npz":
            raise BaseWorkflowError(
                f"{split} prediction sample archive members must be .npz files: "
                f"{resolved_sample_path}"
            )
        require_file(resolved_sample_path, label=f"{split} prediction sample")
        members.append(repo_relative_path(resolved_sample_path, repo_root=project_root))
    return _validate_base_archive_members(tuple(members), label=f"{split} prediction samples")


def _qualitative_sample_members(
    qualitative_dir: Path,
    *,
    project_root: Path,
) -> tuple[str, ...]:
    members: list[str] = []
    for dirname in (REFERENCE_ARTIFACTS_DIRNAME, PREDICTION_ARTIFACTS_DIRNAME):
        artifact_dir = require_dir(
            qualitative_dir / dirname,
            label=f"Base qualitative {dirname} directory",
        )
        for path in sorted(artifact_dir.glob("*.npz")):
            require_file(path, label=f"Base qualitative {dirname} sample")
            members.append(repo_relative_path(path, repo_root=project_root))
    return _validate_base_archive_members(tuple(members), label="Base qualitative samples")


def _validate_base_archive_members(members: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if not members:
        raise BaseWorkflowError(f"{label} archive must include at least one member.")
    seen_members: set[str] = set()
    invalid_members: list[str] = []
    for member in members:
        if (
            not member
            or member in seen_members
            or member.startswith("/")
            or "\\" in member
            or ".." in member.split("/")
            or not member.startswith("runs/")
        ):
            invalid_members.append(member)
        seen_members.add(member)
    if invalid_members:
        preview = "\n".join(invalid_members[:10])
        raise BaseWorkflowError(
            "Base archive member paths must be unique, project-root-relative, and start with "
            f"'runs/'. Invalid members for {label}:\n{preview}"
        )
    return members


def _verify_sample_archive_members(
    *,
    label: str,
    observed_member_list_path: Path,
    expected_members: tuple[str, ...],
) -> tuple[str, ...]:
    observed_path = require_non_empty_file(
        observed_member_list_path,
        label=f"{label} observed member list",
    )
    observed_members = tuple(
        line.strip()
        for line in observed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if observed_members != expected_members:
        raise BaseWorkflowError(
            _archive_member_mismatch_message(label, expected_members, observed_members)
        )
    return observed_members


def _checkpoint_payload(path: Path) -> dict[str, object]:
    return load_training_checkpoint(path, map_location="cpu")


def summary_epoch(path: Path) -> int | None:
    value = _checkpoint_payload(path).get("epoch")
    return int(value) if isinstance(value, int) else None


def summary_metric_name(path: Path) -> str | None:
    value = _checkpoint_payload(path).get("metric_name")
    return value if isinstance(value, str) else None


def summary_metric_value(path: Path) -> float | None:
    value = _checkpoint_payload(path).get("metric_value")
    return float(value) if isinstance(value, int | float) else None


def summary_config_hash(path: Path) -> str | None:
    value = _checkpoint_payload(path).get("config_hash")
    return value if isinstance(value, str) else None


def summary_model_id(path: Path) -> str | None:
    value = _checkpoint_payload(path).get("backbone_name")
    return value if isinstance(value, str) else None


def summary_model_revision(path: Path) -> str | None:
    value = _checkpoint_payload(path).get("model_revision")
    return value if isinstance(value, str) else None


def summary_run_mode(path: Path) -> str | None:
    value = _checkpoint_payload(path).get("run_mode")
    return value if isinstance(value, str) else None


def _archive_member_mismatch_message(
    label: str,
    expected_members: tuple[str, ...],
    observed_members: tuple[str, ...],
) -> str:
    mismatch_count = sum(
        1
        for expected, observed in zip(expected_members, observed_members, strict=False)
        if expected != observed
    ) + abs(len(expected_members) - len(observed_members))
    return (
        f"{label} archive members do not match the workflow spec. "
        f"expected_count={len(expected_members)} observed_count={len(observed_members)} "
        f"mismatch_count={mismatch_count} "
        f"expected_first5={list(expected_members[:5])} "
        f"expected_last5={list(expected_members[-5:])} "
        f"observed_first5={list(observed_members[:5])} "
        f"observed_last5={list(observed_members[-5:])}."
    )


def _publish_file_command_spec(
    *,
    label: str,
    source_path: Path,
    target_path: Path,
    expected_input_bytes: int,
) -> CommandSpec:
    return build_byte_progress_copy_command(
        label=label,
        source_path=source_path,
        target_path=target_path,
        expected_input_bytes=expected_input_bytes,
        failure=f"Failed to publish {label}: {source_path} -> {target_path}",
    )


def _compress_checkpoint_command_spec(
    *,
    role: str,
    source_path: Path,
    target_path: Path,
    expected_input_bytes: int,
) -> CommandSpec:
    label = f"Base {role} checkpoint"
    bash = (
        f"mkdir -p {_shell_quote(target_path.parent)} && "
        f"tqdm --bytes --total {expected_input_bytes} "
        f"< {_shell_quote(source_path)} "
        f"| zstd -f -q -o {_shell_quote(target_path)} -"
    )
    return command(
        label=label,
        bash=bash,
        failure=f"Failed to publish compressed {label}: {source_path} -> {target_path}",
    )


def _sample_archive_create_command_spec(
    *,
    label: str,
    archive_path: Path,
    archive_parent: Path,
    source_root: Path,
    member_list_path: Path,
    expected_member_count: int,
) -> CommandSpec:
    return build_tar_zstd_create_from_member_list_command(
        label=label,
        archive_path=archive_path,
        archive_parent=archive_parent,
        source_root=source_root,
        member_list_path=member_list_path,
        expected_member_count=expected_member_count,
        failure=f"Failed to create {label} archive: {archive_path}",
    )


def _sample_archive_list_command_spec(
    *,
    label: str,
    archive_path: Path,
    observed_member_list_path: Path,
    expected_member_count: int,
) -> CommandSpec:
    return build_tar_zstd_list_command(
        label=label,
        archive_path=archive_path,
        observed_member_list_path=observed_member_list_path,
        expected_member_count=expected_member_count,
        failure=f"Failed to list {label} archive members: {archive_path}",
    )


def _verify_portable_json_file(path: Path, *, project_root: Path) -> None:
    verify_portable_json_file(path, project_root=project_root, error_factory=BaseWorkflowError)


__all__ = [
    "build_base_publish_plan",
    "write_base_checkpoint_manifest",
    "write_base_prediction_sample_archive_member_list",
    "write_base_qualitative_sample_archive_member_list",
]
