"""Reports for evaluation protocol planning and claim readiness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.modeling.evaluation.config import EvaluationProtocolConfig
from text_to_sign_production.modeling.evaluation.contrasts import EvaluationContrastReadiness
from text_to_sign_production.modeling.evaluation.inventory import (
    EvaluationSurfaceInventory,
    build_planned_run_requests,
)
from text_to_sign_production.modeling.evaluation.io import write_json, write_jsonl, write_text


@dataclass(frozen=True, slots=True)
class EvaluationProtocolReportPaths:
    plan_json_path: Path
    plan_markdown_path: Path
    surface_inventory_jsonl_path: Path
    surface_inventory_markdown_path: Path
    missing_outputs_report_path: Path
    planned_run_requests_path: Path
    contrast_readiness_path: Path
    ablation_readiness_report_path: Path
    claim_readiness_summary_path: Path
    claim_readiness_report_path: Path
    evaluation_protocol_summary_path: Path


def write_evaluation_protocol_reports(
    *,
    report_root: Path,
    config: EvaluationProtocolConfig,
    inventory: EvaluationSurfaceInventory,
    planned_run_requests: tuple[dict[str, object], ...],
    readiness: tuple[EvaluationContrastReadiness, ...],
    validation_engine_result_path: Path | None,
) -> EvaluationProtocolReportPaths:
    report_root.mkdir(parents=True, exist_ok=True)
    paths = EvaluationProtocolReportPaths(
        plan_json_path=report_root / "evaluation_protocol_plan.json",
        plan_markdown_path=report_root / "evaluation_protocol_plan.md",
        surface_inventory_jsonl_path=report_root / "surface_inventory.jsonl",
        surface_inventory_markdown_path=report_root / "surface_inventory.md",
        missing_outputs_report_path=report_root / "missing_outputs_report.md",
        planned_run_requests_path=report_root / "planned_run_requests.jsonl",
        contrast_readiness_path=report_root / "contrast_readiness.jsonl",
        ablation_readiness_report_path=report_root / "ablation_readiness_report.md",
        claim_readiness_summary_path=report_root / "claim_readiness_summary.json",
        claim_readiness_report_path=report_root / "claim_readiness_report.md",
        evaluation_protocol_summary_path=report_root / "evaluation_protocol_summary.md",
    )
    write_json(paths.plan_json_path, _plan_document(config))
    write_text(paths.plan_markdown_path, _plan_markdown(config))
    write_jsonl(paths.surface_inventory_jsonl_path, (entry.to_dict() for entry in inventory.entries))
    write_text(paths.surface_inventory_markdown_path, _surface_inventory_markdown(inventory))
    write_text(paths.missing_outputs_report_path, _missing_outputs_markdown(inventory, planned_run_requests))
    write_jsonl(paths.planned_run_requests_path, planned_run_requests)
    write_jsonl(paths.contrast_readiness_path, (entry.to_dict() for entry in readiness))
    write_text(paths.ablation_readiness_report_path, _ablation_readiness_markdown(readiness))
    write_json(
        paths.claim_readiness_summary_path,
        _claim_summary(
            inventory=inventory,
            readiness=readiness,
            validation_engine_result_path=validation_engine_result_path,
        ),
    )
    write_text(paths.claim_readiness_report_path, _claim_readiness_markdown(readiness))
    write_text(
        paths.evaluation_protocol_summary_path,
        _summary_markdown(
            inventory=inventory,
            readiness=readiness,
            planned_run_requests=planned_run_requests,
            validation_engine_result_path=validation_engine_result_path,
        ),
    )
    return paths


def _plan_document(config: EvaluationProtocolConfig) -> dict[str, object]:
    return {
        "schema_version": config.identity.schema_version,
        "name": config.identity.name,
        "split": config.split.value,
        "manifest_family": (
            None if config.identity.manifest_family is None else config.identity.manifest_family.family_id
        ),
        "variants": [
            {
                "variant_id": variant.variant_id,
                "label": variant.label,
                "producer_type": variant.producer_type,
                "producer_key": variant.producer_key,
                "config_path": variant.config_path,
                "run_name": variant.run_name,
                "auxiliary_objectives": list(variant.auxiliary_objectives),
                "objective_config_paths": dict(variant.objective_config_paths),
                "tags": list(variant.tags),
                "required": variant.required,
            }
            for variant in config.variants
        ],
        "contrasts": [
            {
                "contrast_id": contrast.contrast_id,
                "contrast_type": contrast.contrast_type,
                "control_variant_id": contrast.control_variant_id,
                "treatment_variant_id": contrast.treatment_variant_id,
                "intended_difference": contrast.intended_difference,
                "required": contrast.required,
            }
            for contrast in config.contrasts
        ],
    }


def _plan_markdown(config: EvaluationProtocolConfig) -> str:
    variant_rows = "\n".join(
        f"| {v.variant_id} | {v.producer_type} | {v.producer_key} | {v.run_name} | {v.required} |"
        for v in config.variants
    )
    contrast_rows = "\n".join(
        f"| {c.contrast_id} | {c.contrast_type} | {c.control_variant_id} | {c.treatment_variant_id} | {c.intended_difference} |"
        for c in config.contrasts
    )
    return (
        "# Evaluation Protocol Plan\n\n"
        "## Variants\n\n"
        "| Variant | Producer type | Producer key | Run | Required |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{variant_rows}\n\n"
        "## Contrasts\n\n"
        "| Contrast | Type | Control | Treatment | Intended difference |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{contrast_rows}\n\n"
        + _caveats()
    )


def _surface_inventory_markdown(inventory: EvaluationSurfaceInventory) -> str:
    rows = "\n".join(
        "| "
        + " | ".join(
            (
                entry.variant_id,
                entry.producer_type,
                entry.producer_key,
                entry.run_name,
                entry.status,
                str(entry.entry_count),
                str(entry.success_count),
                str(entry.failure_count),
            )
        )
        + " |"
        for entry in inventory.entries
    )
    return (
        "# Surface Inventory\n\n"
        "| Variant | Producer type | Producer key | Run | Status | Entries | Successes | Failures |\n"
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |\n"
        f"{rows}\n\n"
        + _caveats()
    )


def _missing_outputs_markdown(
    inventory: EvaluationSurfaceInventory,
    planned_run_requests: tuple[dict[str, object], ...],
) -> str:
    missing = [entry for entry in inventory.entries if entry.status != "ready"]
    if not missing:
        body = "No missing or degraded generated-pose surfaces were found.\n"
    else:
        body = "\n".join(
            f"- `{entry.variant_id}`: status=`{entry.status}`, reason={entry.missing_reason!r}, manifest=`{entry.manifest_path}`"
            for entry in missing
        )
        body += "\n"
    return (
        "# Missing Outputs Report\n\n"
        f"{body}\n"
        f"Planned run request count: `{len(planned_run_requests)}`.\n\n"
        "Missing surfaces make claims not ready; this report records requests only and does not execute them.\n"
    )


def _ablation_readiness_markdown(readiness: tuple[EvaluationContrastReadiness, ...]) -> str:
    rows = "\n".join(
        f"| {entry.contrast_id} | {entry.contrast_type} | {entry.readiness_status} | {entry.claim_ready} | {entry.shared_pairing_count} | {json.dumps(entry.protocol_violations)} |"
        for entry in readiness
        if entry.contrast_type == "ablation"
    )
    insufficient = _insufficient_subset_note(readiness)
    return (
        "# Ablation Readiness Report\n\n"
        "| Contrast | Type | Status | Claim ready | Contrast-specific shared paired count | Protocol violations |\n"
        "| --- | --- | --- | --- | ---: | --- |\n"
        f"{rows}\n\n"
        f"{insufficient}"
        "Semantic objective contribution requires a strict with/without-objective ablation.\n"
    )


def _claim_readiness_markdown(readiness: tuple[EvaluationContrastReadiness, ...]) -> str:
    rows = "\n".join(
        f"| {entry.contrast_id} | {entry.contrast_type} | {entry.readiness_status} | {entry.claim_ready} | {entry.shared_pairing_count} |"
        for entry in readiness
    )
    insufficient = _insufficient_subset_note(readiness)
    return (
        "# Claim Readiness Report\n\n"
        "| Contrast | Type | Status | Claim ready | Contrast-specific shared paired count |\n"
        "| --- | --- | --- | --- | ---: |\n"
        f"{rows}\n\n"
        f"{insufficient}"
        "Retrieval baseline comparisons are counter-baselines, not ablations. Retrieval uses real source pose reuse, so retrieval realism is not learned generation.\n\n"
        + _caveats()
    )


def _summary_markdown(
    *,
    inventory: EvaluationSurfaceInventory,
    readiness: tuple[EvaluationContrastReadiness, ...],
    planned_run_requests: tuple[dict[str, object], ...],
    validation_engine_result_path: Path | None,
) -> str:
    return (
        "# Evaluation Protocol Summary\n\n"
        f"- Ready variant count: `{len(inventory.ready_entries)}`\n"
        f"- Missing variant count: `{len(inventory.missing_entries)}`\n"
        f"- Planned run request count: `{len(planned_run_requests)}`\n"
        f"- Claim-ready contrast count: `{sum(entry.claim_ready for entry in readiness)}`\n"
        f"- Validation engine comparison: `{validation_engine_result_path}`\n\n"
        "Reports may identify `best_by_metric`; they must not make model-level or sign intelligibility claims.\n\n"
        + _caveats()
    )


def _claim_summary(
    *,
    inventory: EvaluationSurfaceInventory,
    readiness: tuple[EvaluationContrastReadiness, ...],
    validation_engine_result_path: Path | None,
) -> dict[str, object]:
    return {
        "ready_variant_count": len(inventory.ready_entries),
        "missing_variant_count": len(inventory.missing_entries),
        "claim_ready_count": sum(entry.claim_ready for entry in readiness),
        "validation_engine_result_path": None
        if validation_engine_result_path is None
        else str(validation_engine_result_path),
        "readiness": [entry.to_dict() for entry in readiness],
        "caveats": [
            "Validation split is not final test performance.",
            "Automatic pose metrics do not prove sign intelligibility.",
            "Semantic objective contribution requires with/without-objective ablation.",
            "Retrieval baseline uses real source pose reuse.",
            "Missing surfaces make claims not ready.",
            "Contrast-specific shared paired subset improves fairness but may reduce sample count.",
        ],
    }


def _insufficient_subset_note(readiness: tuple[EvaluationContrastReadiness, ...]) -> str:
    if any(
        entry.readiness_status == "not_ready_insufficient_shared_subset" for entry in readiness
    ):
        return (
            "Contrast-specific shared subset count is below minimum_shared_paired_count "
            "for contrasts marked `not_ready_insufficient_shared_subset`.\n\n"
        )
    return ""


def _caveats() -> str:
    return (
        "## Caveats\n\n"
        "- Validation split is not final test performance.\n"
        "- Automatic pose metrics do not prove sign intelligibility.\n"
        "- Semantic objective contribution requires with/without-objective ablation.\n"
        "- Retrieval baseline uses real source pose reuse.\n"
        "- Missing surfaces make claims not ready.\n"
        "- Contrast-specific shared paired subset improves fairness but may reduce sample count.\n"
    )


__all__ = ["EvaluationProtocolReportPaths", "write_evaluation_protocol_reports"]
