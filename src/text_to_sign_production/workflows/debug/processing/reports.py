from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.workflows.debug.constants import DEBUG_STAGE_REPORT_WRITE
from text_to_sign_production.workflows.debug.contracts import (
    DebugGateResult,
    DebugReportResult,
    DebugSampleDossier,
    DebugTierResult,
    DebugVisualizationResult,
)
from text_to_sign_production.workflows.debug.layout import DebugLayout, debug_sample_run_root
from text_to_sign_production.workflows.debug.progress import debug_progress_stage


def write_debug_reports(
    layout: DebugLayout,
    *,
    run_id: str,
    dossier: DebugSampleDossier,
    gate_result: DebugGateResult,
    tier_result: DebugTierResult,
    visual_result: DebugVisualizationResult,
    progress_session: ProgressSession | None = None,
) -> DebugReportResult:
    output_root = debug_sample_run_root(
        layout,
        sentence_name=dossier.translation.sentence_name,
        run_id=run_id,
    )
    files: list[Path] = []
    errors: list[str] = []
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        payload = _debug_result_payload(dossier, gate_result, tier_result, visual_result, run_id)
        writes = (
            lambda: _write_text(output_root / "debug_summary.md", _summary_markdown(payload)),
            lambda: _write_json(output_root / "debug_result.json", payload),
            lambda: _write_json(output_root / "gate_comparison.json", _gate_payload(gate_result)),
            lambda: _write_json(output_root / "tier_comparison.json", _tier_payload(tier_result)),
            lambda: _write_json(
                output_root / "visual_artifacts.json",
                _visual_payload(visual_result),
            ),
        )
        if progress_session is not None:
            with progress_session.task(_report_write_progress_spec(), total=len(writes)) as task:
                for write in writes:
                    files.append(write())
                    task.advance()
        else:
            for write in writes:
                files.append(write())
    except (OSError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return DebugReportResult(
        output_root=output_root,
        run_id=run_id,
        files=tuple(files),
        succeeded=not errors,
        errors=tuple(errors),
        warnings=(),
    )


def _debug_result_payload(
    dossier: DebugSampleDossier,
    gate_result: DebugGateResult,
    tier_result: DebugTierResult,
    visual_result: DebugVisualizationResult,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "debug_sample.v1",
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "target": _translation_payload(dossier),
        "source": {
            "keypoint_json_dir": dossier.source.keypoint_json_dir,
            "keypoint_json_dir_exists": dossier.source.keypoint_json_dir_exists,
            "keypoint_frame_json_count": dossier.source.keypoint_frame_json_count,
            "raw_video_path": dossier.source.raw_video_path,
            "raw_video_exists": dossier.source.raw_video_exists,
            "source_issues": dossier.source.source_issues,
        },
        "existing_gate_status": dossier.existing_gate.status,
        "existing_tier_states": {
            tier.tier.value: tier.state.value for tier in dossier.existing_tiers
        },
        "gate": _gate_payload(gate_result),
        "tier": _tier_payload(tier_result),
        "visualization": _visual_payload(visual_result),
        "consistency_notes": dossier.consistency_notes,
    }


def _translation_payload(dossier: DebugSampleDossier) -> dict[str, Any]:
    row = dossier.translation
    return {
        "split": row.split,
        "video_id": row.video_id,
        "video_name": row.video_name,
        "sentence_id": row.sentence_id,
        "sentence_name": row.sentence_name,
        "start_realigned": row.start_realigned,
        "end_realigned": row.end_realigned,
        "sentence": row.sentence,
        "translation_path": row.translation_path,
        "row_index": row.row_index,
    }


def _gate_payload(result: DebugGateResult) -> dict[str, Any]:
    gate = result.computed_gate
    return {
        "verdict": result.verdict,
        "computed_status": result.computed_status,
        "terminal_gate": None if gate is None or gate.terminal_gate is None else gate.terminal_gate,
        "failed_gates": () if gate is None else gate.failed_gates,
        "decision_issue_codes": ()
        if gate is None
        else tuple(code for decision in gate.decisions for code in decision.issue_codes),
        "comparison": result.comparison,
        "errors": result.errors,
        "warnings": result.warnings,
    }


def _tier_payload(result: DebugTierResult) -> dict[str, Any]:
    decision = result.computed_decision
    leakage = result.leakage_summary
    return {
        "verdict": result.verdict,
        "applicable": result.applicable,
        "applicability_reason": result.applicability_reason,
        "context": result.leakage_context,
        "computed_status": None if decision is None else decision.status,
        "selected_tier": None if decision is None else decision.selected_tier,
        "issue_codes": () if decision is None else tuple(issue.code for issue in decision.issues),
        "family_decisions": result.family_decisions,
        "quality_metrics": result.quality_metrics,
        "leakage": None
        if leakage is None
        else {
            "has_leakage": leakage.has_leakage,
            "max_severity": leakage.max_severity,
            "same_source_sentence_match_count": leakage.same_source_sentence_match_count,
            "exact_text_match_count": leakage.exact_text_match_count,
            "same_source_video_match_count": leakage.same_source_video_match_count,
            "matched_samples": leakage.matched_samples,
        },
        "comparison": result.comparison,
        "errors": result.errors,
        "warnings": result.warnings,
    }


def _visual_payload(result: DebugVisualizationResult) -> dict[str, Any]:
    return {
        "verdict": result.verdict,
        "output_root": result.output_root,
        "artifacts": result.artifacts,
        "errors": result.errors,
        "warnings": result.warnings,
    }


def _summary_markdown(payload: dict[str, Any]) -> str:
    target = payload["target"]
    gate = payload["gate"]
    tier = payload["tier"]
    visual = payload["visualization"]
    return "\n".join(
        (
            "# Single Sample Debug Summary",
            "",
            f"- SENTENCE_NAME: {target['sentence_name']}",
            f"- Resolved split: {_json_value(target['split'])}",
            f"- Gate verdict: {_json_value(gate['verdict'])}",
            f"- Tier verdict: {_json_value(tier['verdict'])}",
            f"- Visualization verdict: {_json_value(visual['verdict'])}",
            "",
            "## Notes",
            *[f"- {note}" for note in payload["consistency_notes"]],
            "",
        )
    )


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(_json_value(key)): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _report_write_progress_spec() -> ProgressStageSpec:
    return debug_progress_stage(
        stage_id=DEBUG_STAGE_REPORT_WRITE,
        label="debug report write",
        unit="file",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="report_write",
        total_semantics="debug report projection files written",
    )


__all__ = ["write_debug_reports"]
