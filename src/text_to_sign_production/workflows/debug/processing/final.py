from __future__ import annotations

from text_to_sign_production.core.ids import TierName
from text_to_sign_production.workflows.debug.contracts import (
    DebugFinalResult,
    DebugFinalTierExistingSummary,
    DebugGateResult,
    DebugPublishResult,
    DebugPublishVerification,
    DebugReportResult,
    DebugSampleDossier,
    DebugTierResult,
    DebugVisualizationResult,
)
from text_to_sign_production.workflows.debug.contracts.verdicts import (
    DebugFinalVerdict,
    DebugGateVerdict,
    DebugPublishVerificationStatus,
    DebugTierVerdict,
    DebugVisualizationVerdict,
)


def build_final_result(
    *,
    dossier: DebugSampleDossier,
    gate_result: DebugGateResult,
    tier_result: DebugTierResult,
    visual_result: DebugVisualizationResult,
    report_result: DebugReportResult,
    publish_result: DebugPublishResult | None = None,
    publish_verification: DebugPublishVerification | None = None,
) -> DebugFinalResult:
    final_verdict = _final_verdict(
        gate_result,
        tier_result,
        visual_result,
        report_result,
        publish_verification,
    )
    return DebugFinalResult(
        target_sentence_name=dossier.translation.sentence_name,
        resolved_split=dossier.translation.split,
        debug_splits=dossier.resolution.request.debug_splits,
        source_translation_found=True,
        source_keypoints_found=dossier.source.keypoint_json_dir_exists,
        source_video_found=dossier.source.raw_video_exists,
        gate_computed=gate_result.computed_status,
        gate_existing=dossier.existing_gate.status,
        gate_verdict=gate_result.verdict,
        tier_context_mode=tier_result.leakage_context.mode,
        tier_context_production_like=tier_result.leakage_context.production_like,
        tier_computed_status=None
        if tier_result.computed_decision is None
        else tier_result.computed_decision.status,
        tier_selected_tier=None
        if tier_result.computed_decision is None
        else tier_result.computed_decision.selected_tier,
        tier_existing=tuple(
            DebugFinalTierExistingSummary(tier=tier.tier, state=tier.state)
            for tier in dossier.existing_tiers
        ),
        tier_verdict=tier_result.verdict,
        visualization_verdict=visual_result.verdict,
        visual_artifact_count=len(visual_result.artifacts),
        report_output_path=report_result.output_root if report_result.succeeded else None,
        runtime_output_root=report_result.output_root,
        drive_output_root=None
        if publish_verification is None
        else publish_verification.drive_output_root,
        publish_status=None if publish_verification is None else publish_verification.status,
        published_file_count=0 if publish_result is None else publish_result.published_file_count,
        final_verdict=final_verdict,
        next_actions=_next_actions(final_verdict, tier_result, publish_verification),
        warnings=(
            *gate_result.warnings,
            *tier_result.warnings,
            *visual_result.warnings,
            *report_result.warnings,
            *report_result.errors,
            *(() if publish_result is None else tuple(issue.message for issue in publish_result.warnings)),
            *(
                ()
                if publish_result is None
                else tuple(issue.message for issue in publish_result.blocking_errors)
            ),
            *(
                ()
                if publish_verification is None
                else tuple(issue.message for issue in publish_verification.warnings)
            ),
        ),
    )


def print_final_result(result: DebugFinalResult) -> None:
    print("=" * 60)
    print("SINGLE SAMPLE DEBUG RESULT")
    print("=" * 60)
    print()
    print("Target:")
    print(f"  SENTENCE_NAME: {result.target_sentence_name}")
    print(f"  Resolved split: {result.resolved_split.value if result.resolved_split else 'none'}")
    print(f"  DEBUG_SPLITS: {', '.join(split.value for split in result.debug_splits)}")
    print()
    print("Source:")
    print(f"  Translation: {'FOUND' if result.source_translation_found else 'MISSING'}")
    print(f"  BFH keypoints: {'FOUND' if result.source_keypoints_found else 'MISSING'}")
    print(f"  Raw video: {'FOUND' if result.source_video_found else 'MISSING'}")
    print()
    print("Gate:")
    print(f"  Computed: {result.gate_computed.value if result.gate_computed else 'unavailable'}")
    print(f"  Existing: {result.gate_existing.value}")
    print(f"  Verdict: {result.gate_verdict.value.upper()}")
    print()
    print("Tier:")
    print(f"  Context: {result.tier_context_mode.value}")
    print(f"  Production-like leakage: {'yes' if result.tier_context_production_like else 'no'}")
    print(
        "  Computed: "
        f"{_tier_computed_label(result.tier_computed_status, result.tier_selected_tier)}"
    )
    print(f"  Existing: {_tier_existing_label(result.tier_existing)}")
    print(f"  Verdict: {result.tier_verdict.value.upper()}")
    print()
    print("Visualization:")
    print(f"  Verdict: {result.visualization_verdict.value.upper()}")
    print(f"  Artifact count: {result.visual_artifact_count}")
    print()
    print("Reports:")
    print(
        "  Runtime output: "
        f"{result.runtime_output_root if result.runtime_output_root else 'unavailable'}"
    )
    print(
        "  Drive output: "
        f"{result.drive_output_root if result.drive_output_root else 'not published'}"
    )
    print(f"  Published files: {result.published_file_count}")
    print(
        "  Publish status: "
        f"{result.publish_status.value if result.publish_status else 'not_run'}"
    )
    print()
    print("Final verdict:")
    print(f"  {result.final_verdict.value.upper()}")
    print()
    print("Next action:")
    for action in result.next_actions:
        print(f"  {action}")
    print("=" * 60)


def _tier_computed_label(status, selected_tier: TierName | None) -> str:
    if status is None:
        return "unavailable"
    if selected_tier is None:
        return "excluded"
    return selected_tier.value


def _tier_existing_label(existing: tuple[DebugFinalTierExistingSummary, ...]) -> str:
    if not existing:
        return "unavailable"
    return ", ".join(f"{item.tier.value}:{item.state.value}" for item in existing)


def _final_verdict(
    gate_result: DebugGateResult,
    tier_result: DebugTierResult,
    visual_result: DebugVisualizationResult,
    report_result: DebugReportResult,
    publish_verification: DebugPublishVerification | None,
) -> DebugFinalVerdict:
    if gate_result.verdict is DebugGateVerdict.MISMATCH:
        return DebugFinalVerdict.FAIL_GATE_MISMATCH
    if tier_result.verdict is DebugTierVerdict.MISMATCH:
        return DebugFinalVerdict.FAIL_TIER_MISMATCH
    if gate_result.verdict in {
        DebugGateVerdict.COMPUTE_FAILED,
        DebugGateVerdict.EXISTING_MISSING,
        DebugGateVerdict.EXISTING_AMBIGUOUS,
    }:
        return DebugFinalVerdict.PARTIAL_DEBUG
    if tier_result.verdict is DebugTierVerdict.CONTEXT_LIMITED_DIFFERENCE:
        return DebugFinalVerdict.PASS_WITH_CONTEXT_WARNING
    if tier_result.verdict in {
        DebugTierVerdict.COMPUTE_FAILED,
        DebugTierVerdict.EXISTING_MISSING,
        DebugTierVerdict.EXISTING_AMBIGUOUS,
    }:
        return DebugFinalVerdict.PARTIAL_DEBUG
    if visual_result.verdict is DebugVisualizationVerdict.FAILED:
        return DebugFinalVerdict.PASS_WITH_VISUALIZATION_WARNING
    if not report_result.succeeded:
        return DebugFinalVerdict.PASS_WITH_REPORT_WARNING
    if publish_verification is None:
        return DebugFinalVerdict.PASS_WITH_PUBLISH_WARNING
    if publish_verification.status is not DebugPublishVerificationStatus.VERIFIED:
        return DebugFinalVerdict.PASS_WITH_PUBLISH_WARNING
    if visual_result.verdict in {
        DebugVisualizationVerdict.PARTIAL,
        DebugVisualizationVerdict.SKIPPED,
    }:
        return DebugFinalVerdict.PASS_WITH_VISUALIZATION_WARNING
    return DebugFinalVerdict.PASS


def _next_actions(
    verdict: DebugFinalVerdict,
    tier_result: DebugTierResult,
    publish_verification: DebugPublishVerification | None,
) -> tuple[str, ...]:
    if verdict is DebugFinalVerdict.PASS_WITH_CONTEXT_WARNING:
        return (
            "This sample satisfies tier criteria in the selected debug context.",
            "However, the existing tier output excludes it.",
            "Because DEBUG_SPLITS does not include the full train/val/test context, cross-split leakage may not be observable.",
            'Rerun with: DEBUG_SPLITS = ("train", "val", "test") to verify leakage-driven exclusion.',
        )
    if verdict is DebugFinalVerdict.FAIL_GATE_MISMATCH:
        return ("Inspect gate comparison details before changing production artifacts.",)
    if verdict is DebugFinalVerdict.FAIL_TIER_MISMATCH:
        return ("Inspect tier comparison details with full train/val/test leakage context.",)
    if (
        verdict is DebugFinalVerdict.PARTIAL_DEBUG
        and tier_result.verdict is DebugTierVerdict.NOT_APPLICABLE
    ):
        return (
            "Tier debug was skipped because the production tier pipeline only processes passed samples.",
        )
    if verdict is DebugFinalVerdict.PASS:
        return ("No immediate action required from this debug pass.",)
    if verdict is DebugFinalVerdict.PASS_WITH_PUBLISH_WARNING:
        if publish_verification is None:
            return ("Publish debug outputs to Drive before sharing this debug pass.",)
        return ("Inspect publish review sections and rerun publish after fixing Drive output availability.",)
    return ("Review section warnings and missing inputs, then rerun the notebook if needed.",)


__all__ = ["build_final_result", "print_final_result"]
