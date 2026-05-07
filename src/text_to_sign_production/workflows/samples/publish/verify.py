from __future__ import annotations

from text_to_sign_production.workflows.samples.contracts import (
    SamplesPublishCheck,
    SamplesPublishPlan,
    SamplesPublishVerification,
)


def verify_samples_publish(
    plan: SamplesPublishPlan,
) -> SamplesPublishVerification:
    return SamplesPublishVerification(
        checks=tuple(
            SamplesPublishCheck(
                label=target.label,
                target_path=target.target_path,
                exists=target.target_path.exists(),
            )
            for target in plan.targets
        )
    )
