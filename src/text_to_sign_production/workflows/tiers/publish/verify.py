from __future__ import annotations

from text_to_sign_production.workflows.tiers.contracts import (
    TiersPublishCheck,
    TiersPublishPlan,
    TiersPublishVerification,
)


def verify_tiers_publish(
    plan: TiersPublishPlan,
) -> TiersPublishVerification:
    return TiersPublishVerification(
        checks=tuple(
            TiersPublishCheck(
                label=target.label,
                target_path=target.target_path,
                exists=target.target_path.exists(),
            )
            for target in plan.targets
        ),
    )
