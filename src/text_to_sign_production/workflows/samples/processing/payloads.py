from __future__ import annotations

from text_to_sign_production.artifacts.store import sample_manifest_relative_path
from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.dataset.payloads import write_prepared_sample_payload
from text_to_sign_production.workflows.samples.layout import SamplesLayout
from text_to_sign_production.workflows.samples.processing.models import SamplesPayloadOutput


def write_prepared_sample_workflow_payload(
    *,
    sample: PreparedSample,
    status: SampleStatus,
    layout: SamplesLayout,
) -> SamplesPayloadOutput:
    """Persist a PreparedSample payload and return workflow bookkeeping."""
    payload_ref = sample_manifest_relative_path(
        status,
        sample.source.split,
        sample.source.sample_id,
    ).as_posix()
    payload_path = layout.outputs.root / payload_ref
    write_prepared_sample_payload(payload_path, sample)
    return SamplesPayloadOutput(
        sample=sample,
        path=payload_path,
        payload_ref=payload_ref,
        status=status,
    )


__all__ = ["write_prepared_sample_workflow_payload"]
