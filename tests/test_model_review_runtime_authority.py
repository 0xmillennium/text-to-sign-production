from __future__ import annotations

from text_to_sign_production.workflows.model.processing.reports import (
    _performance_summary_markdown,
)
from text_to_sign_production.workflows.model.review.sections import (
    _active_provider_performance_review_fields,
)


def test_review_active_fields_come_from_compute_profile_application() -> None:
    fields = _active_provider_performance_review_fields(
        "learned_pose_token",
        {
            "training": {"tokenizer_batch_size": 999},
            "compute_profile_application": {
                "active_applied": {"tokenizer_batch_size": 512},
                "dataloader_applied": {"num_workers": 4},
            },
        },
    )
    labels = [label for label, _value in fields]

    assert "active.tokenizer_batch_size" in labels
    assert "dataloader.num_workers" in labels


def test_review_does_not_show_materialization_workers_as_active() -> None:
    fields = _active_provider_performance_review_fields(
        "learned_pose_token",
        {
            "training": {"materialization_workers": 4},
            "compute_profile_application": {
                "active_applied": {},
                "dataloader_applied": {"num_workers": 4},
            },
        },
    )
    labels = [label for label, _value in fields]

    assert all("materialization_workers" not in label for label in labels)


def test_performance_markdown_displays_provider_specific_telemetry() -> None:
    markdown = _performance_summary_markdown(
        {
            "device": {},
            "slowest_stage": {},
            "stages": [
                {
                    "provider_stage_id": "latent_diffusion.train_denoiser",
                    "peak_cuda_memory_reserved_fraction": 0.1,
                    "gpu_utilization_class": "moderate",
                    "bottleneck_hint": "none",
                    "denoiser_batch_size": 128,
                },
                {
                    "provider_stage_id": "articulator_aware.train_structure_aware",
                    "peak_cuda_memory_reserved_fraction": 0.1,
                    "gpu_utilization_class": "moderate",
                    "bottleneck_hint": "none",
                    "source_batch_size": 64,
                    "frame_batch_size": 2048,
                },
            ],
        }
    )

    assert "denoiser_batch_size=128" in markdown
    assert "source_batch_size=64" in markdown
    assert "frame_batch_size=2048" in markdown
