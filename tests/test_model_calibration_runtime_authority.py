from __future__ import annotations

from pathlib import Path

from text_to_sign_production.modeling.candidates import ModelRunMode, ModelRunRequest
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDiffusionProvider,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)
from text_to_sign_production.workflows.model.processing.calibration import (
    run_compute_calibration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_run_compute_calibration_does_not_read_raw_profile_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("raw profile candidates must not be read")

    monkeypatch.setattr(
        "text_to_sign_production.workflows.model.processing.calibration.provider_candidate_overrides",
        forbidden,
    )
    provider = LatentDiffusionProvider()
    request = ModelRunRequest(
        model_key=ModelKey.LATENT_DIFFUSION,
        run_name="calibration-runtime-authority",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
        config_path=ROOT / "configs/modeling/latent_diffusion.yaml",
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )
    loaded = provider.load_config(request)
    plan = provider.plan_stages(request, loaded)

    result = run_compute_calibration(
        provider_config=loaded,
        stage_plan=plan,
        output_root=tmp_path,
        measurements=(
            {
                "parameter": "denoiser_batch_size",
                "value": 128,
                "units_per_second": 10.0,
                "benchmark_type": "synthetic_device_smoke",
            },
        ),
    )

    assert result.payload["candidate_values"] == {"denoiser_batch_size": (32, 64, 128, 256)}
