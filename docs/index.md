# text-to-sign-production

`text-to-sign-production` is a graduation thesis repository focused on building a reproducible
research foundation for future text-to-sign work. The current implemented stage is **Dataset
Build**, a How2Sign/BFH data pipeline that produces training-ready manifests and `.npz` samples
without making Baseline Modeling the main thesis-contribution stage. Sprint 3 now has a
baseline-only public workflow surface for reproducible train/val runs, qualitative panel export,
and runtime-side evidence packaging on processed Dataset Build outputs.

## Current State

- The repository is installable as a Python package.
- Tests, linting, type checks, and docs builds are automated.
- Heavy Colab runs are supported through one fixed notebook that covers Dataset Build and Sprint 3
  baseline orchestration.
- Local terminal runs are supported through one Dataset Build CLI script.
- Sprint 3 baseline terminal runs are supported through one stage-oriented CLI script.
- Dataset Build reproducibility is expressed through the workflow entrypoint, manifests, reports,
  and packaged `.tar.zst` archives.
- The active default filter policy is `configs/data/filter-v2.yaml` with usable body plus at least
  one usable hand; the stricter `configs/data/filter-v1.yaml` policy is
  legacy/reproducibility-oriented.
- Processed dataset access is manifest-driven.
- Runtime assumption, data-quality, and split-integrity reporting are built into the pipeline.
- Artifact packaging and private/shared storage guidance are documented without exposing private
  links.
- Notebooks remain explicitly limited to thin runner behavior.
- Sprint 3 baseline workflow operationalization is available; formal experiment-record authoring,
  broader evaluation, and thesis-contribution modeling remain future work.

For details, start with [Dataset Build execution](execution/dataset-build.md),
[Baseline Modeling execution](execution/baseline-modeling.md), [filter policy](data/filter-policy.md),
[data versioning](data/versioning.md), [processed schema](data/processed-schema.md), and
[artifact storage](storage/artifacts.md). For the post-Dataset-Build thesis direction, read the
[roadmap](roadmap.md) and [literature positioning](literature-positioning.md).

## Not Yet Implemented

The following are intentionally outside Dataset Build:

- text tokenization
- retrieval components
- pose-token generation or training
- production text-to-pose inference
- broad baseline evaluation
- formal experiment-record authoring
- skeleton or avatar rendering

This boundary is deliberate so the repository remains honest about its current maturity.
