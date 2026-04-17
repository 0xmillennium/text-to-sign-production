# text-to-sign-production

`text-to-sign-production` is a graduation thesis repository focused on building a reproducible
research foundation for future text-to-sign work. The current implemented stage is **Dataset
Build**, a How2Sign/BFH data pipeline that produces training-ready manifests and `.npz` samples
without crossing into modeling.

## Current State

- The repository is installable as a Python package.
- Tests, linting, type checks, and docs builds are automated.
- Heavy Colab runs are supported through one fixed Dataset Build notebook.
- Local terminal runs are supported through one Dataset Build CLI script.
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

For details, start with [Dataset Build execution](execution/dataset-build.md),
[filter policy](data/filter-policy.md), [data versioning](data/versioning.md),
[processed schema](data/processed-schema.md), and [artifact storage](storage/artifacts.md).

## Not Yet Implemented

The following are intentionally outside Dataset Build:

- text tokenization
- retrieval components
- pose-token generation or training
- text-to-pose model training or inference
- skeleton or avatar rendering

This boundary is deliberate so the repository remains honest about its current maturity.
