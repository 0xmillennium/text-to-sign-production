# text-to-sign-production

`text-to-sign-production` is a graduation thesis repository focused on building a reproducible
research foundation for future text-to-sign work. The current implemented stage is **Dataset
Build**, a How2Sign/BFH data pipeline that produces training-ready manifests and `.npz` samples
without crossing into modeling.

## Current State

- The repository is installable as a Python package.
- Tests, linting, type checks, and docs builds are automated.
- DVC reproduces the Dataset Build stage.
- Heavy Colab runs are supported through one fixed Dataset Build notebook.
- Local terminal runs are supported through one Dataset Build CLI script.
- Processed dataset access is manifest-driven.
- Runtime assumption, data-quality, and split-integrity reporting are built into the pipeline.
- Artifact packaging and private/shared storage guidance are documented without exposing private
  links.
- Notebooks remain explicitly limited to thin runner behavior.

## Not Yet Implemented

The following are intentionally outside Dataset Build:

- text tokenization
- retrieval components
- pose-token generation or training
- text-to-pose model training or inference
- skeleton or avatar rendering

This boundary is deliberate so the repository remains honest about its current maturity.
