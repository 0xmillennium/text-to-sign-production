# text-to-sign-production

`text-to-sign-production` is a graduation thesis repository for reproducible English
text-to-sign-pose research.

Current public stage: Dataset Build.

Implemented internal downstream surface: Baseline Modeling.

Not yet implemented: broader evaluation, contribution modeling, playback/demo.

## Current State

- Dataset Build produces manifest-driven processed Dataset Build outputs from fixed How2Sign/BFH
  inputs.
- Baseline Modeling consumes processed manifests and `.npz` samples for baseline-only train/val
  runs, qualitative panel export, and runtime evidence packaging.
- The single project-wide Colab notebook is
  `notebooks/colab/text_to_sign_production_colab.ipynb`.
- The active Dataset Build filter policy is `configs/data/filter-v2.yaml`: usable body plus at
  least one usable hand.
- Processed schema version is `t2sp-processed-v1`.
- Large generated artifacts stay outside GitHub and are restored through extracted outputs or
  archived outputs.
- Sprint 3 has a formal experiment-record documentation surface that cites runtime artifacts rather
  than replacing them with a tracking server.

## Start Here

- Run the current public stage with [Dataset Build execution](execution/dataset-build.md).
- Run the internal baseline surface with [Baseline Modeling execution](execution/baseline-modeling.md).
- Use [Getting Started](getting-started.md) for install and first commands.
- Use [Artifact Storage](storage/artifacts.md) for run roots, publish locations, and archive names.
- Use [Experiments](experiments.md) for Dataset Build validation records and Sprint 3 baseline
  experiment-record practice.
- Use [Roadmap](roadmap.md) and [Literature Positioning](literature-positioning.md) for the thesis
  direction after Dataset Build.

## Boundaries

The repository is intentionally honest about maturity. Dataset Build is public and implemented.
Baseline Modeling is implemented as baseline evidence for later comparisons. Broader evaluation,
contribution modeling, and playback/demo remain later phases.
