# Experiment Logs

Experiment logs capture empirical work that needs provenance beyond routine generated reports.

## Why Log Experiments

- to preserve dataset and configuration provenance
- to compare results across iterations
- to make future claims traceable and reviewable

## Current Practice

Dataset Build is the implemented data stage. Baseline Modeling is the next major stage, so most
new experiment records should describe modeling runs, modeling ablations, or intentional Dataset
Build comparisons used as experiment evidence.

Routine Dataset Build runs are documented by manifests, reports, and artifact archives. Create an
experiment log when a run supports a research or engineering claim that needs interpretation across
datasets, configs, metrics, or artifacts.

When Dataset Build output is used, record the processed manifest or archive references, split
scope, filter config, and processed schema version. The active filter config is
`configs/data/filter-v2.yaml`; the legacy strict policy at `configs/data/filter-v1.yaml` should be
named only when it is intentionally selected for reproducibility or comparison.

## Validation Records

- [Dataset Build Filter V2 Full-Run Validation](experiments/2026-04-dataset-build-filter-v2-full-validation.md):
  full `train / val / test` validation after manifest-driven packaging hardening and Filter V2
  adoption.
