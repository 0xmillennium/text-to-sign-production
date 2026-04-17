# Experiment Log Template

## Objective

State the research question or engineering hypothesis.

## Dataset / Version

Identify the dataset source, split scope, processed manifest or archive references, and
`processed_schema_version` used. For Dataset Build outputs, include the relevant
`data/processed/v1/manifests/<split>.jsonl` files or packaged archive names.

## Config

List the relevant configuration values, files, and environment assumptions. For Dataset Build
inputs, name the filter config: `configs/data/filter-v2.yaml` is the active default, while
`configs/data/filter-v1.yaml` should appear only when intentionally selected for legacy
reproducibility or comparison.

## Command

Record the exact command or notebook entry point used to run the experiment.

## Result Summary

Summarize the main outputs, metrics, counts, artifacts, and interpretation.

## Notes

Capture anomalies, caveats, or operational details.

## Next Step

State the most important follow-up action.
