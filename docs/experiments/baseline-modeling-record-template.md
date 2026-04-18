# Baseline Modeling Record Template

Copy this template for a formal Baseline Modeling experiment record.

## Objective

State the baseline question or comparison need. Identify whether this is the default baseline run,
a rerun, or an intentional config comparison.

## Run Identity

| Field | Value |
| --- | --- |
| Run name | `<run_name>` |
| Run root | `outputs/modeling/baseline-modeling/runs/<run_name>/` or Colab Drive run root |
| Execution surface | `python scripts/baseline_modeling.py ...` or `notebooks/colab/text_to_sign_production_colab.ipynb` |
| Date | `YYYY-MM-DD` |
| Git revision | `<revision>` |
| Environment | `<local/Colab, Python, accelerator notes>` |

## Dataset Build Provenance

| Field | Value |
| --- | --- |
| Processed schema | `t2sp-processed-v1` |
| Filter config | `configs/data/filter-v2.yaml` |
| Train manifest | `data/processed/v1/manifests/train.jsonl` or archive reference |
| Val manifest | `data/processed/v1/manifests/val.jsonl` or archive reference |
| Sample archives | `dataset_build_samples_train.tar.zst`, `dataset_build_samples_val.tar.zst` |
| Reports | `data/processed/v1/reports/` or `dataset_build_manifests_reports.tar.zst` |

## Baseline Config

| Field | Value |
| --- | --- |
| Source config | `config/source_baseline.yaml` |
| Effective config | `config/baseline.yaml` |
| Text backbone | `google/flan-t5-base` or configured value |
| Target channels | `body`, `left_hand`, `right_hand` |
| Seed | `<seed>` |

## Training Artifacts

| Artifact | Reference |
| --- | --- |
| Run summary | `checkpoints/run_summary.json` and/or `metrics/run_summary.json` |
| Last checkpoint | `checkpoints/last.pt` |
| Best checkpoint | `checkpoints/best.pt` or `not produced` |
| Training archive | `archives/baseline_training_outputs.tar.zst` |

Summarize final train loss, final validation loss, validation metric, best validation loss, and
best epoch from `run_summary.json`.

## Qualitative Artifacts

| Artifact | Reference |
| --- | --- |
| Panel definition | `qualitative/panel_definition.json` |
| Records | `qualitative/records.jsonl` |
| Panel summary | `qualitative/panel_summary.json` |
| Evidence bundle | `qualitative/baseline_evidence_bundle.json` |
| Reference exports | `qualitative/references/*.npz` |
| Prediction exports | `qualitative/predictions/*.npz` |
| Qualitative archive | `archives/baseline_qualitative_outputs.tar.zst` |

Summarize panel size, sample IDs, selection rule, and any visible qualitative observations.

## Record/Package Artifacts

| Artifact | Reference |
| --- | --- |
| Package manifest | `record/baseline_modeling_package.json` |
| Copied evidence bundle | `record/baseline_evidence_bundle.json` |
| Copied run summary | `record/run_summary.json` |
| Record archive | `archives/baseline_record_package.tar.zst` |

Explain whether the package was reused, extracted, or assembled during this run.

## Result Summary

Summarize the main numeric and qualitative outcomes. Keep claims baseline-scoped.

## Interpretation And Limits

Explain what this baseline evidence supports and what it does not support. Do not claim broader
evaluation, contribution modeling, or playback/demo behavior.

## Later-Phase Consumption

State how Sprint 4, Sprint 5, Sprint 6, or later phases should consume this record.

## Notes

Capture anomalies, missing artifacts, operational caveats, or follow-up needs.
