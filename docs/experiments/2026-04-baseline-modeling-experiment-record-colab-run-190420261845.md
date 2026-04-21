# Baseline Modeling Colab Run 190420261845

## Objective

Verify that the supported Colab workflow successfully produced the training, qualitative, and
record/package runtime surfaces for the Baseline Modeling run `190420261845`.

## Record Type

- Document type: `Experiment Record`
- Record type: `baseline-run`
- Stage: `Baseline Modeling`

## Run Identity

| Field | Value |
| --- | --- |
| Stage | `Baseline Modeling` |
| Run name | `190420261845` |
| Date / time context | run name `190420261845`; package `created_at` `2026-04-19T16:03:44+00:00` |
| Git revision | `284a7bac5f061488eb548720d58567dc0257f0d1` |
| Execution surface | `Colab notebook` |
| Python version | `3.12.13` |
| Conda env | `None` |
| Torch version | `2.10.0+cu128` |
| CUDA available | `True` |
| GPU | `NVIDIA A100-SXM4-40GB` |

## Input Provenance

Use `docs/data` as the canonical artifact dictionary for raw, processed, and archive schema
details.

| Field | Value |
| --- | --- |
| Raw input root | `/content/drive/MyDrive/text-to-sign-production/raw/how2sign` |
| Raw input references | [How2Sign Translation Files](../data/raw/how2sign-translations.md); [How2Sign Keypoint Archives](../data/raw/how2sign-keypoint-archives.md) |
| Dataset Build artifact root | `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1` |
| Consumed processed manifests | `data/processed/v1/manifests/train.jsonl`; `data/processed/v1/manifests/val.jsonl` |
| Processed manifest reference | [Processed v1 Manifests](../data/processed/v1/manifests.md) |
| Processed sample reference | [Processed v1 Samples](../data/processed/v1/samples.md) |
| Processed report reference | [Processed v1 Reports](../data/processed/v1/reports.md) |
| Processed schema version | `t2sp-processed-v1` |

| Split | Processed manifest count |
| --- | ---: |
| `train` | 31043 |
| `val` | 1739 |
| `test` | 2343 |

## Config Provenance

| Field | Value |
| --- | --- |
| Effective config | `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/190420261845/config/baseline.yaml` |
| Source config | `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/190420261845/config/source_baseline.yaml` |
| Repo default config | `/content/text-to-sign-production/src/text_to_sign_production/modeling/config/baseline.yaml` |
| Canonical config reference | [Baseline Modeling Configs](../data/artifacts/baseline-modeling/configs.md) |
| Backbone name | `google/flan-t5-base` |
| Backbone freeze | `True` |
| Epochs | `1` |
| Batch size | `2` |
| Seed | `13` |
| Train split | `train` |
| Val split | `val` |
| Effective config checkpoint output dir | points into the run root |
| Source config checkpoint output dir | `outputs/modeling/baseline` |
| Repo default checkpoint output dir | `outputs/modeling/baseline` |

## Artifact References

The formal experiment record cites these runtime artifacts; their schema details remain defined in
`docs/data`.

| Category | Runtime references | Canonical `docs/data` page |
| --- | --- | --- |
| Configs | `config/baseline.yaml`; `config/source_baseline.yaml` | [Baseline Modeling Configs](../data/artifacts/baseline-modeling/configs.md) |
| Checkpoints | `checkpoints/best.pt`; `checkpoints/last.pt` | [Baseline Modeling Checkpoints](../data/artifacts/baseline-modeling/checkpoints.md) |
| Run summaries | `checkpoints/run_summary.json`; `metrics/run_summary.json`; `record/run_summary.json` | [Baseline Modeling Run Summaries](../data/artifacts/baseline-modeling/run-summaries.md) |
| Qualitative metadata | `qualitative/panel_definition.json`; `qualitative/records.jsonl`; `qualitative/panel_summary.json` | [Baseline Qualitative Metadata](../data/artifacts/baseline-modeling/qualitative-metadata.md) |
| Qualitative sample artifacts | qualitative validation exports referenced by `records.jsonl` | [Baseline Qualitative Sample Artifacts](../data/artifacts/baseline-modeling/qualitative-sample-artifacts.md) |
| Evidence bundles | `qualitative/baseline_evidence_bundle.json`; `record/baseline_evidence_bundle.json` | [Baseline Evidence Bundles](../data/artifacts/baseline-modeling/evidence-bundles.md) |
| Package manifest | `record/baseline_modeling_package.json` | [Baseline Package Manifest](../data/artifacts/baseline-modeling/package-manifest.md) |
| Training archive | `archives/baseline_training_outputs.tar.zst` | [Baseline Training Archive](../data/artifacts/baseline-modeling/training-archive.md) |
| Qualitative archive | `archives/baseline_qualitative_outputs.tar.zst` | [Baseline Qualitative Archive](../data/artifacts/baseline-modeling/qualitative-archive.md) |
| Record archive | `archives/baseline_record_package.tar.zst` | [Baseline Record Archive](../data/artifacts/baseline-modeling/record-archive.md) |

## Results

Training and summary facts:

| Field | Value |
| --- | --- |
| Backbone name | `google/flan-t5-base` |
| Best epoch | `1` |
| Best validation loss | `0.01348864989244703` |
| Final train loss | `0.01487835251553219` |
| Final validation loss | `0.01348864989244703` |
| Final validation metric | `0.10629466834051284` |
| Target channels | `body`, `left_hand`, `right_hand` |
| History length | `1` |

Checkpoint facts:

| Field | Value |
| --- | --- |
| Checkpoint schema version | `t2sp-baseline-checkpoint-v1` |
| `best.pt` role | `best` |
| `last.pt` role | `last` |

Qualitative facts:

| Field | Value |
| --- | --- |
| Panel schema version | `t2sp-qualitative-panel-v1` |
| Export schema version | `t2sp-qualitative-export-v1` |
| Split | `val` |
| Selection rule | `sorted-by-sample-id:first-8-of-1739-validation-records` |
| Sample count | `8` |

Evidence and package facts:

| Field | Value |
| --- | --- |
| Evidence schema version | `t2sp-baseline-evidence-v1` |
| Package schema version | `t2sp-baseline-modeling-package-v1` |
| Artifact role | `runtime-side baseline evidence, not a formal experiment record` |

Archive composition facts:

| Archive | Contains |
| --- | --- |
| `baseline_training_outputs.tar.zst` | `config/`, `checkpoints/`, `metrics/` |
| `baseline_qualitative_outputs.tar.zst` | `qualitative/` |
| `baseline_record_package.tar.zst` | `record/` |

## Interpretation

This run shows that the supported Colab Baseline Modeling workflow executed successfully end to end
for one baseline run. The training surface produced real checkpoints and scalar run summaries, the
qualitative panel was deterministic and reference-backed, the runtime evidence/package surface was
assembled successfully, and archive publication worked for training, qualitative, and record
outputs.

These results validate the workflow surfaces and their published runtime artifacts. They do not, by
themselves, establish broader evaluation coverage or baseline quality beyond the recorded one-epoch
run.

## Limits

- This is a one-epoch baseline only.
- Broader evaluation is not yet implemented in the supported workflow.
- Contribution modeling is not yet implemented.
- Playback/demo is not yet implemented.
- The runtime package is not the formal experiment record; this Markdown record is the formal record that cites it.

## Follow-up

- Run additional baseline records beyond a single one-epoch reference run.
- Add comparison records against later baseline reruns or changed configurations.
- Add ablation records once alternative baseline configurations are supported.
- Attach later evaluation surfaces when the supported workflow includes them.
