# Dataset Build Filter V2 Full Validation

## Objective

Preserve the completed full `train / val / test` Dataset Build validation after manifest-driven
split archive packaging hardening and adoption of the Filter V2 policy.

## Record Type

- Document type: `Experiment Record`
- Record type: `validation`
- Stage: `Dataset Build`

## Run Identity

| Field | Value |
| --- | --- |
| Date / time context | `2026-04`; exact run timestamp not recorded |
| Git revision | `7e573e326b5f067302fec77a1f62953d90a0d14b` |
| Execution surface | Colab runtime publishing Dataset Build outputs to Drive, followed by local downloaded-archive validation |
| Environment | Python `3.12.13`; archive test tool `zstd`; machine class not recorded |

## Input Provenance

| Field | Value |
| --- | --- |
| Raw input root | `/content/drive/MyDrive/text-to-sign-production/raw/how2sign` |
| Split scope | `train`, `val`, `test` |
| Raw input references | [How2Sign Translation Files](../data/raw/how2sign-translations.md); [How2Sign Keypoint Archives](../data/raw/how2sign-keypoint-archives.md) |
| Fixed local archive workspace | `/content/how2sign_downloads` |

## Config Provenance

| Field | Value |
| --- | --- |
| Active filter config | `configs/data/filter-v2.yaml` |
| Legacy strict policy | `configs/data/filter-v1.yaml` remains available, but was not the active default |
| Active filter semantics | `body` required; at least one of `left_hand` or `right_hand` required |
| Processed schema version | `t2sp-processed-v1` |

The Filter V2 policy changes dataset membership relative to the old strict policy. It does not
change the processed schema layout.

## Artifact References

| Artifact surface | Reference |
| --- | --- |
| Processed manifest schema and paths | [Processed v1 Manifests](../data/processed/v1/manifests.md) |
| Processed sample schema and paths | [Processed v1 Samples](../data/processed/v1/samples.md) |
| Processed report schema and paths | [Processed v1 Reports](../data/processed/v1/reports.md) |
| Dataset Build manifests and reports archive | [Dataset Build Manifests And Reports Archive](../data/artifacts/dataset-build/manifests-reports-archive.md); published path `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_manifests_reports.tar.zst` |
| Dataset Build sample archives | [Dataset Build Sample Archives](../data/artifacts/dataset-build/sample-archives.md); published paths `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_samples_train.tar.zst`, `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_samples_val.tar.zst`, `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_samples_test.tar.zst` |

This record cites the canonical `docs/data` pages for artifact structure instead of re-documenting
those schemas here.

## Results

The validation covered full `train / val / test` Dataset Build execution, archive integrity,
processed-manifest to split-sample-archive equality, split-report consistency, filter-report
consistency, and one-hand sample retention evidence under Filter V2.

All four archives passed `zstd -t`.

For each split, the processed manifest sample set exactly matched the split sample archive sample
set. No `only in manifest` or `only in archive` residuals remained for any split.

| Split | Processed manifest samples | Split archive samples | Equality result |
| --- | ---: | ---: | --- |
| `train` | 31043 | 31043 | exact match |
| `val` | 1739 | 1739 | exact match |
| `test` | 2343 | 2343 | exact match |

`split-report.json` reported the following counts:

| Split | Raw samples | Processed samples | Raw videos | Processed videos |
| --- | ---: | ---: | ---: | ---: |
| `train` | 31165 | 31043 | 2192 | 2192 |
| `val` | 1741 | 1739 | 115 | 115 |
| `test` | 2357 | 2343 | 149 | 149 |

No sample-id overlap was reported between splits. Video-id overlap values were all zero.

`filter-report.json` reported the following counts:

| Split | Input samples | Kept samples | Dropped samples |
| --- | ---: | ---: | ---: |
| `train` | 31165 | 31043 | 122 |
| `val` | 1741 | 1739 | 2 |
| `test` | 2357 | 2343 | 14 |

The following split-level summary lines were observed at completion:

| Split | Summary |
| --- | --- |
| `train` | `rows=31165 matched=31043 kept=31043 dropped=122` |
| `val` | `rows=1741 matched=1739 kept=1739 dropped=2` |
| `test` | `rows=2357 matched=2343 kept=2343 dropped=14` |

Local downloaded artifacts were hashed after download:

| Artifact | SHA256 |
| --- | --- |
| `dataset_build_manifests_reports.tar.zst` | `ff0118a9a45efe8c463aa3434a5c2f9cea36b5dd667356e001e28cd49d7253d8` |
| `dataset_build_samples_train.tar.zst` | `e2cdbc81f01a1b4681ea9817f931d5d6cea01a5644b26d3b95963f8aa8b23efa` |
| `dataset_build_samples_val.tar.zst` | `89a908bf7815a9716533375dc14f4c71b4f7bcbc7046b28c968252cdf15c712b` |
| `dataset_build_samples_test.tar.zst` | `1e0fb954595b8bed108dc3c2d2d58dc4fa36c79e1ebdd9931f783230cd0599a2` |

The sample `f_zna_wG3zo_15-5-rgb_front` was validated as:

- present in the raw manifest
- present in the normalized manifest
- present in the filtered manifest
- present in the processed manifest
- present in the split sample archive
- absent from `dropped_examples`

Visible progress was observed for archive copy, archive extract, raw manifest generation,
normalization, filtering, processed manifest export, archive creation, and publish copy. These
observations are operational context only and are not runtime measurements or benchmark results.

## Interpretation

The manifest-driven packaging behavior is considered fixed behaviorally for this validated full
run: each split sample archive contained exactly the processed manifest sample set for that split.

The Filter V2 one-hand policy is considered validated behaviorally for this validated full run:
`body` plus at least one usable hand was sufficient for retention, and the one-hand sample
`f_zna_wG3zo_15-5-rgb_front` was retained through the processed artifact set.

Dataset membership differs from the old strict policy, but the processed schema layout remains
unchanged at `t2sp-processed-v1`. This record is based on manually verified run outputs and locally
validated downloaded artifacts, with the downloaded archives integrity-checked via `zstd -t` and
hashed locally.

## Limits

- This is a validation record, not a benchmark paper.
- Exact runtime duration is not recorded.
- Exact run timestamp is not recorded beyond `2026-04`.
- Machine class and fuller environment provenance beyond the recorded fields are not recorded.
- No inaccessible screenshots or Drive-only assets are required to interpret this record.

## Follow-up

- Create separate `rerun` or `comparison` experiment records for future Dataset Build behavior changes instead of expanding this record.
- Cite this validation record from downstream experiments that consume the validated `t2sp-processed-v1` Dataset Build surface.
