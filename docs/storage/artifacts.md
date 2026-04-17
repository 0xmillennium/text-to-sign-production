# Artifact Storage Strategy

Dataset Build produces large outputs that are important for reproducibility but are not appropriate
for GitHub storage. This repository keeps the workflow explicit: code and documentation stay in
Git, while heavy generated artifacts stay in the fixed Google Drive location used by the supported
Colab workflow.

## What Stays Outside GitHub

The following should not be committed to GitHub:

- raw How2Sign downloads
- processed `.npz` samples
- large interim manifests and reports
- packaged Dataset Build archives

Git ignore rules cover the canonical generated locations, including `data/archives/`.

## Fixed Google Drive Pattern

The supported Colab workflow uses exactly one Drive layout:

- Inputs live under `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`
- Published archives live under
  `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`

There is no storage-provider switch, no local-vs-Drive branch in the notebook, and no user-edited
artifact destination configuration. The notebook does not support public download URLs or `gdown`.

## Archive Packaging

Local packaging is part of the primary CLI by default:

```bash
python scripts/dataset_build.py
```

To skip packaging:

```bash
python scripts/dataset_build.py --no-package
```

The archive set is:

- `dataset_build_manifests_reports.tar.zst`
- `dataset_build_samples_train.tar.zst`
- `dataset_build_samples_val.tar.zst`
- `dataset_build_samples_test.tar.zst`

Subset packaging removes stale unrequested sample archives from the local archive directory so the
local packaging output reflects the requested split selection.

Split sample archives are manifest-driven. Each `dataset_build_samples_<split>.tar.zst` contains
exactly the `.npz` files referenced by `sample_path` values in
`data/processed/v1/manifests/<split>.jsonl`; it is not a blind snapshot of
`data/processed/v1/samples/<split>/`.

## Future Stage Reuse

Later thesis stages can reuse stored Dataset Build artifacts by restoring the packaged outputs into
a working environment instead of rerunning the full raw-data pipeline every time.

The boundary remains unchanged:

- GitHub Pages may describe the workflow.
- GitHub Pages must not expose private artifact locations.
- The repository remains the source of truth for code, docs, and reproducible commands.
