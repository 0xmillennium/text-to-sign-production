# Artifact Storage Strategy

Sprint 2 produces large outputs that are important for reproducibility but are not appropriate for
GitHub storage. This repository keeps the workflow explicit: code and documentation stay in Git,
while heavy generated artifacts stay in the fixed Google Drive location used by the supported
Colab workflow.

## What Stays Outside GitHub

The following should not be committed to GitHub:

- raw How2Sign downloads
- processed `.npz` samples
- large interim manifests and reports
- packaged Sprint 2 archives
- DVC cache content

Git ignore rules cover the canonical generated locations used by Sprint 2, including
`data/archives/`.

## Fixed Google Drive Pattern

The supported Colab workflow uses exactly one Drive layout:

- Inputs live under `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`
- Published archives live under
  `/content/drive/MyDrive/text-to-sign-production/artifacts/sprint2/processed-v1/`

There is no storage-provider switch, no local-vs-Drive branch in the notebook, and no user-edited
artifact destination configuration.

## Archive Packaging

For local packaging, use:

```bash
python scripts/package_sprint2_outputs.py
```

For subset runs:

```bash
python scripts/package_sprint2_outputs.py --splits train
```

For the supported Colab publish flow, use:

```bash
python scripts/publish_colab_outputs.py --splits train val test
```

The archive set is:

- `sprint2_manifests_reports.tar.zst`
- `sprint2_samples_train.tar.zst`
- `sprint2_samples_val.tar.zst`
- `sprint2_samples_test.tar.zst`

Subset packaging removes stale unrequested sample archives from the local archive directory so the
local packaging output reflects the requested split selection.

## Future Sprint Reuse

Later thesis sprints can reuse stored Sprint 2 artifacts by restoring the packaged outputs into a
working environment instead of rerunning the full raw-data pipeline every time.

The boundary remains unchanged:

- GitHub Pages may describe the workflow.
- GitHub Pages must not expose private artifact locations.
- The repository remains the source of truth for code, docs, and reproducible commands.
