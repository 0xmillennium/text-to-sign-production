# Artifact Storage Strategy

Sprint 2 produces large outputs that are important for reproducibility but are not appropriate for
GitHub storage. This repository keeps the workflow explicit: code and documentation stay in Git,
while heavy generated artifacts stay in private/shared storage.

## What Stays Outside GitHub

The following should not be committed to GitHub:

- raw How2Sign downloads
- processed `.npz` samples
- large interim manifests and reports
- packaged Sprint 2 archives
- DVC cache content

Git ignore rules cover the canonical generated locations used by Sprint 2, including
`data/archives/`.

## Example Storage Configuration

The repository provides:

- `configs/storage.example.yaml`

This file documents the expected operational fields:

- `provider`
- `artifact_root`
- `run_label`
- `local_download`
- `upload_to_storage`
- `google_drive.mount_path`
- `google_drive.relative_root`
- archive naming fields

If you need private runtime-specific values, create:

- `configs/storage.local.yaml`

That file is intentionally ignored by Git and must never be committed with private paths, folder
IDs, links, or credentials.

## Google Drive Pattern

Google Drive is an allowed private/shared storage target for Sprint 2 operational runs.

The intended pattern is:

1. Mount Google Drive inside Colab.
2. Copy the generated `.tar.zst` archives into a user-controlled path derived from:
   - `google_drive.mount_path`
   - `google_drive.relative_root`
   - `artifact_root`
   - `run_label`
3. Keep those private/shared locations out of the repository and out of GitHub Pages.

The repository documents the pattern only. It does not publish real folder IDs, share URLs, or
private location details.

## Archive Packaging

Use the packaging helper after the Sprint 2 scripts finish:

```bash
python scripts/package_sprint2_outputs.py
```

The helper creates:

- `sprint2_manifests_reports.tar.zst`
- `sprint2_samples_train.tar.zst`
- `sprint2_samples_val.tar.zst`
- `sprint2_samples_test.tar.zst`

This packaging step reduces the operational pain of handling many small files and makes later
transfer or reuse more practical.

## Future Sprint Reuse

Later thesis sprints can reuse stored Sprint 2 artifacts by restoring the packaged outputs into a
working environment instead of rerunning the full raw-data pipeline every time.

The important boundary remains unchanged:

- GitHub Pages may describe the workflow.
- GitHub Pages must not expose private artifact locations.
- The repository remains the source of truth for code, docs, and reproducible commands.
