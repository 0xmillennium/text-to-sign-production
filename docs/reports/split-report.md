# Split Integrity Report

Dataset Build generates split-integrity reports at:

- `data/processed/v1/reports/split-report.json`
- `data/processed/v1/reports/split-report.md`

## What The Report Checks

- official split names remain `train`, `val`, and `test`
- processed sample counts per split
- raw and processed video counts per split
- sample-id overlap across processed splits
- video-id overlap across processed splits

## Verified Raw Snapshot

Local inspection of the downloaded raw dataset found:

- no `SENTENCE_NAME` overlap across the official splits
- no `VIDEO_ID` overlap across the official splits
- unmatched translation rows exist within splits and are handled explicitly rather than reassigned

## Why It Matters

The thesis pipeline must preserve the official How2Sign split boundaries. The split report makes
that preservation visible and machine-checkable after filtering and manifest export.
