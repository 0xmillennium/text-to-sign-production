# Data Quality Report

Dataset Build generates data-quality reports at:

- `data/processed/v1/reports/data-quality-report.json`
- `data/processed/v1/reports/data-quality-report.md`

## What The Report Covers

Per split, the generated report includes:

- processed sample counts
- dropped sample counts and reasons
- text length statistics
- frame count statistics
- FPS statistics when video metadata is readable
- face-missing ratios
- multi-person sample and frame counts
- parse or schema issue counts
- out-of-bounds coordinate counts
- `frames_with_any_zeroed_required_joint`

Drop reasons include the canonical grouped hand reason
`unusable_core_channel_group:left_hand|right_hand` when both hands are unusable. A single unusable
secondary hand is not a drop reason under the active filter v2 policy; it remains auditable through
`core_channel_nonzero_frames` in the processed manifest.

`frames_with_any_zeroed_required_joint` counts frames where at least one parser-required
core-channel joint is encoded as `[0, 0, 0]` in the raw OpenPose output. It is an audit/debug
metric and does not by itself invalidate the frame.

## Why It Exists

Dataset Build treats raw-data assumptions as runtime-checked facts rather than undocumented beliefs.
The data-quality report is the main human-readable summary of what the pipeline kept, dropped, and
observed while exporting the dataset.

## Related Reports

- machine-readable assumptions: `data/interim/reports/assumption-report.json`
- human-readable assumptions: `data/processed/v1/reports/assumption-report.md`
- split integrity: `data/processed/v1/reports/split-report.md`
