# Filter Policy

Dataset Build uses intentionally minimal filtering so the dataset remains explicit and auditable.

## Guiding Rule

Drop a sample only when keeping it would make the processed dataset structurally unreliable. Report
everything else.

## v1 Structural Drop Reasons

The Dataset Build filter policy in `configs/data/filter-v1.yaml` enables these checks:

- missing text
- non-positive time range
- missing keypoint directory
- zero frame JSON files
- sample-level parse failure
- all frames structurally invalid
- unusable required core channels

## Required Core Channels

These channels are required in v1:

- body
- left hand
- right hand

The face channel is retained in schema and export, but missing face data does not drop a sample.

## Frame-Level Anomalies That Do Not Automatically Drop A Sample

- zeroed joints in some frames
- out-of-bounds coordinates
- per-frame parse or schema issues when at least one valid frame remains
- multi-person frames
- missing or unreadable video metadata

Those conditions are surfaced in reports and audit fields instead of being silently hidden.
