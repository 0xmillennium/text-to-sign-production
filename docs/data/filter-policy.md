# Filter Policy

Dataset Build uses intentionally minimal filtering so the dataset remains explicit and auditable.

## Guiding Rule

Drop a sample only when keeping it would make the processed dataset structurally unreliable. Report
everything else.

## Active Policy

The active Dataset Build filter policy is `configs/data/filter-v2.yaml`. It enables these
structural checks:

- missing text
- non-positive time range
- missing keypoint directory
- zero frame JSON files
- sample-level parse failure
- all frames structurally invalid
- unusable required body channel
- unusable required hand group

The legacy `configs/data/filter-v1.yaml` policy is still loadable for reproducibility. It used the
stricter rule that body, left hand, and right hand all had to be usable.

## Required Core Channel Policy

Filter v2 keeps a sample only when:

- body is usable
- at least one hand is usable: left hand or right hand

Filter v2 drops a sample when body is unusable, or when both hands are unusable. A single missing or
unusable hand does not itself force a drop. The canonical both-hands drop reason is
`unusable_core_channel_group:left_hand|right_hand`.

A missing secondary hand or a secondary hand with zero usable frames remains auditable through
`core_channel_nonzero_frames` in normalized and processed manifests.

The face channel is retained in schema and export, but missing face data does not drop a sample.

## Frame-Level Anomalies That Do Not Automatically Drop A Sample

- zeroed joints in some frames
- out-of-bounds coordinates
- per-frame parse or schema issues when at least one valid frame remains
- multi-person frames
- missing or unreadable video metadata

Those conditions are surfaced in reports and audit fields instead of being silently hidden.
