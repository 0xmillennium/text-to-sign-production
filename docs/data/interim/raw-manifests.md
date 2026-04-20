# Raw Manifests

## Purpose

Raw manifests preserve each translation row together with its verified keypoint alignment status
before normalization or filtering.

## Artifact Role

`intermediate`, `manifest`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Worktree path: `/content/text-to-sign-production/data/interim/raw_manifests/raw_<split>.jsonl`
- Archive member path: `data/interim/raw_manifests/raw_<split>.jsonl`
- `<split>` is one of `train`, `val`, or `test`.

## Verified Example Path

- `/content/text-to-sign-production/data/interim/raw_manifests/raw_train.jsonl`

## Produced By

Section 3.3, Dataset Build run/publish, writes raw manifests from translation files and extracted
keypoint directories.

## Consumed By

Section 3.3, Dataset Build run/publish, consumes raw manifests when producing normalized manifests.

## Lifecycle

Raw manifests are regenerated Dataset Build intermediates and are packaged into the Dataset Build
manifests and reports archive.

## Structure / Contents

Each JSONL record describes one translation row. Required fields include `sample_id`,
`source_split`, `video_id`, `video_name`, `sentence_id`, `sentence_name`, `text`, `start_time`,
`end_time`, `keypoints_dir`, `source_metadata_path`, `has_face`, and `num_frames`. Audit fields may
include `source_video_path`, `video_width`, `video_height`, `video_fps`, and
`video_metadata_error`.

## Validation

Unmatched translation rows remain explicit: `keypoints_dir` is `null`, `num_frames` is `0`, and
`has_face` is `null`. No fallback join key is used.

## Related Artifacts

- [How2Sign translations](../raw/how2sign-translations.md)
- [How2Sign keypoint archives](../raw/how2sign-keypoint-archives.md)
- [Normalized manifests](normalized-manifests.md)
- [Manifests and reports archive](../artifacts/dataset-build/manifests-reports-archive.md)

## Related Docs

- [Data leaf template](../template.md)
- [Dataset Build execution](../../execution/dataset-build.md)
