# Raw Schema

Dataset Build emits one raw manifest per official split:

- `data/interim/raw_manifests/raw_train.jsonl`
- `data/interim/raw_manifests/raw_val.jsonl`
- `data/interim/raw_manifests/raw_test.jsonl`

Each line is a JSON object describing one translation row plus its verified keypoint alignment
status.

## Required Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Canonical sample identifier. Equal to `SENTENCE_NAME`. |
| `source_split` | string | One of `train`, `val`, `test`. |
| `video_id` | string | Raw `VIDEO_ID` from the TSV row. |
| `video_name` | string | Raw `VIDEO_NAME` from the TSV row. |
| `sentence_id` | string | Raw `SENTENCE_ID` from the TSV row. |
| `sentence_name` | string | Raw `SENTENCE_NAME` from the TSV row. |
| `text` | string | Raw `SENTENCE` text. |
| `start_time` | float | Parsed `START_REALIGNED` value in seconds. |
| `end_time` | float | Parsed `END_REALIGNED` value in seconds. |
| `keypoints_dir` | string or null | Repo-relative clip directory for frame JSONs. |
| `source_metadata_path` | string | Repo-relative path to the tab-delimited translation `.csv` source. |
| `has_face` | boolean or null | Whether first-frame inspection found a usable face channel. |
| `num_frames` | integer | Number of JSON frame files in the matched clip directory. |

## Additional Audit Fields

Dataset Build also records video metadata facts when available:

| Field | Type | Meaning |
| --- | --- | --- |
| `source_video_path` | string or null | Expected repo-relative MP4 path. |
| `video_width` | integer or null | Width parsed from the MP4 header. |
| `video_height` | integer or null | Height parsed from the MP4 header. |
| `video_fps` | float or null | FPS parsed from the MP4 header. |
| `video_metadata_error` | string or null | Explicit reason when metadata could not be read. |

## Unmatched Rows

If a translation row has no matching keypoint directory, Dataset Build preserves the row in the raw
manifest and writes:

- `keypoints_dir = null`
- `num_frames = 0`
- `has_face = null`

No alternate join key or fallback directory search is attempted.
