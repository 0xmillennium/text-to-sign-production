# Processed Schema

Sprint 2 exports a manifest-driven processed dataset under `data/processed/v1/`.

## Schema Version

The explicit processed schema version is:

`t2sp-processed-v1`

This value is stored in every processed manifest entry and every `.npz` sample payload.

## Processed Sample Files

Location:

- `data/processed/v1/samples/<split>/<sample_id>.npz`

Required arrays and fields:

| Key | Shape | Meaning |
| --- | --- | --- |
| `processed_schema_version` | scalar | Version tag for consumers and validators. |
| `body` | `(frames, 25, 2)` | Normalized body coordinates. |
| `body_confidence` | `(frames, 25)` | Body confidence values. |
| `left_hand` | `(frames, 21, 2)` | Normalized left-hand coordinates. |
| `left_hand_confidence` | `(frames, 21)` | Left-hand confidence values. |
| `right_hand` | `(frames, 21, 2)` | Normalized right-hand coordinates. |
| `right_hand_confidence` | `(frames, 21)` | Right-hand confidence values. |
| `face` | `(frames, 70, 2)` | Normalized face coordinates or zero-fill. |
| `face_confidence` | `(frames, 70)` | Face confidence values or zero-fill. |
| `people_per_frame` | `(frames,)` | Number of people observed in each frame. |
| `selected_person_index` | scalar | Always `0` in v1. |
| `frame_valid_mask` | `(frames,)` | Whether required core channels were structurally valid. |

## Normalization

Sprint 2 normalizes coordinates against the observed BFH clip canvas:

- `x / 1280`
- `y / 720`

Coordinates are not clipped to `[0, 1]`. Out-of-range values are counted and reported.
Video-header dimension deviations are also reported at runtime. On the inspected repository data,
the headers were mostly `1280x720` with a small `1920x1080` minority in the train split.

## Processed Manifests

Location:

- `data/processed/v1/manifests/train.jsonl`
- `data/processed/v1/manifests/val.jsonl`
- `data/processed/v1/manifests/test.jsonl`

Required fields:

| Field | Meaning |
| --- | --- |
| `sample_id` | Canonical sample identifier. |
| `processed_schema_version` | Explicit schema version. |
| `text` | English sentence text. |
| `split` | Official split name. |
| `fps` | Per-sample FPS from MP4 header when readable. |
| `num_frames` | Number of processed frames. |
| `sample_path` | Repo-relative `.npz` path. |
| `source_video_id` | Source `VIDEO_ID`. |
| `source_sentence_id` | Source `SENTENCE_ID`. |
| `source_sentence_name` | Source `SENTENCE_NAME`. |
| `selected_person_index` | Deterministic selected person index. |
| `multi_person_frame_count` | Frames with more than one detected person. |
| `max_people_per_frame` | Maximum detected people count in the clip. |

Sprint 2 also carries audit fields such as source paths, video metadata, frame-valid counts,
face-missing counts, `frames_with_any_zeroed_required_joint`, and per-sample issue summaries.

`frames_with_any_zeroed_required_joint` counts frames where at least one required core-channel
joint is encoded as `[0, 0, 0]` in the raw OpenPose output. This is an audit metric and does not
by itself invalidate the frame. The previous metric name was more ambiguous about that behavior.
