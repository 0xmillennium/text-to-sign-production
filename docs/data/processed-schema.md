# Processed Schema

Dataset Build exports a manifest-driven processed dataset under `data/processed/v1/`.

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
| `selected_person_index` | scalar | Pipeline-generated samples use `0` in v1; validators treat non-zero manifest values as auditable warnings and payload/manifest mismatches as errors. |
| `frame_valid_mask` | `(frames,)` | Whether parser-required OpenPose core channels were structurally valid. |

The processed sample file contract still requires both hand arrays. Under the active filter v2
policy, a kept sample may have zero usable frames for one hand when body and the other hand are
usable.

## Normalization

Dataset Build normalizes coordinates against the observed BFH clip canvas:

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
| `selected_person_index` | Deterministic selected person index; the current v1 pipeline writes `0` and validators warn on non-zero manifest values. |
| `multi_person_frame_count` | Frames with more than one detected person. |
| `max_people_per_frame` | Maximum detected people count in the clip. |

Dataset Build also carries audit fields such as source paths, video metadata, frame-valid counts,
face-missing counts, `core_channel_nonzero_frames`,
`frames_with_any_zeroed_required_joint`, and per-sample issue summaries.

`core_channel_nonzero_frames` records usable-frame counts for body, left hand, and right hand.
Consumers that require two complete hands should inspect this field explicitly instead of assuming
that every kept sample has both hands usable.

`frames_with_any_zeroed_required_joint` counts frames where at least one parser-required
core-channel joint is encoded as `[0, 0, 0]` in the raw OpenPose output. This is an audit metric and
does not by itself invalidate the frame. The previous metric name was more ambiguous about that
behavior.
