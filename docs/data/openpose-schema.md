# OpenPose Schema

Dataset Build documents and validates the observed OpenPose JSON contract before processing.

## Observed Top-Level Keys

Representative frame JSON files contain:

- `version`
- `people`

## Observed `people[0]` Keys

Representative frames contain:

- `person_id`
- `pose_keypoints_2d`
- `face_keypoints_2d`
- `hand_left_keypoints_2d`
- `hand_right_keypoints_2d`
- `pose_keypoints_3d`
- `face_keypoints_3d`
- `hand_left_keypoints_3d`
- `hand_right_keypoints_3d`

## 2D Channel Layout

The 2D arrays use flat OpenPose triplets of `(x, y, confidence)`:

| Channel | Raw Key | Flat Length | Points |
| --- | --- | ---: | ---: |
| body | `pose_keypoints_2d` | 75 | 25 |
| face | `face_keypoints_2d` | 210 | 70 |
| left hand | `hand_left_keypoints_2d` | 63 | 21 |
| right hand | `hand_right_keypoints_2d` | 63 | 21 |

Dataset Build reshapes these into `(points, 2)` coordinates plus `(points,)` confidence arrays.

## 3D Keys

The observed 3D arrays are present in the JSON schema but empty in the inspected data. Dataset Build
records this fact in reports but does not use the 3D arrays in v1 processing.

## Multi-Person Policy

Dataset Build deterministically selects `people[0]` when more than one person is present and makes that
fallback auditable by storing:

- `people_per_frame` in each `.npz`
- `selected_person_index = 0`
- `multi_person_frame_count`
- `max_people_per_frame`

## Face Policy

Face remains part of the observed schema and the processed export schema, but it is not allowed to
block pipeline completion in v1. Missing or weak face detections are reported, zero-filled when
necessary, and retained in the processed sample contract.
