# Raw Layout

Sprint 2 treats the following root as the only canonical How2Sign raw input:

`data/raw/how2sign/`

## Canonical Subtrees

Translations:

- `data/raw/how2sign/translations/how2sign_realigned_train.csv`
- `data/raw/how2sign/translations/how2sign_realigned_val.csv`
- `data/raw/how2sign/translations/how2sign_realigned_test.csv`

These translation files are tab-delimited even though the observed How2Sign filenames use the
`.csv` extension.

BFH keypoints:

- `data/raw/how2sign/bfh_keypoints/train_2D_keypoints/openpose_output/json/<clip_id>/*.json`
- `data/raw/how2sign/bfh_keypoints/train_2D_keypoints/openpose_output/video/<clip>.mp4`
- `data/raw/how2sign/bfh_keypoints/val_2D_keypoints/openpose_output/json/<clip_id>/*.json`
- `data/raw/how2sign/bfh_keypoints/val_2D_keypoints/openpose_output/video/<clip>.mp4`
- `data/raw/how2sign/bfh_keypoints/test_2D_keypoints/openpose_output/json/<clip_id>/*.json`
- `data/raw/how2sign/bfh_keypoints/test_2D_keypoints/openpose_output/video/<clip>.mp4`

The OpenPose frame JSON directories are the canonical Sprint 2 signal source. The MP4 files are
used only for metadata inspection and optional manual viewing.

## Official Split Preservation

Sprint 2 preserves the observed official split names exactly:

- `train`
- `val`
- `test`

There is no split remapping, no extra dev split, and no heuristic resampling.

## Verified Alignment Rule

The only alignment rule used by Sprint 2 is:

`SENTENCE_NAME == openpose_output/json/<clip_dir_name>`

This direct mapping was verified locally and is preferred over any fallback heuristic.

## Observed Raw Counts

Local inspection of the downloaded raw dataset produced these counts:

| Split | Translation Rows | Keypoint Dirs | Unmatched Translation Rows |
| --- | ---: | ---: | ---: |
| train | 31,165 | 31,047 | 118 |
| val | 1,741 | 1,739 | 2 |
| test | 2,357 | 2,343 | 14 |

The unmatched rows are preserved in raw manifests with explicit null or zero values and are later
handled by documented filtering and reporting rather than hidden repair logic.
