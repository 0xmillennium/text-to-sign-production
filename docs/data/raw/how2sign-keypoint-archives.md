# How2Sign Keypoint Archives

## Purpose

Raw split keypoint archives provide the OpenPose frame JSON files restored before Dataset Build
parses and normalizes pose data.

## Artifact Role

`input`, `archive`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Raw external Drive input: `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/<split>_2D_keypoints.tar.zst`
- Worktree extracted path: `/content/text-to-sign-production/data/raw/how2sign/bfh_keypoints/<split>_2D_keypoints/`
- `<split>` is one of `train`, `val`, or `test`.

## Verified Example Path

- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/train_2D_keypoints.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/val_2D_keypoints.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/test_2D_keypoints.tar.zst`

## Produced By

External How2Sign/OpenPose artifact distribution, placed under the Drive raw input root before
Dataset Build extraction.

## Consumed By

Section 3.2, Dataset Build archive extraction, restores the archives into the worktree. Section 3.3,
Dataset Build run/publish, reads the extracted keypoint directories.

## Lifecycle

The archives are raw immutable inputs. Extracted worktree directories are restorable working copies.

## Structure / Contents

Each archive contains the split-specific OpenPose output tree. Dataset Build reads 2D frame JSONs
under `openpose_output/json/<clip_id>/*.json`; source MP4 files under `openpose_output/video/` are
metadata inputs only.

## Validation

Dataset Build expects OpenPose `people` records with body, face, left-hand, and right-hand 2D
channels. The workflow records missing directories, zero-frame clips, parse failures, and
multi-person observations as auditable data facts.

## Related Artifacts

- [How2Sign translations](how2sign-translations.md)
- [Raw manifests](../interim/raw-manifests.md)
- [Normalized manifests](../interim/normalized-manifests.md)

## Related Docs

- [Data leaf template](../template.md)
- [Dataset Build execution](../../execution.md)
