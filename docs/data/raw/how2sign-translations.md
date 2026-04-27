# How2Sign Translation Files

## Purpose

Raw split translation files provide the English text, source identifiers, and realigned time ranges
used to create Dataset Build raw manifests.

## Artifact Role

`input`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Raw external Drive input: `<COLAB_DRIVE_PROJECT_ROOT>/raw/how2sign/translations/how2sign_realigned_<split>.csv`
- Worktree read path: `/content/text-to-sign-production/data/raw/how2sign/translations/how2sign_realigned_<split>.csv`
- `<split>` is one of `train`, `val`, or `test`.

## Verified Example Path

- `<COLAB_DRIVE_PROJECT_ROOT>/raw/how2sign/translations/how2sign_realigned_train.csv`
- `<COLAB_DRIVE_PROJECT_ROOT>/raw/how2sign/translations/how2sign_realigned_val.csv`
- `<COLAB_DRIVE_PROJECT_ROOT>/raw/how2sign/translations/how2sign_realigned_test.csv`

## Produced By

External How2Sign distribution, placed under the Drive raw input root before Dataset Build runs.

## Consumed By

Section 3.3, Dataset Build run/publish, reads the split translation files when building raw
manifests.

## Lifecycle

These are stable raw inputs. They are not produced by this repository and are not a model-training
interface.

## Structure / Contents

The files are tab-delimited even though the filenames use `.csv`. Required columns are `VIDEO_ID`,
`VIDEO_NAME`, `SENTENCE_ID`, `SENTENCE_NAME`, `START_REALIGNED`, `END_REALIGNED`, and `SENTENCE`.
`SENTENCE_NAME` becomes the canonical `sample_id`.

## Validation

Dataset Build requires the expected header, parses start and end times as seconds, and aligns
translation rows to keypoint clip directories by `SENTENCE_NAME`.

## Related Artifacts

- [How2Sign keypoint archives](how2sign-keypoint-archives.md)
- [Raw manifests](../interim/raw-manifests.md)

## Related Docs

- [Data leaf template](../template.md)
- [Dataset Build execution](../../execution.md)
