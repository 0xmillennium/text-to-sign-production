# Translation TSV Schema

Dataset Build reads the translation files as tab-separated values only, even though the canonical
How2Sign filenames keep the `.csv` extension.

## Verified Delimiter

The real translation files under `data/raw/how2sign/translations/` are tab-separated rather than
comma-separated.

## Verified Columns

Local inspection confirmed the exact header:

| Column | Meaning |
| --- | --- |
| `VIDEO_ID` | Original video identifier. |
| `VIDEO_NAME` | Raw video filename stem with camera suffix. |
| `SENTENCE_ID` | Sentence-level segment identifier. |
| `SENTENCE_NAME` | Sentence-level clip name used for direct alignment. |
| `START_REALIGNED` | Realigned segment start time in seconds. |
| `END_REALIGNED` | Realigned segment end time in seconds. |
| `SENTENCE` | English text for the segment. |

Dataset Build does not fabricate extra TSV columns and treats a header mismatch as a hard parser
failure.

## Parsing Rules

- `START_REALIGNED` and `END_REALIGNED` are parsed as floating-point seconds.
- `SENTENCE_NAME` becomes the canonical `sample_id`.
- The raw manifest stores the translation TSV path in `source_metadata_path`.
- Empty text is not rewritten or guessed; it is handled by the filtering policy if observed.
