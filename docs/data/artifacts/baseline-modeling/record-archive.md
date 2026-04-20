# Baseline Record Archive

## Purpose

The record archive publishes the runtime record package directory for reuse or transfer.

## Artifact Role

`archive`

## Artifact Unit

`archive bundle`

## Canonical Path Pattern

- Drive-published archive path: `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/archives/baseline_record_package.tar.zst`

## Verified Example Path

- `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/190420261845/archives/baseline_record_package.tar.zst`

## Produced By

Section 6.3, Record/package assembly and publish, packages and publishes the record archive.

## Consumed By

Section 6.2, Record archive extraction, restores record outputs. Section 6.1 checks whether
record/package outputs can be reused. Section 7.1 checks the final artifact surface.

## Lifecycle

This archive is created from record outputs and can restore the record-side run-root contents.

## Structure / Contents

The archive contains:

- `record/`

## Validation

The archive must contain the package manifest, copied evidence bundle, and copied run summary.

## Related Artifacts

- [Run summaries](run-summaries.md)
- [Evidence bundles](evidence-bundles.md)
- [Package manifest](package-manifest.md)
- [Training archive](training-archive.md)
- [Qualitative archive](qualitative-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution.md)
