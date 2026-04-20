# Baseline Qualitative Archive

## Purpose

The qualitative archive publishes qualitative validation panel outputs for reuse or restoration
under the same run root.

## Artifact Role

`archive`

## Artifact Unit

`archive bundle`

## Canonical Path Pattern

- Drive-published archive path: `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/archives/baseline_qualitative_outputs.tar.zst`

## Verified Example Path

- `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/190420261845/archives/baseline_qualitative_outputs.tar.zst`

## Produced By

Section 5.3, Qualitative export/publish, packages and publishes the qualitative archive.

## Consumed By

Section 5.2, Qualitative archive extraction, restores qualitative outputs. Section 5.1 checks
whether qualitative outputs can be reused. Section 7.1 checks the final artifact surface.

## Lifecycle

This archive is created from qualitative outputs and can restore the qualitative run-root contents.

## Structure / Contents

The archive contains:

- `qualitative/`

## Validation

The archive must contain qualitative metadata, qualitative sample artifacts, and the qualitative
evidence bundle required by downstream packaging.

## Related Artifacts

- [Qualitative metadata](qualitative-metadata.md)
- [Qualitative sample artifacts](qualitative-sample-artifacts.md)
- [Evidence bundles](evidence-bundles.md)
- [Training archive](training-archive.md)
- [Record archive](record-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution/baseline-modeling.md)
