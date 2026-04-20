# Baseline Package Manifest

## Purpose

The package manifest describes the runtime-side baseline modeling package assembled for a run.

## Artifact Role

`package`, `manifest`

## Artifact Unit

`single file`

## Canonical Path Pattern

- Drive-published package manifest: `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/record/baseline_modeling_package.json`

## Verified Example Path

- `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/190420261845/record/baseline_modeling_package.json`

## Produced By

Section 6.3, Record/package assembly and publish, writes the package manifest.

## Consumed By

Section 6.3 packages the `record/` directory into the record archive. Section 7.1 checks the final
artifact surface.

## Lifecycle

This is a runtime-side package manifest under `record/`, not the formal experiment record.

## Structure / Contents

Schema version is `t2sp-baseline-modeling-package-v1`. Keys include `archives`, `artifact_role`,
`config`, `created_at`, `qualitative`, `record`, `run_name`, `run_root`, `schema_version`, and
`training`.

## Validation

Package assembly requires configs, metrics run summary, last checkpoint, qualitative metadata, and
qualitative evidence before writing the manifest.

## Related Artifacts

- [Configs](configs.md)
- [Checkpoints](checkpoints.md)
- [Run summaries](run-summaries.md)
- [Evidence bundles](evidence-bundles.md)
- [Record archive](record-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution/baseline-modeling.md)
