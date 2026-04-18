# Colab Smoke Checklist

Use this checklist before release-time claims about the real Dataset Build Colab workflow.

## Preconditions

- The notebook is `notebooks/colab/text_to_sign_production_colab.ipynb`.
- Google Drive contains the fixed raw input layout under
  `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`.
- Translation files exist for `train`, `val`, and `test`.
- Keypoint archives exist as `.tar.zst` files for `train`, `val`, and `test`.
- The Colab runtime has enough disk space for local archive copy and extraction.
- The active default filter config is `configs/data/filter-v2.yaml`.

## Smoke Steps

1. Open the main Colab notebook in a fresh runtime.
2. Run Section 0 runtime/session setup.
3. Mount Google Drive at `/content/drive`.
4. Acquire and install the repository.
5. In Section 3, reuse extracted Dataset Build outputs if they already exist.
6. Otherwise extract archived Dataset Build outputs when the fixed Drive archives are present.
7. Otherwise run Dataset Build with the default `PIPELINE_SPLITS = ["train", "val", "test"]`.
8. Confirm staging copies translations and extracts keypoint archives into the canonical raw layout
   when the run/publish cell is used.
9. Confirm raw, normalized, filtered, and processed manifests are produced.
10. Confirm processed `.npz` samples are produced for each split.
11. Confirm reports are produced under `data/processed/v1/reports/`.
12. Confirm output archives are published under
   `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`.
13. Confirm the published archive set includes `dataset_build_manifests_reports.tar.zst` and
    `dataset_build_samples_train.tar.zst`, `dataset_build_samples_val.tar.zst`, and
    `dataset_build_samples_test.tar.zst`.
14. Confirm each `dataset_build_samples_<split>.tar.zst` contains exactly the `.npz` files
    referenced by `data/processed/v1/manifests/<split>.jsonl`.
15. Record any runtime, storage, archive, or Drive anomalies in the thesis execution notes.
16. If continuing into Sprint 3 baseline modeling, use
    `text_to_sign_production_colab_resume_checklist.md` for the separate baseline archive/resume
    checks.

## Failure Notes

- Drive mount failures are operational environment failures, not unit-test failures.
- Large archive transfer or extraction failures should capture available disk space, archive names,
  and the failing split.
- Sample archive mismatches should be treated as manifest-driven packaging failures, not
  whole-directory copy differences.
- Under `filter-v2.yaml`, a sample with usable body and one usable hand may be kept; do not treat
  one-hand retention as a smoke failure by itself.
- Any real data schema deviation should be traced back to the generated assumption and quality
  reports before changing product code.
