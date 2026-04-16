# Colab Smoke Checklist

Use this checklist before release-time claims about the real Colab workflow.

## Preconditions

- The notebook is `notebooks/colab/dataset_build_colab.ipynb`.
- Google Drive contains the fixed raw input layout under
  `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`.
- Translation files exist for `train`, `val`, and `test`.
- Keypoint archives exist as `.tar.zst` files for `train`, `val`, and `test`.
- The Colab runtime has enough disk space for local archive copy and extraction.

## Smoke Steps

1. Open the Dataset Build Colab notebook in a fresh runtime.
2. Run the environment setup cell and confirm the repository imports from `src/`.
3. Mount Google Drive at `/content/drive`.
4. Run Dataset Build with the default `PIPELINE_SPLITS = ["train", "val", "test"]`.
5. Confirm staging copies translations and extracts keypoint archives into the canonical raw layout.
6. Confirm raw, normalized, filtered, and processed manifests are produced.
7. Confirm processed `.npz` samples are produced for each split.
8. Confirm reports are produced under `data/processed/v1/reports/`.
9. Confirm output archives are published under
   `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`.
10. Record any runtime, storage, archive, or Drive anomalies in the thesis execution notes.

## Failure Notes

- Drive mount failures are operational environment failures, not unit-test failures.
- Large archive transfer or extraction failures should capture available disk space, archive names,
  and the failing split.
- Any real data schema deviation should be traced back to the generated assumption and quality
  reports before changing product code.
