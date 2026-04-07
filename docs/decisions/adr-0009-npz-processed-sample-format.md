# ADR-0009: Use `.npz` For Processed Samples

- Status: Accepted
- Date: 2026-04-06

## Context

Sprint 2 needs a compact, widely supported format for storing multiple per-sample arrays together:
body, hands, face, confidences, and audit metadata. Splitting every channel into separate files
would complicate the manifests and make atomic sample handling harder.

## Decision

Sprint 2 stores each processed sample as one compressed `.npz` file containing the normalized
arrays, confidences, and audit fields needed by downstream consumers.

## Consequences

- Each sample has one canonical file path in the processed manifest.
- The format stays simple for NumPy-based research code.
- Compression reduces storage overhead relative to many small uncompressed array files.
