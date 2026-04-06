# ADR-0010: Face Is Optional In v1

- Status: Accepted
- Date: 2026-04-06

## Context

Observed OpenPose JSON files include `face_keypoints_2d`, so the processed schema should not erase
face support. At the same time, Sprint 2 should not fail the whole dataset build because some face
data is missing, weak, or inconsistent while body and hands remain usable.

## Decision

Sprint 2 keeps face in the raw schema checks, processed `.npz` payloads, processed manifests, and
reports, but it does not make face completeness a prerequisite for sample retention. Missing face
data is zero-filled when necessary and reported explicitly.

## Consequences

- The processed schema stays aligned with the observed raw data.
- Body and hand coverage can drive v1 completion without pretending face data is absent.
- Future sprints can tighten or expand face handling without redesigning the whole export format.
