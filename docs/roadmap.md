# Roadmap

## Sprint 1

Sprint 1 delivers repository infrastructure only:

- packaging
- tests
- documentation
- CI/CD
- DVC bootstrap
- ADR and experiment logging support

## Planned Follow-On Work

Future sprints may introduce:

1. reproducible dataset intake and preparation
2. explicit DVC stages for:
   - `prepare_raw`
   - `normalize_keypoints`
   - `filter_samples`
   - `build_splits`
   - `export_training_manifest`
3. baseline model experimentation
4. evaluation and reporting workflows
5. downstream skeleton or avatar rendering integration

## Guardrails

Future implementation should preserve the core Sprint 1 principles:

- code in `src/`
- notebooks as thin drivers
- documented decisions
- reproducible commands and logs
