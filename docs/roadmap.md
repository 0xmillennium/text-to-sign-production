# Roadmap

## Sprint 1

Sprint 1 delivers repository infrastructure only:

- packaging
- tests
- documentation
- CI/CD
- historical DVC bootstrap, later removed from the active workflow by ADR-0012
- ADR and experiment logging support

## Sprint 2

Sprint 2 delivers the current implemented Dataset Build pipeline:

- raw How2Sign layout inspection and documentation
- raw manifest generation and validation
- OpenPose 2D normalization into versioned `.npz` samples
- active `filter-v2.yaml` policy with usable body plus at least one usable hand
- manifest-driven processed dataset export
- data-quality, split-integrity, and assumption reporting
- Dataset Build execution through the reusable workflow, CLI, and Colab runner
- Colab-oriented heavy execution support through the Dataset Build workflow
- manifest-driven output packaging and private/shared artifact storage guidance

## Planned Follow-On Work

The next major stage is Baseline Modeling on top of the processed manifests. Future sprints may
introduce:

1. tokenizer, retrieval, and modeling experiments on top of the processed manifests
2. richer evaluation and error-analysis workflows
3. optional pose smoothing or temporal cleanup if empirical evidence justifies it
4. downstream skeleton or avatar rendering integration
5. broader experiment tracking beyond the initial data pipeline

## Guardrails

Future implementation should preserve the core Sprint 1 principles:

- code in `src/`
- notebooks as thin drivers
- documented decisions
- reproducible commands and logs
- raw-data access behind documented manifests and schemas
- large generated artifacts outside GitHub unless there is a strong reason to change policy
