# Candidate Universe

## Purpose

This surface stores family-level audit-universe records. It defines the method, boundary,
counter-alternative, frontier, and evaluation families that later candidate cards may draw from.

Family-level instantiation does not mean contribution selection, scoring, or implementation
commitment. This is not the candidate-analysis surface itself.

## Design Basis

The refreshed family universe is derived from the
[Source Selection Criteria](../../source-selection-criteria.md), the
[Source Corpus](../../source-corpus.md), the 28 reviewed sources recorded in that corpus, and the
[Candidate Universe Template](template.md).

Families are organized by audit question, not by paper count and not by pre-existing contribution
choices. Families are analytic audit axes, not mutually exclusive paper buckets.

## Why These Families?

| Audit question | Family |
| --- | --- |
| What data and supervision regime is compatible? | [FAM-DATASET-SUPERVISION-BOUNDARY](dataset-supervision-boundary.md) |
| What is the baseline floor? | [FAM-FOUNDATIONAL-TEXT-TO-POSE-BASELINES](foundational-text-to-pose-baselines.md) |
| What intermediate representation should be generated or conditioned on? | [FAM-LEARNED-POSE-MOTION-REPRESENTATIONS](learned-pose-motion-representations.md) |
| How should sign-relevant articulator structure be preserved? | [FAM-STRUCTURE-AWARE-ARTICULATOR-MODELING](structure-aware-articulator-modeling.md) |
| How should sign motion be generated from latent structure? | [FAM-LATENT-GENERATIVE-PRODUCTION](latent-generative-production.md) |
| How should generated motion stay semantically aligned with text? | [FAM-TEXT-POSE-SEMANTIC-ALIGNMENT](text-pose-semantic-alignment.md) |
| What serious alternatives exist to direct neural generation? | [FAM-RETRIEVAL-STITCHING-PRIMITIVES](retrieval-stitching-primitives.md) |
| Which useful sources are supervision-incompatible by default? | [FAM-GLOSS-NOTATION-DEPENDENT-BOUNDARY](gloss-notation-dependent-boundary.md) |
| What frontier output surfaces should be tracked but not forced into scope? | [FAM-THREED-AVATAR-PARAMETRIC-FRONTIER](threed-avatar-parametric-frontier.md) |
| How should candidates be evaluated and compared? | [FAM-EVALUATION-BENCHMARK-METHODOLOGY](evaluation-benchmark-methodology.md) |

## Why Ten Families?

Ten families are used because the refreshed corpus separates into ten audit questions that should
not be collapsed without losing scientific precision. The split prevents representation evidence
from being confused with generation architecture, structure preservation, semantic alignment,
evaluation design, supervision boundary, or frontier visualization pressure.

Fewer families would merge scientifically different questions. More families would fragment the
audit and risk turning the candidate universe into a source-by-source literature list. The
survey/background source remains in the source corpus as background unless it changes a downstream
audit decision; it does not require a separate background family in the current refresh.

## Family Type / Status Labels

| Status | Meaning |
| --- | --- |
| `primary_candidate_family` | May produce concrete candidate cards later. |
| `counter_alternative_family` | Serious competing family kept to avoid premature narrowing. |
| `boundary_family` | Defines dataset, supervision, baseline, technical, or compatibility boundaries. |
| `evaluation_family` | Defines evaluation and benchmark methodology. |
| `frontier_watch_family` | Tracks frontier pressure or future-work direction without immediate implementation commitment. |
| `background_context_family` | Reserved for taxonomy or coverage context; not instantiated in the current 10-family refresh. |

## Instantiated Family Records

### Boundary Families

- [Dataset and Supervision Boundary](dataset-supervision-boundary.md): `boundary_family`
- [Foundational Text-to-Pose Baselines](foundational-text-to-pose-baselines.md): `boundary_family`
- [Gloss/Notation-Dependent Pose Generation Boundary](gloss-notation-dependent-boundary.md): `boundary_family`

### Primary Candidate Families

- [Learned Pose/Motion Representations](learned-pose-motion-representations.md): `primary_candidate_family`
- [Structure-Aware and Articulator-Aware Modeling](structure-aware-articulator-modeling.md): `primary_candidate_family`
- [Latent Generative Production](latent-generative-production.md): `primary_candidate_family`
- [Text-Pose Semantic Alignment and Conditioning](text-pose-semantic-alignment.md): `primary_candidate_family`

### Counter-Alternative Families

- [Retrieval, Stitching, and Motion-Primitives Alternatives](retrieval-stitching-primitives.md): `counter_alternative_family`

### Frontier-Watch Families

- [3D Avatar and Parametric Motion Frontier](threed-avatar-parametric-frontier.md): `frontier_watch_family`

### Evaluation Families

- [Evaluation and Benchmark Methodology](evaluation-benchmark-methodology.md): `evaluation_family`

## Downstream Use

Candidate cards may instantiate concrete candidate directions from primary candidate families.
Counter-alternative families may instantiate comparison or alternative candidate directions if
justified. Frontier-watch families may inform positioning or future work, but should not be
treated as near-term implementation scope by default.

Boundary families constrain direct-support claims and adaptation requirements. The evaluation
family constrains scorecard and selection-decision interpretation. Scorecards and selection
decisions must distinguish model evidence from boundary, evaluation, counter-alternative, and
frontier-watch evidence.

## Record Status

- Family-level records are instantiated for the refreshed audit universe.
- Family-level instantiation does not mean that any family has been selected, rejected, scored, or assigned as a contribution.
- [Candidate Cards](../candidate-cards/index.md), [Scorecards](../scorecards/index.md),
  [Selection Decisions](../selection-decisions/index.md), and [Audit Result](../audit-result.md)
  remain downstream audit surfaces and are not recorded here.
