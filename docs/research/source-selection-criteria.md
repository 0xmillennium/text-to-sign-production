# Source Selection Criteria

## Purpose

This document defines the candidate-agnostic literature source-selection standard for the
project's literature review. It exists upstream of contribution selection and is intentionally
independent of any existing contribution-selection outcome.

A source is included when it provides verifiable evidence relevant to English text-to-sign-pose
production, sign-pose or motion representation, gloss-free or gloss-adaptable supervision, dataset
compatibility, evaluation methodology, or reproducible implementation. Inclusion is independent of
any existing contribution choice; sources may support, weaken, replace, or reopen later
contribution decisions.

The standard is used to find, include, exclude, classify, and reassess sources. It is not a
literature review, an updated bibliography, or a contribution-ranking document.

## Non-Retrospective Use

This document must not be used retroactively to justify an already-selected `C1/C2` pair. Source
inclusion is based on relevance to the project's research problem and data constraints, not on
agreement with existing candidate choices.

Later contribution-selection records must use this criteria document as an upstream standard.
Existing contribution decisions may be preserved, revised, replaced, or reopened after the updated
source corpus is reviewed under this standard.

## Project Research Frame

The project research frame is:

- English text input.
- Sign-pose, keypoint, skeleton, motion representation, or avatar-compatible downstream output.
- Sign-language production, not sign-language recognition.
- Gloss-free supervision at the linguistic-supervision level.
- How2Sign-style data compatibility as the practical anchor, not an absolute exclusion rule.
- Reproducible thesis engineering as a core standard.

This document does not select `C1`, does not select `C2`, and does not rank contribution
candidates.

## Dataset and Supervision Constraints

The primary practical data regime is How2Sign-style supervision: English text/transcript, sign
video, and extracted keypoints or pose.

The project does not assume access to public gloss annotations. How2Sign compatibility is a strong
preference because it matches the current practical data anchor, but it is not an absolute
requirement.

Non-How2Sign sources may be included when their method, evidence, or evaluation design transfers
clearly to the project's available data regime. Gloss-based sources are not automatically excluded,
but their gloss dependency must be classified.

## Gloss-Free Policy

The project is gloss-free at the linguistic-supervision level. Manual gloss annotation is not
assumed. This does not prohibit intermediate representations when they are derived from pose,
motion, learned latent structure, or similar non-gloss supervision.

Pose keypoints, skeletons, learned pose tokens, latent motion units, articulator channels, and
motion representations are not gloss by default. A method that requires manually annotated gloss
sequences must not be treated as directly compatible unless a plausible gloss-free adaptation path
is documented.

| Representation | Gloss status | Notes |
| --- | --- | --- |
| Manual gloss annotation | Gloss | Linguistic label sequence created by human annotation. |
| Text transcripts | Not gloss | Spoken-language input or supervision text, not a sign gloss sequence. |
| Pose keypoints | Not gloss by default | Numeric body, hand, or face locations extracted from video. |
| Skeleton | Not gloss by default | Structured pose topology or joint sequence. |
| Learned pose tokens | Not gloss by default | Data-derived units learned from pose or motion. |
| Latent pose/motion units | Not gloss by default | Non-linguistic latent structure derived from motion data. |
| Articulator channels | Not gloss by default | Body-part or channel grouping for pose or motion. |
| Motion primitives | Not gloss by default | Motion units are acceptable when not dependent on manual gloss labels. |

## Inclusion Criteria

Hard criteria:

1. Full paper or equivalent complete technical source is accessible.
2. The source is related to sign language production, text-to-sign, text-to-pose, sign pose
   generation, sign motion representation, production evaluation, or a clearly transferable method.
3. Method, dataset, evaluation, and limitations can be extracted.
4. The source can be related to the project's data regime: text, video, pose, keypoint, and
   gloss-free or gloss-adaptable supervision.
5. The source can affect at least one downstream research decision: baseline design,
   representation strategy, supervision assumptions, evaluation design, dataset boundary,
   reproducibility standard, or later contribution selection.

Soft preference criteria:

- English input.
- ASL or How2Sign use.
- Public or research-accessible dataset.
- Pose, keypoint, skeleton, or motion output.
- Code or artifact availability.
- Ablation evidence.
- Baseline comparison.
- Human evaluation or pose-based evaluation.
- Recent publication date.
- Peer-reviewed venue, while allowing arXiv or preprint sources when technically important.

## Exclusion Criteria

Exclusion does not mean the source is scientifically poor. It only means the source is not part of
the project's decision-bearing corpus.

- `recognition_only`: addresses sign-language recognition without production relevance.
- `translation_only`: addresses text or gloss translation without pose, motion, generation, or
  transferable production evidence.
- `generic_pose_only`: addresses generic human pose generation without clear transfer to sign
  language production.
- `gesture_only_no_sign_language`: addresses gesture generation without sign-language scope or
  transfer.
- `rendering_only`: focuses only on rendering, avatar display, or visualization without source-side
  production, pose generation, evaluation, or reproducibility relevance.
- `no_full_paper`: no full paper or equivalent complete technical source is accessible.
- `method_insufficient`: method, dataset, evaluation, or limitations cannot be extracted.
- `dataset_incompatible_no_transfer`: dataset assumptions are incompatible and no transfer path to
  text/video/pose/keypoint supervision is clear.
- `gloss_required_no_adaptation_path`: requires manually annotated gloss sequences and gives no
  plausible gloss-free adaptation path.
- `duplicate_or_superseded`: duplicates another reviewed source or is superseded by a more complete
  version.
- `not_decision_relevant`: does not affect baseline design, representation strategy, supervision,
  evaluation, dataset boundary, reproducibility, or later contribution selection.

## Evidence Classification

Sources are classified by evidence type, not by whether they support existing candidates.

- `direct_task_evidence`: directly studies text-to-sign, text-to-sign-pose, sign pose generation,
  or sign-language production.
- `dataset_evidence`: informs dataset fit, data preprocessing, annotation assumptions, split
  handling, or source-data limitations.
- `representation_evidence`: informs pose, skeleton, keypoint, token, latent, motion, or
  avatar-compatible representations.
- `architecture_evidence`: informs model families or training architectures transferable to the
  project.
- `evaluation_evidence`: informs automatic, human, pose-based, back-translation, or qualitative
  evaluation design.
- `reproducibility_evidence`: informs code, artifact, dataset, checkpoint, license, or re-run
  feasibility expectations.
- `negative_evidence`: gives evidence that weakens a method family, assumption, dataset fit, or
  evaluation strategy.
- `background_evidence`: provides context needed to understand the task, field, or constraints but
  does not directly determine a contribution choice.

## Dataset Compatibility Labels

- `how2sign_direct`: uses How2Sign directly.
- `how2sign_compatible`: does not use How2Sign but uses a compatible text, video, pose, keypoint,
  or skeleton supervision pattern.
- `asl_compatible`: uses ASL or evidence likely to transfer to the project's English-input,
  ASL-oriented data regime.
- `cross_lingual_transferable`: uses another sign language or language pair with a clear transfer
  argument for method, representation, or evaluation.
- `dataset_private_transferable`: uses private data but provides enough method or evaluation detail
  to transfer to the project.
- `dataset_incompatible`: depends on data unavailable to this project and has no clear transfer
  path.
- `dataset_unclear`: dataset, language, annotation, or supervision details are not clear enough to
  classify.

## Gloss Dependency Labels

- `gloss_free`: does not require manual gloss sequences at training, inference, or evaluation time.
- `gloss_optional`: can use glosses, but the main method can operate without them.
- `gloss_adaptable`: uses glosses in the source setting, but a plausible gloss-free adaptation path
  can be documented.
- `gloss_dependent`: requires manually annotated gloss sequences and is not directly compatible
  with the current supervision assumptions.
- `gloss_unclear`: gloss usage cannot be determined from the reviewed source.

## Search Protocol

Search must start from the research problem, not from existing candidate choices. Existing
candidate choices must not be used as the starting point for search.

Source venues:

- ACL Anthology.
- CVF / OpenAccess.
- arXiv.
- ACM Digital Library.
- IEEE Xplore.
- Google Scholar for discovery only.
- Official project pages for artifact verification.
- GitHub repositories for artifact verification only.

Neutral query families:

- `"text to sign language production"`
- `"text to sign pose generation"`
- `"text-to-pose sign language"`
- `"gloss-free sign language production"`
- `"sign language production keypoints"`
- `"sign language generation skeleton"`
- `"sign pose generation"`
- `"sign language production evaluation"`
- `"pose-based sign language evaluation"`
- `"How2Sign sign language production"`
- `"ASL text to pose generation"`
- `"sign language motion generation"`
- `"sign language generation without gloss"`
- `"sign language production transformer"`
- `"sign language production diffusion"`
- `"retrieval sign language production"`
- `"data-driven sign language representation"`

Search order:

1. Broad task-level queries.
2. Dataset-specific queries.
3. Evaluation queries.
4. Representation queries.
5. Architecture or method-family queries.
6. Candidate-specific queries only after the broad search pass.

## Source Review Requirements

Each reviewed source must remain reviewable against the following extraction checklist:

- Title.
- Authors.
- Year.
- Venue or archive.
- PDF reviewed.
- DOI / arXiv / URL.
- Code availability.
- Dataset.
- Language / sign language.
- Input.
- Output.
- Gloss usage.
- Artifact availability.
- Evidence class.
- Dataset compatibility.
- Gloss dependency.
- Task fit.
- Representation fit.
- Evaluation relevance.
- Reproducibility relevance.
- Method summary.
- Evaluation summary.
- Limitations and risks.
- Project-relevant evidence.
- Reason for inclusion or exclusion.
- Potential impact on later contribution selection.

This checklist defines the full review standard. The public source corpus is the canonical
registry for source identity, role, compatibility, and decision relevance; it is not required to
duplicate every detailed extraction field from this checklist. Detailed extraction metadata may be
recorded in separate per-source detail records or future review artifacts when the project needs
that level of public traceability. Public research claims should remain traceable through
source-corpus IDs and downstream audit records.

## PDF-Based Review Standard

Included decision-bearing sources require PDF or full-paper review. Abstract-only, landing-page,
repository-only, or citation-only inspection is insufficient for decision-bearing inclusion.

Minimum extracted items:

- Bibliographic metadata.
- Dataset.
- Language/sign language.
- Input/output.
- Gloss dependency.
- Method summary.
- Evaluation setup.
- Baseline comparisons.
- Ablations, if present.
- Artifact availability.
- Limitations.
- Project decision impact.

## Bias-Control Rules

1. Existing contribution decisions must not be used as inclusion criteria.
2. Sources supporting existing preferred directions must not receive automatic priority.
3. Counter-evidence and competing method families must be included when they meet scope criteria.
4. Gloss-based sources must be classified, not automatically discarded.
5. How2Sign mismatch lowers direct applicability but does not automatically exclude a source.
6. A source may be included because it challenges the current project direction.
7. Search must start broad before method-specific queries are used.
8. Source exclusion must be recorded with a reason label.
9. A source must not be upgraded in relevance merely because it matches an existing selected
   candidate.
10. A source must not be downgraded merely because it suggests a different contribution direction.

## Recency and Refresh Policy

Recency categories:

- `foundational`: older source that remains necessary for task framing, baseline context, dataset
  understanding, or evaluation interpretation.
- `current_core`: source that directly informs the current source corpus and decision-bearing
  research frame.
- `recent_refresh`: newer source reviewed to detect changed assumptions, stronger evidence, or new
  method families.
- `superseded`: source replaced by a more complete, corrected, or technically current version.

Recent sources must be reviewed before treating literature refresh closure as stable, especially
when they affect:

- Gloss-free production.
- Text-to-pose modeling.
- Sign-pose representation.
- Production evaluation.
- Reproducibility expectations.
- Alternative method families.

This document defines criteria only. It does not cite, identify, or add specific refreshed papers.

## Reproducibility Fields

The full review standard includes the following reproducibility fields:

- Code availability: `available`, `partial`, `not_found`, or `not_applicable`.
- Data availability: `public`, `request_required`, `private`, or `unclear`.
- Model/checkpoint availability: `available`, `not_available`, or `unclear`.
- License clarity: `clear`, `restricted`, or `unclear`.
- Re-run feasibility: `high`, `medium`, or `low`.

## Relationship to Later Contribution Selection

This file does not select contributions. This file does not preserve previous contribution
decisions.

Later contribution-audit documents must use this source-selection standard. Updated source
evidence may support, weaken, replace, or reopen contribution choices. Contribution choices should
be based on the updated source corpus, not the other way around.

## Maintenance Policy

This file must be revisited:

- Before a new literature refresh.
- When adding new decision-bearing sources.
- When the dataset regime changes.
- When gloss annotations become available or unavailable.
- Before reopening contribution selection.
- Before declaring research audit closure complete.
