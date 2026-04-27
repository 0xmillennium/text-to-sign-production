# How2Sign Dataset Boundary

## Purpose

This page records the How2Sign access, license, redistribution, citation, and supervision
assumptions used by this project.

It is a dataset boundary page, not an implementation guide. Data pipeline behavior, artifact
schemas, and model workflows remain documented in their own data, execution, and code surfaces.

## Official Source

- Official URL: [https://how2sign.github.io/](https://how2sign.github.io/)
- How2Sign is a multimodal and multiview continuous American Sign Language (ASL) dataset.
- The official page describes the dataset as more than 80 hours of sign-language video with
  corresponding modalities.
- The official page notes that some How2-derived data cannot be redistributed by the How2Sign
  maintainers; users may need to obtain original How2 data through the original How2 repository.
- Re-check the official page for current access, license, redistribution, and citation terms before
  publishing or sharing any How2Sign-related artifact.

## Used Files

The current local workflow is based on these official How2Sign download labels:

- `B-F-H 2D Keypoints clips* (frontal view)`
- `English Translation (manually re-aligned)`

These labels identify the relevant official How2Sign files for the current local workflow. The
current local path is keypoint/text centered: raw RGB video is not the primary required local
modality for the documented workflow. If raw video is used later, it must follow the same
access, license, and redistribution boundaries as the current keypoint and text inputs.

## Local Access And Storage

Required How2Sign files were downloaded locally from the official distribution. The official
archives were extracted and re-archived as `.tar.zst` archives for Colab usability.

The local `.tar.zst` archives are stored in private Google Drive. These private Drive archives are
not public repository artifacts and are not redistributed by this repository.

Public docs should use placeholders such as `<COLAB_DRIVE_PROJECT_ROOT>` or
`<PROJECT_ARTIFACT_ROOT>` instead of personal Drive paths.

The main Colab notebook is an orchestration and navigation surface. Implementation belongs under
`src/`.

## License And Attribution

The repository code license is separate from dataset terms. The repository code can be
MIT-licensed, but How2Sign data and restricted dataset-derived artifacts are governed by How2Sign
terms, not by the MIT code license.

Dataset license: Creative Commons Attribution-NonCommercial 4.0 International License.
The official How2Sign page also says the dataset is available for research purposes only.
Commercial use is not allowed under the dataset terms.

How2Sign use requires attribution and citation according to the official page. The concise
citation note for this project is: Duarte et al., "How2Sign: A Large-scale Multimodal Dataset for
Continuous American Sign Language", CVPR 2021. Use the official page for the exact current
citation text.

## Redistribution Boundary

| Artifact class | Public status | Rationale |
| --- | --- | --- |
| Official raw How2Sign files | Not public; not redistributed | Official dataset files remain under How2Sign access and license terms. |
| Project-local `.tar.zst` re-archives | Private/local only | Re-archives are convenience copies for Colab use, not public repository artifacts. |
| Extracted archive contents | Restricted; not public by default | Extracted files preserve restricted How2Sign payloads. |
| English translation files | Restricted; not public by default | These files come from the How2Sign distribution and remain governed by dataset terms. |
| B-F-H 2D keypoint files | Restricted; not public by default | Keypoints are dataset-derived modality files from the official distribution. |
| Processed manifests | Conditional; public only if cleared | Manifests must preserve source traceability and may include restricted dataset-derived fields. |
| Processed training/evaluation samples | Restricted unless explicitly cleared | Samples are dataset-derived payloads and are not assumed to be redistributable. |
| Reports and aggregate metadata | May be public if payload-free | Reports can be public only when they contain no restricted examples, text, keypoints, media, or derived payloads. |
| Synthetic or tiny test fixtures | Public-safe when non-derived | Fixtures may be public if they are synthetic or otherwise not derived from restricted How2Sign data. |
| Model checkpoints | Not claimed as public release artifacts | Checkpoint release rights for How2Sign-trained or How2Sign-derived runs remain unresolved. |
| Qualitative exports | Restricted inspection artifacts by default | Visual or sample exports are for inspection and are not proof of sign-language intelligibility. |

## Supervision Boundary

| Signal | Current boundary | Rationale |
| --- | --- | --- |
| English translation text | Usable for local research under dataset terms | This is an official translation modality used by the local workflow. |
| Segment timing / manually re-aligned timestamps | Usable for local research under dataset terms | Timing comes from the manually re-aligned translation files and supports local alignment. |
| B-F-H 2D keypoints | Usable for local research under dataset terms | Keypoints are the primary pose modality for the current documented workflow. |
| Raw RGB video | Not primary for current local workflow; restricted if used | Raw video is not required by the current keypoint/text path and remains under dataset terms. |
| Gloss annotations | No public gloss dependency; do not release through this repository | Gloss annotations are not a public dependency for this project and must not become a public payload. |
| Human intelligibility labels | Not currently assumed | Automatic metrics or visual inspection do not create human intelligibility evidence. |
| Qualitative visual exports | Inspection only | Qualitative exports can help debugging, but they are not proof of sign-language intelligibility. |

## How2Sign-compatible Does Not Mean

- this repo redistributes How2Sign,
- this repo has unrestricted publication rights over dataset-derived artifacts,
- the MIT code license applies to How2Sign data,
- automatic metrics or visual plausibility prove sign-language intelligibility,
- a public model artifact has been released.

## Handoff To Data Artifacts

Data artifact docs may rely on the access, license, redistribution, citation, and supervision
boundaries above without restating them.

Raw media and restricted dataset-derived artifacts remain outside Git. Processed artifacts must
preserve source traceability. Public-safe fixtures must be synthetic or otherwise non-restricted.
Artifact schemas and implementation details live in the appropriate data and code docs.

## Open Questions And Review Triggers

- Whether any dataset-derived processed samples can ever be public.
- Whether model checkpoints trained on How2Sign-derived artifacts can be released.
- Whether qualitative exports are redistributable.
- Whether official How2Sign terms change.
- Re-review this page before public artifact release, dataset redistribution, checkpoint release,
  qualitative export publication, commercial use, or any change in official How2Sign terms.

## Related Documentation

- [Data Artifact Dictionary](data/index.md)
- [Execution](execution.md)
- [Experiments](experiments/index.md)
- [Testing](testing/index.md)
