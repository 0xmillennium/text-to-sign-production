# Execution

## Purpose

This file explains how to run the project through the supported Colab notebook from setup to final
artifact inspection.

The notebook is the operational source of truth. Follow it in order when producing Dataset Build
outputs or Baseline Modeling runtime evidence.

This is an operator guide. It is not an artifact schema reference. Artifact paths, contents,
producers, consumers, and schema details live in [docs/data](data/index.md).

## Supported Workflow

The single supported workflow is the project-wide Colab notebook:

`notebooks/colab/text_to_sign_production_colab.ipynb`

Current public stage: Dataset Build.

Implemented internal downstream surface: Baseline Modeling.

Broader evaluation, contribution modeling, and playback/demo are not part of the supported workflow
yet.

## Before You Start

Use a Colab runtime and mount Google Drive when the notebook asks for it. The workflow assumes the
fixed Drive layout used by the notebook.

Run the notebook in order. Do not jump into Dataset Build, Baseline Modeling, qualitative export, or
final inspection cells in a fresh runtime.

The install step requires a runtime restart. After restart, rerun the setup path before continuing:

- Section 0 runtime settings and tools
- Section 1 Drive mount
- Section 2 repository acquisition, environment configuration, and shared helper loading

Continue only after the environment cells have run successfully in the active runtime.

## Notebook Execution Order

### Section 0: Runtime And Session Setup

This section defines the session settings and prepares runtime tools.

Completing it gives the operator a known notebook configuration. Later cells use these values for
the repository checkout, Dataset Build splits, Baseline Modeling run name, and qualitative panel
size.

It matters because every later section depends on the same session choices. It unlocks Drive setup,
repository setup, and all output-producing cells.

- `0.1 Runtime Settings`: set the repository target, split list, baseline run name, and qualitative
  panel size before running later cells.
- `0.2 Runtime Tool Setup`: ensures `zstd` is available for `.tar.zst` archives used by extraction
  and publication.

### Section 1: Drive Mount

This section mounts Google Drive at the path expected by the notebook.

Completing it gives the operator access to raw inputs, published Dataset Build archives, and
Baseline Modeling run roots. It matters because Colab runtimes are temporary while Drive persists
the large artifacts.

It unlocks repository acquisition, fixed raw input access, archive extraction, and archive
publication.

- `1.1 Mount Google Drive`: connect the runtime to the project Drive layout.

### Section 2: Repo Acquisition And Install

This section prepares the repository and Python environment used by all later cells.

Completing it gives the operator an importable worktree at the requested revision with the notebook
dependencies installed. It matters because Dataset Build and Baseline Modeling are run through the
repository modules, while the notebook stays an operational surface.

It unlocks shared helper loading, Dataset Build output resolution, and Baseline Modeling cells.

- `2.1 Acquire Repository`: place the requested repository revision in the Colab worktree.
- `2.2 Install Repository Requirements`: install dependencies. This step requires a runtime restart.
- `2.3 Configure Python Environment`: set the working directory, import path, and visible revision.
  Run this again after every restart.
- `2.4 Load Shared Workflow Helpers`: load constants and archive helpers needed by reuse, extract,
  run/publish, and inspection cells.

### Section 3: Dataset Build Outputs

This section resolves the Dataset Build outputs needed by the rest of the notebook.

Completing it gives the operator processed train, validation, and test surfaces. It matters because
Baseline Modeling does not consume raw How2Sign inputs. It consumes processed manifests and sample
payloads produced by Dataset Build.

It unlocks Baseline Modeling run preparation and training.

- `3.1 Reuse Extracted Dataset Build Outputs`: checks whether processed Dataset Build outputs
  already exist in the worktree.
- `3.2 Extract Archived Dataset Build Outputs`: restores processed outputs from published Drive
  archives when they already exist.
- `3.3 Run Dataset Build And Publish Outputs`: generates Dataset Build outputs fresh from the fixed
  Drive raw inputs and publishes archives.

Run/publish is the step that creates the processed training, validation, and test surfaces used by
all later modeling work. It gives the project a reproducible processed dataset boundary, not just
another output folder. The published archives make that boundary reusable after runtime resets and
available to downstream Baseline Modeling without repeating the expensive raw-data pass.

### Section 4: Baseline Modeling Training Outputs

This section checks the modeling runtime, prepares the run root, and resolves training outputs.

Completing it gives the operator a prepared baseline run plus training outputs. It matters because
later qualitative and record steps need real checkpoints, summaries, and configuration evidence.

It unlocks qualitative panel export.

- `4.0 Modeling/GPU Sanity Check`: verifies the modeling imports and CUDA device before training.
- `4.1 Prepare Baseline Run Root`: validates processed Dataset Build train/validation inputs and
  writes baseline config files into the run root.
- `4.2 Reuse Extracted Training Outputs`: checks whether checkpoints and summaries already exist.
- `4.3 Extract Archived Training Outputs`: restores training outputs from the run root archive.
- `4.4 Run Training And Publish Outputs`: trains the baseline and publishes the training archive.

Run/publish creates the actual baseline checkpoints and scalar outcome summaries. This is the point
where Dataset Build outputs become model evidence. The checkpoints support qualitative export, while
the summaries make later record/package assembly meaningful and citable. Publishing the archive also
keeps the trained baseline recoverable across Colab runtime resets.

### Section 5: Qualitative Panel Outputs

This section resolves qualitative panel outputs for a trained baseline.

Completing it gives the operator a compact comparison surface between model output and reference
samples. It matters because scalar summaries alone do not show what the baseline produces on
specific validation examples.

It unlocks runtime record/package assembly.

- `5.1 Reuse Extracted Qualitative Panel Outputs`: checks whether qualitative outputs already exist.
- `5.2 Extract Archived Qualitative Panel Outputs`: restores qualitative outputs from the run root
  archive.
- `5.3 Run Qualitative Panel Export And Publish Outputs`: exports the validation panel and publishes
  the qualitative archive.

Run/publish produces the review surface that connects a trained checkpoint to concrete examples. It
is more than another folder because it preserves the model/reference comparison used to inspect
baseline behavior. The published archive lets later record assembly cite the qualitative evidence
without rerunning export.

### Section 6: Record/Package Outputs

This section resolves runtime record/package outputs for the baseline run.

Completing it gives the operator a coherent runtime evidence package. It matters because training
outputs and qualitative outputs need to be tied together before they can be handed off or cited.

It unlocks final artifact inspection and copy-back into experiment records or handoff notes.

- `6.1 Reuse Extracted Record/Package Outputs`: checks whether record/package outputs already exist.
- `6.2 Extract Archived Record/Package Outputs`: restores record outputs from the run root archive.
- `6.3 Assemble/Package And Publish Record Outputs`: gathers runtime evidence and publishes the
  record package archive.

Run/publish gathers the runtime-side evidence needed to preserve the run coherently. It connects the
configuration, checkpoints, summaries, qualitative panel, and evidence files into a single package
surface. This supports later citation and comparison because the run can be described as one
traceable artifact set instead of scattered runtime files.

### Section 7: Final Artifact Inspection

This section prints the resolved artifact surfaces from the notebook run.

Completing it gives the operator a concise list of outputs to cite or copy into handoff notes. It
matters because successful runs should end with visible, checkable evidence of what was produced and
where it was published.

It unlocks clean handoff to artifact inspection, experiment records, or later project stages.

- `7.1 Inspect Final Artifact Paths`: prints key extracted outputs, archived outputs, and formal
  record inputs.

## Reuse / Extract / Run-Publish Semantics

After Dataset Build outputs, large output-producing phases use the same execution model.

- Reuse means use already extracted outputs in the expected worktree or run root.
- Extract means restore outputs from an existing published archive.
- Run/Publish means generate the outputs fresh and publish the archive.

This model avoids unnecessary recomputation. It supports resuming work after interrupted sessions.
It uses Drive-backed archives to survive Colab runtime resets. It also keeps later notebook steps
unblocked when valid outputs already exist.

## Restart Rule

The install step requires a runtime restart.

After restart, rerun the environment and setup path before resuming later sections. At minimum, rerun
the runtime settings, Drive mount, environment configuration, and shared helper loading cells.

Restarting is expected behavior. It is not a failure condition.

## Expected Guard Errors

Missing-archive failures in extract cells are expected guard behavior.

They mean there is nothing to restore yet. Move to the corresponding run/publish step.

These guard errors are not workflow bugs by themselves.

## Artifact Details

Artifact structure, paths, archive contents, schemas, producers, and consumers are documented in
[docs/data](data/index.md).

Use this guide to operate the notebook. Use [docs/data](data/index.md) to inspect artifact details.

## After A Successful Run

At the end of a successful full notebook run, the operator has the major output surfaces needed for
handoff and inspection:

- Dataset Build outputs
- Baseline config files
- checkpoints and summaries
- qualitative outputs
- record/package outputs
- published archives

For exact structure and artifact meaning, use [docs/data](data/index.md).
