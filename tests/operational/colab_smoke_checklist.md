# Colab Smoke Checklist

Status: obsolete after Phase 1 archive infrastructure removal; pending notebook rewrite.

Do not use this checklist for release signoff. It previously described the old Dataset Build Colab
archive copy, extraction, packaging, and publish behavior that depended on repository Python archive
helpers. Those helpers have been removed, and the final post-Phase-1 notebook extraction standard is
not defined here.

## Limited Transition Use

- Use this file only to record that the Dataset Build Colab smoke procedure needs a rewrite.
- Manual operators may still validate basic external conditions such as Colab startup, Drive mount,
  repository acquisition, and access to private raw How2Sign inputs.
- Do not treat the old archive copy, extraction, packaging, or publish steps as current
  requirements.
- Record any current notebook/runtime mismatch as deferred notebook-rewrite work instead of fixing
  it by reintroducing Python archive helpers.

## Deferred Rewrite Scope

The replacement checklist should be written with the final notebook architecture. It should validate
only the operational behavior that is actually supported at that point.
