# Data Leaf Template

This file defines the canonical standard for `docs/data` leaf pages. Each leaf page represents one
canonical artifact surface. An artifact surface may be a single file, a file family, a directory
surface, or an archive bundle.

## Required Section Order

Every leaf page must use these exact headings in this order:

1. `Purpose`
2. `Artifact Role`
3. `Artifact Unit`
4. `Canonical Path Pattern`
5. `Verified Example Path`
6. `Produced By`
7. `Consumed By`
8. `Lifecycle`
9. `Structure / Contents`
10. `Validation`
11. `Related Artifacts`
12. `Related Docs`

## Artifact Role Vocabulary

Use only these values in `Artifact Role`:

- `input`
- `output`
- `intermediate`
- `archive`
- `manifest`
- `config`
- `checkpoint`
- `summary`
- `evidence`
- `package`

List more than one role only when the artifact surface genuinely has more than one role.

## Artifact Unit Vocabulary

Use only these values in `Artifact Unit`:

- `single file`
- `directory surface`
- `file family`
- `archive bundle`

## Path-Writing Rules

- `Canonical Path Pattern` must include a reusable pattern with placeholders such as `<split>` or
  `<run_name>` when a stable pattern exists.
- Wrap angle-bracket placeholders in inline code so Markdown renders them visibly.
- `Verified Example Path` should include at least one concrete path from verified inventory or
  repo-observed constants. If only a verified directory/root is available, state that clearly.
- Distinguish raw external input paths, worktree extracted paths, and Drive-published artifact or
  archive paths.
- Use absolute Colab paths for Drive and worktree runtime surfaces when those are the canonical
  runtime paths.
- Use repo-relative paths for archive member paths or compact internal references.

## Producer And Consumer Rules

- `Produced By` must name the workflow step or external source that creates the surface.
- `Consumed By` must name the workflow step that reads the surface.
- Use notebook section references where they are known, such as Section 3.3 or Section 5.3.
- Do not use vague language when a specific section or step is known.

## Related Artifact Rules

- Use `Related Artifacts` to link neighboring `docs/data` leaf pages.
- Prefer links over repeating structure details already documented in another leaf page.
- Do not add new artifact categories from related links alone.

## Related Docs Rules

- Use `Related Docs` for non-leaf documentation that helps operate or interpret the artifact
  surface.
- Do not restate execution, storage, experiment, or roadmap content inside `docs/data` leaf pages.
