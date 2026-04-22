# Candidate Universe

This surface stores family-level audit-universe records. It is the family-level input surface for candidate derivation and family-level audit framing, not the candidate-analysis surface itself. Family-level instantiation does not mean contribution selection.

## Record Types

Family-level records classify the audited family as one of the following:

- `primary_candidate`
- `counter_alternative`
- `wildcard`

The distinction is:

- `primary_candidate`: a family retained as part of the main contribution universe under audit
- `counter_alternative`: a family retained as a mandatory alternative family for serious evaluation
- `wildcard`: a tightly controlled additional family slot, if explicitly authorized

## Instantiated Family Records

- [Discrete / Data-Driven Representation](discrete-data-driven-representation.md): `primary_candidate`
- [Structure-Aware / Multi-Channel Improvement](structure-aware-multi-channel-improvement.md): `primary_candidate`
- [Retrieval / Stitching](retrieval-stitching.md): `counter_alternative`
- [Diffusion-Based Generation](diffusion-based-generation.md): `counter_alternative`
- [Controlled Wildcard Slot](controlled-wildcard-slot.md): `wildcard`

## Record Status

- Family-level records are instantiated for the frozen audit universe.
- Family-level instantiation does not mean that any family has been selected, rejected, or assigned to `C1` or `C2`.
- [Candidate Cards](../candidate-cards/index.md), [Scorecards](../scorecards/index.md),
  [Selection Decisions](../selection-decisions/index.md), and [Audit Result](../audit-result.md)
  remain downstream audit surfaces and are not recorded here.
