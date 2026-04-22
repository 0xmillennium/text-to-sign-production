# Controlled Wildcard Slot

## Family ID

`controlled-wildcard-slot`

## Family Name

Controlled Wildcard Slot

## Family Type / Status

`wildcard`

## Scope Definition

This record is not a real method family. It is a reserved slot for at most one tightly controlled additional family if explicit justification later requires inclusion of such a family in the audit universe.

## Included Directions / Subfamilies

No concrete family is instantiated in this slot.

## Explicit Exclusions / Out-of-Scope Directions

- no concrete wildcard family is instantiated in this file
- no candidate-card leaf record is derived from this slot
- no score, veto outcome, or selection outcome is recorded in this file
- no family or candidate is assigned to `C1` or `C2` through this slot
- no evaluation record is instantiated from this slot

## Rationale For Inclusion In The Audit Universe

This slot exists only as a controlled reservation mechanism. It preserves a narrow path for one additional family-level addition if explicit justification requires it, without silently expanding the frozen audit universe.

## Relation To The Overall Audit

This record preserves the accepted wildcard reservation at the family level only. It may be activated only by explicit justification. Its presence in the audit universe does not mean that any additional family has already been admitted.

## Downstream Record Policy

No downstream [Candidate Cards](../candidate-cards/index.md) may be created from this slot unless a
concrete family-level record is explicitly authorized and instantiated. Until then, this slot
remains a reservation mechanism only.

## Record Status

- This slot is instantiated as a `wildcard` reservation.
- The slot is currently empty.
- No concrete wildcard family is instantiated in this record.
