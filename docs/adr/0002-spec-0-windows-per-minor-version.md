# 0002. Count SPEC 0 dependency windows per minor version

- **Status:** Proposed
- **Date:** 2026-10-11
- **Issues:** #77, #78

## Context

[ADR 0001](0001-spec-0-support-policy.md) decisions 2 and 3 set dependency
floors by a 2-year window, but don't say what the window is counted over. Two
readings give different floors:

- **Per release:** the floor is the oldest *release* (any patch) from the last
  2 years. In 2026 Q4 this gives `igraph>=0.11.8` (released 2024-10-28).
- **Per minor version:** the floor is the oldest *minor version* whose x.y.0
  release came out in the last 2 years. igraph 0.11's first release was
  2023-10-12, so 0.11 is out and the floor is `igraph>=1.0` (2025-10-23).

[SPEC 0](https://scientific-python.org/specs/spec-0000/) uses the
per-minor-version reading: its drop schedule lists minor versions by the date of
their x.y.0 release. spec0-action, which #78 uses to automate the quarterly
bumps, does the same.

## Options considered

- **Per release.** Floors can land on a patch release. This is not what SPEC 0
  or spec0-action computes, so the automated bumps (#78) would disagree with
  the floors set by hand.
- **Per minor version.** Floors move in bigger steps (igraph skips from
  0.11.x straight to 1.0), but they match SPEC 0 and the tool that maintains
  them.

## Decision

Every dependency floor under ADR 0001 decisions 2 and 3, for core and other
runtime dependencies alike, is the oldest minor version whose x.y.0 release is
within the 2-year window, written `>=x.y`. Floors are never set to a patch
release. Python versions are already counted per minor version (ADR 0001
decision 1).

## Consequences

- #77 sets `igraph>=1.0`, not `>=0.11.8`. SymPy is `>=1.14` under either
  reading.
- #78's spec0-action configuration must use this rule. Its default already does.
- ADR 0001 is otherwise unchanged.
