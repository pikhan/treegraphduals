# 0003. Don't test free-threaded Python builds in CI for now

- **Status:** Proposed
- **Date:** 2026-10-11
- **Issues:** #75, #79
- **Supersedes:** decision 6 of [ADR 0001](0001-spec-0-support-policy.md)

## Context

[ADR 0001](0001-spec-0-support-policy.md) decision 6 put free-threaded builds
(3.14t, 3.15t) in non-blocking CI jobs as an early warning. #75 added those
jobs. On their first runs (PR #88), neither job got as far as running a test:

- igraph 1.0.0, a runtime dependency, publishes only limited-API (`abi3`)
  wheels. Free-threaded CPython rejects the limited API
  (`#error "The limited API is not currently supported in the free-threaded
  build"`), so there is no wheel to install and the source build fails.
- NumPy, SciPy, Matplotlib and Pillow already ship free-threaded wheels. igraph
  is the only blocker.

We can't fix this. Until igraph ships free-threaded wheels, these jobs fail at
`uv sync` on every PR. They test none of our code, and a permanent red ✗ teaches
everyone to ignore red marks.

We don't claim free-threading support, and the ensembles in #49 parallelize just
as well with a process pool on any Python. Nothing we ship depends on free
threading.

## Options considered

- **A. Keep the jobs, non-blocking (ADR 0001 decision 6).** They give no signal
  until igraph changes, and every PR shows red marks until then. Each new
  free-threaded version would then repeat the wait for dependency wheels.
- **B. Remove the jobs, and revisit once igraph ships free-threaded wheels.**
  No noise, and no loss of signal, because the jobs test nothing today.
- **C. Build igraph from source in CI without the limited API.** That would
  mean maintaining a build of a C extension we don't own, for a feature we
  don't claim.

## Decision

1. CI runs no free-threaded (`t`) Python builds. This supersedes ADR 0001
   decision 6.
2. We still claim no free-threading support: no classifier, and no
   thread-safety guarantee.
3. Revisit once igraph publishes free-threaded wheels (`cp3XXt`) on PyPI. Then a
   new ADR can restore non-blocking free-threaded jobs.

## Consequences

- `ci.yml` drops the 3.14t and 3.15t jobs (#75).
- #79 covers only the normal 3.15 build. 3.15t is no longer in CI.
- ADR 0001 decision 5's "and its free-threaded build" no longer applies to new
  Python versions.
