# 0001. Follow SPEC 0 for supported Python and dependency versions

- **Status:** Accepted
- **Date:** 2026-10-10
- **Issues:** #76, #77, #78, #75, #79

## Context

treegraphduals is a scientific Python library. Its core dependencies (NumPy,
SciPy, Matplotlib, NetworkX) follow [SPEC 0](https://scientific-python.org/specs/spec-0000/)
and drop old versions on a published quarterly schedule. SPEC 0 recommends
dropping a Python version **3 years** after its release and a core package
version **2 years** after its release. The stated purpose is to limit
maintenance burden. Projects may drop earlier when a critical fix isn't
backported. SPEC 0 doesn't forbid supporting versions longer.

Before this ADR, the floors (`requires-python >=3.12`, `numpy>=1.26`,
`scipy>=1.11`, `matplotlib>=3.8`, `networkx>=3.2`) were all older than SPEC 0
allows. CI tested none of them: it tested only the latest locked versions, on
one interpreter. The floors were claims, not tested facts.

Installers honor `Requires-Python`. When a release drops a Python version,
users on that version keep getting the last release that supported it.
Dropping a version never breaks existing users.

JOSS and pyOpenSci reviewers (#64) look for a stated support policy.

## Options considered

- **A. Follow SPEC 0.** Drop each Python version 3 years after release and each
  core-package version 2 years after release, quarterly. This is the same
  schedule our dependencies use, and it keeps CI small.
- **B. SPEC 0, but keep each Python version once it works ("perpetual
  support").** The claim decays. Once NumPy and SciPy stop publishing wheels
  for a version, CI on it tests ever-older dependencies that nobody else tests.
  CPython itself stops security fixes 5 years after release (3.13: October 2029).
  "It worked when we tested it" says nothing about code written later. Users
  on old versions are already served by old releases (see Context).
- **C. A fixed floor, bumped when someone remembers.** What the project did
  until now. The floors drifted (see Context).
- **D. NEP 29.** Superseded by SPEC 0.

## Decision

1. **Python.** Support each CPython minor version until its SPEC 0 drop date
   (3 years after its x.y.0 release). The `requires-python` floor is the oldest
   version still in that window. Drop earlier only if a dependency we need
   forces it. Python 3.12 is dropped at the next release (#77), at its SPEC 0
   date (2026 Q4), or a few days before it if that release lands before
   2026-10-01.
2. **Core dependencies** (the SPEC 0 core packages we depend on: NumPy, SciPy,
   Matplotlib, NetworkX). Floors follow SPEC 0's 2-year window. As in SPEC 0,
   the window is counted per minor version from its x.y.0 release: a floor is
   the oldest minor version (`>=x.y`) whose x.y.0 came out within the last 2
   years, never a patch release.
3. **Other runtime dependencies** (igraph, SymPy, and any added later). Floors
   follow the same 2-year, per-minor-version window (spec0-action's
   `update_all: 2`). They are
   raised further when CI's lowest-direct job (#78) shows they must be.
4. **Cadence.** Floors are bumped quarterly by an automated PR (#78) that a
   maintainer reviews and merges. The PR's GitHub notification (email) is the
   maintainer's prompt to act; no separate calendar or email reminder is
   required. A release after a quarter boundary includes that quarter's drops.
5. **New Python versions.** As soon as uv can install a prerelease, add it and
   its free-threaded build to CI as non-blocking jobs. Make the normal build
   blocking, and add its classifier, once the final release is out, every
   runtime dependency installs from wheels, and the job has been green on
   `main` (#79 for 3.15).
6. **Free-threaded builds** are tested in non-blocking jobs only. We don't
   claim free-threading support (no classifier, no thread-safety guarantee)
   until a separate ADR decides to.
7. **Development** happens on the newest blocking version. CI's lint and
   type-check job runs on the floor.
8. **No upper bounds** on dependencies. `uv.lock` pins exact versions for
   development.

## Consequences

- The next release requires Python 3.13+ and raises the dependency floors
  (#77). Python 3.12 users keep getting 0.1.x.
- Expected Python drops: 3.13 in 2027 Q4, 3.14 in 2028 Q4.
- Every quarter an automated PR arrives. CI must pass on the new floors,
  including the lowest-direct job.
- The supported-version list lives in several files (pyproject, CI matrix,
  `.readthedocs.yaml`, the release smoke test, README, CLAUDE.md). #78
  keeps them in sync.
- Rules out perpetual support (option B) and unscheduled floor bumps (option C).
