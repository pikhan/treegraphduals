# 0005. Security fixes go into the latest release only

- **Status:** Proposed
- **Date:** 2026-10-11
- **Issues:** #70, #76, #83

## Context

`SECURITY.md` (#70) has to say which versions get security fixes. GitHub shows
the policy under the repository's Security tab, the OpenSSF Scorecard
Security-Policy check looks for it (#83), and JOSS and pyOpenSci reviewers
expect one (#64).

Vulnerabilities are reported privately through GitHub's private vulnerability
reporting (#83). A fix is coordinated in a GitHub security advisory and
published with a release.

The project has one maintainer, and releases are cut from `main`. There are no
maintenance branches.

[ADR 0001](0001-spec-0-support-policy.md) has the project follow SPEC 0. Each
release drops the Python and dependency versions that have left the SPEC 0
window. Installers honor `Requires-Python`, so users on a dropped Python keep
getting the last release that supported it. ADR 0001 covers which versions a
*release* supports. It doesn't say which *releases* get fixes.

## Options considered

- **A. Latest release only.** Fixes ship in a new release from `main`, and
  nothing is backported. This is the least work, and it's consistent with
  ADR 0001: old releases are for old environments, not maintained lines. Users on a Python
  that SPEC 0 has dropped don't get the fix.
- **B. The latest release, plus backports to the previous minor series.** This
  needs maintenance branches, a release process for each one, and CI for
  dependency floors that `main` no longer tests. One maintainer can't keep
  that up.
- **C. The last release for every Python version still supported by CPython.**
  This is the most generous option. It brings back the old-floor maintenance
  that ADR 0001 ruled out, and the dependencies themselves stop releasing for
  those versions.

## Decision

1. Security fixes are made **only in a new release** cut from `main`. Earlier
   releases get no fixes, and nothing is backported.
2. That release follows ADR 0001, so it supports only the Python and
   dependency versions in the SPEC 0 window when it is made.
3. The advisory published with the fix names the affected versions and the
   fixed version, so users on older releases can see that they are affected.
4. `SECURITY.md` states this policy, and it cites this ADR.

## Consequences

- To get a fix, users upgrade to the latest release. A user on a Python that
  SPEC 0 has dropped has to upgrade Python first.
- No maintenance branches or backport process are needed.
- Changing this policy, for example to backport to the previous minor series
  once the project has more maintainers, takes a new ADR that supersedes this
  one.
