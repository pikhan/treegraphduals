# 0004. Read the Docs runs the test suite once and publishes failures

- **Status:** Proposed
- **Date:** 2026-10-11
- **Issues:** #31

## Context

The Testing page (`docs/testing.rst`) includes `docs/test_results.txt` and
`docs/coverage_report.md`, and links to the HTML coverage report. These files
are gitignored. Read the Docs (RTD) generates them in its `pre_build` step.

That step ran the suite twice. The first run aborted the build on any failure,
so the `|| true` on the second run, which wrote `test_results.txt`, never took
effect. The page could only ever show green, and each docs build took twice
as long as it needed to. `pytest.ini`'s `addopts` already add coverage, the HTML
report and doctests to every run, so one run produces everything the page needs.

CI already runs the full suite, and its `test` check is required to merge into
`main`.

## Options considered

- **A. Move report generation into CI, and have RTD only build Sphinx.** RTD
  can't download GitHub Actions artifacts without a token, and those artifacts
  expire. RTD builds start on push, often before CI finishes, so the reports
  would lag a commit behind. Too much machinery for the real problem, which is
  the doubled run time.
- **B. Gate: run once, and a failing test aborts the docs build.** A build that
  aborts leaves RTD serving the last successful build. If broken code reached
  `main`, the docs would keep showing stale green results.
- **C. Report: run once, and a failing test doesn't abort the build.** The docs
  always match `main`. Normally `main` is green, because CI blocks broken
  merges, so the page is green. If broken code gets in anyway, the page shows it.

## Decision

Option C, with tests that fail to load still aborting the build (maintainer,
on #31 and in the PR #98 session):

1. RTD keeps generating the test and coverage reports. They don't move to CI.
2. `pre_build` runs pytest once. Its output goes to `docs/test_results.txt`,
   and the coverage reports (`coverage_report.md`, `coverage.json`, `htmlcov/`)
   come from that same run's data.
3. Failing tests (pytest exit code 1) don't abort the docs build. The failures
   appear on the Testing page.
4. If the suite doesn't run properly (exit code 2 or higher: tests that fail to
   load, no tests collected, or a pytest error), the docs build aborts. The
   same failure already fails CI and blocks the merge. In that case RTD keeps
   serving the last good build, rather than publishing a page with no results
   and no coverage.
5. CI's required `test` check, not the docs build, is the merge gate.

## Consequences

- `.readthedocs.yaml` writes the single pytest run to `test_results.txt`, echoes
  it to the build log, and aborts only when pytest's exit code is 2 or higher
  (#31).
- A red Testing page on `main` means broken code got past CI: treat it as a bug.
- A green docs build no longer means the tests passed. Check CI for that.
