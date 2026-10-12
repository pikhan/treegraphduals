# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## \[0.2.0\] - 2026-10-11

### Added

- Community health files: a Code of Conduct (Contributor Covenant v2.1), issue forms for bug reports, feature requests and math/definition questions, a pull request template, `SECURITY.md` and `SUPPORT.md`. Questions go to GitHub Discussions (#70).
- Contributing guide: the contribution process, requirements for code, and the test policy (#70, #83).
- README and docs badges, and weekly Dependabot updates for GitHub Actions and `uv.lock` (#60).
- A quarterly SPEC 0 workflow that opens a PR raising the Python and dependency floors, and `scripts/sync_python_versions.py`, which keeps every copy of the supported-Python list in sync with `pyproject.toml` (#78).
- AI-use disclosure: every agent-authored PR records the models, session, AI-generated files, human input and review rounds, and `CLAUDE.md` sets the provenance rules (#82).
- AI provenance tracking: each PR's human steering and AI/human line authorship is recorded, and the totals, plus a nightly mutation-testing score, are published on the "AI Provenance" docs page (#81, ADR 0006).
- Docs page: "How This Project Is Built with AI" (#72).
- Agent workflow tooling: the `/work-issue` skill, a `reviewer` subagent based on Matt Pocock's skills, and a hook that keeps pushes to `main`, PR merges and repository-settings changes with the maintainer, with a test suite for the hook (#71).
- Architecture decision records in `docs/adr/` (ADRs 0001–0006), and `.github/CODEOWNERS`.

### Changed

- **Breaking:** requires Python 3.13 or newer, and raises the minimum versions of NumPy (2.3), SciPy (1.15), Matplotlib (3.11), NetworkX (3.5), igraph (1.0) and SymPy (1.14), following the SPEC 0 support policy (ADRs 0001 and 0002). Python 3.12 users should stay on 0.1.0.
- CI tests Python 3.13 and 3.14 (required) and 3.15 (non-blocking), plus a non-blocking job on the minimum dependency versions. Free-threaded builds are not tested yet, because igraph has no free-threaded wheels (ADR 0003).
- Read the Docs builds run the test suite once. Failing tests no longer abort the docs build; the Testing page shows the failures instead (ADR 0004).

### Removed

- The empty `treegraphduals.agents` subpackage (planned LangGraph tooling, now out of scope) (#74).

### Security

- Every GitHub Action is pinned to a full commit SHA. Workflows that don't push no longer keep checkout credentials, and the release workflow defaults to a read-only token (#83).
- An OpenSSF Scorecard workflow and README badge, and zizmor checks of the workflow files in pre-commit (#83).
- GitHub releases carry Sigstore-signed build provenance; the docs explain how to verify it (#83).
- Dependabot version updates wait out a 7-day cooldown; security updates are not delayed (#83).
- Vulnerabilities are reported privately (see `SECURITY.md`), and fixes ship in the latest release only (ADR 0005).

## \[0.1.0\] - 2026-09-18

### Added

- Initial project setup: `uv` packaging, Ruff lint/format, Pyrefly type checking, pytest + Hypothesis + coverage, Sphinx docs on Read the Docs, `CITATION.cff` with Zenodo DOI, pre-commit hooks.
- Packaging: Hatchling build backend, so the project can be built and published (`uv build`). Ships `py.typed`, the MIT license file, README as the long description, classifiers, keywords and project URLs.
- `tests/test_timeseries.py`: regression tests for the extrema cases below, the numerical fallback path, uniform sampling, and level-set tree construction.

### Changed

- **Breaking:** moved `core`, `timeseries`, `visualizations` and `agents` under a single `src/treegraphduals/` package. Imports are now `from treegraphduals.core import Tree` rather than `from core import Tree`, so the library no longer claims generic top-level names.
- `requires-python` raised to `>=3.12`, matching what CI and Read the Docs test.
- Test and documentation tooling moved out of runtime dependencies into the `dev` and `docs` dependency groups; a plain `pip install treegraphduals` no longer pulls in pytest and Sphinx.
- Doctests now run by default: `testpaths` covers `src` as well as `tests`, so the `--doctest-modules` flag in CI is no longer a no-op (62 tests, up from 40).
- `docs/conf.py` reads the version from package metadata instead of hardcoding it, and no longer injects an absolute local path for doctest setup.
- Replaced a `print` with `warnings.warn` when symbolic differentiation falls back to the numerical extrema search.
- Cleared the Ruff backlog (56 findings): added missing module and magic-method docstrings, made ASCII-art docstrings raw so the diagrams render literally, replaced implicit `Optional` annotations, narrowed a bare `except`, collapsed nested conditionals, sorted `__all__` lists, and dropped dead locals.
- README reorganized into what the package computes today versus what is planned. Every original bullet is preserved; the previous single list mixed 25+ unimplemented features (duals, visibility graphs, merge trees, graph metrics) with the 7 that work, and it is the PyPI long description.

### Fixed

- Docstring tree diagrams rendered mangled: 15 docstrings were not raw strings, so a trailing `\` in the ASCII art silently swallowed the following newline and joined rows together (`/ \` + `1   2` became `/             1   2`). The art in the source was correct all along. Diagrams are now raw strings inside RST literal blocks, and the Horton pruning and Horton-Strahler examples are redrawn from what the algorithms actually produce.
- The Sphinx build is now warning-free (was 84 warnings).
- Symbolic extrema search (`TimeSeries.from_function(..., preserve_extrema=True)`) no longer silently discards critical points:
  - Real roots that sympy returns in complex form (Cardano's formula writes the three real roots of a cubic using `I`) were dropped, because the reality test was "does `float()` succeed". Reality is now decided by the magnitude of the imaginary part, so e.g. `t**4/4 - 2*t**2 + t/3` reports its three extrema instead of none.
  - Periodic derivatives only ever produced sympy's principal solutions, so `sin(t)` on `[0, 10]` lost its maximum at `5*pi/2`. Critical points are now solved over the interval, which enumerates the periodic solutions.
- `plot_tree_and_harris_path` no longer leaks one matplotlib figure per call (pyplot kept every discarded `plot_tree` figure alive), and now draws the tree into its left axes instead of a "Tree plot" placeholder.