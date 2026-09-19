# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]
### Fixed
- Docstring tree diagrams rendered mangled: 15 docstrings were not raw strings,
  so a trailing `\` in the ASCII art silently swallowed the following newline
  and joined rows together (`/ \` + `1   2` became `/             1   2`). The
  art in the source was correct all along. Diagrams are now raw strings inside
  RST literal blocks, and the Horton pruning and Horton-Strahler examples are
  redrawn from what the algorithms actually produce.
- The Sphinx build is now warning-free (was 84 warnings).
- Symbolic extrema search (`TimeSeries.from_function(..., preserve_extrema=True)`)
  no longer silently discards critical points:
  - Real roots that sympy returns in complex form (Cardano's formula writes the
    three real roots of a cubic using `I`) were dropped, because the reality test
    was "does `float()` succeed". Reality is now decided by the magnitude of the
    imaginary part, so e.g. `t**4/4 - 2*t**2 + t/3` reports its three extrema
    instead of none.
  - Periodic derivatives only ever produced sympy's principal solutions, so
    `sin(t)` on `[0, 10]` lost its maximum at `5*pi/2`. Critical points are now
    solved over the interval, which enumerates the periodic solutions.
- `plot_tree_and_harris_path` no longer leaks one matplotlib figure per call
  (pyplot kept every discarded `plot_tree` figure alive), and now draws the tree
  into its left axes instead of a "Tree plot" placeholder.

### Changed
- README reorganized into what the package computes today versus what is
  planned. Every original bullet is preserved; the previous single list mixed
  25+ unimplemented features (duals, visibility graphs, merge trees, graph
  metrics) with the 7 that work, and it is the PyPI long description.

### Added
- `tests/test_timeseries.py`: regression tests for the extrema cases above, the
  numerical fallback path, uniform sampling, and level-set tree construction.
- Packaging: Hatchling build backend, so the project can be built and published
  (`uv build`). Ships `py.typed`, the MIT license file, README as the long
  description, classifiers, keywords and project URLs.

### Changed
- **Breaking:** moved `core`, `timeseries`, `visualizations` and `agents` under a
  single `src/treegraphduals/` package. Imports are now
  `from treegraphduals.core import Tree` rather than `from core import Tree`,
  so the library no longer claims generic top-level names.
- `requires-python` raised to `>=3.12`, matching what CI and Read the Docs test.
- Test and documentation tooling moved out of runtime dependencies into the `dev`
  and `docs` dependency groups; a plain `pip install treegraphduals` no longer
  pulls in pytest and Sphinx.
- Doctests now run by default: `testpaths` covers `src` as well as `tests`, so the
  `--doctest-modules` flag in CI is no longer a no-op (62 tests, up from 40).
- `docs/conf.py` reads the version from package metadata instead of hardcoding it,
  and no longer injects an absolute local path for doctest setup.
- Replaced a `print` with `warnings.warn` when symbolic differentiation falls back
  to the numerical extrema search.
- Cleared the Ruff backlog (56 findings): added missing module and magic-method
  docstrings, made ASCII-art docstrings raw so the diagrams render literally,
  replaced implicit `Optional` annotations, narrowed a bare `except`, collapsed
  nested conditionals, sorted `__all__` lists, and dropped dead locals.

## [0.1.0] - 2026-09-18
### Added
- Initial project setup: `uv` packaging, Ruff lint/format, Pyrefly type checking, pytest + Hypothesis + coverage, Sphinx docs on Read the Docs, `CITATION.cff` with Zenodo DOI, pre-commit hooks.