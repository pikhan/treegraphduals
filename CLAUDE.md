# treegraphduals

Extension of my Master's thesis, *The Horizontal Tunnelability Graph is Dual to Level Set Trees*
(University of Nevada, Reno, Sept 2023): https://scholarworks.unr.edu/handle/11714/10548

A package for working with trees, their duals, graphs, and time series.

## Stack

- Python 3.12+, packaged with `uv`, built with Hatchling (`src/` layout)
- Lint/format: Ruff (`ruff check`, `ruff format`)
- Type checking: Pyrefly
- Tests: pytest, coverage via `pytest-cov`; Hypothesis is available in the dev group but no property-based tests are written yet
- Docs: Sphinx (numpydoc-style docstrings double as both API docs and doctests), hosted on Read the Docs
- Pre-commit hooks enforce lint/type-check/citation-sync on every commit; full test suite runs in CI on push/PR

## Commands

- `uv sync` — install/update the environment from `uv.lock`
- `uv run pytest --cov=treegraphduals --cov-report=html --cov-report=term --cov-report=json --doctest-modules -v` — full test + doctest run with coverage
- `uv run coverage report --format=markdown > docs/coverage_report.md` — regenerate the coverage summary embedded in the docs
- `uv run pytest --tb=no --no-header -q > docs/test_results.txt || true` — regenerate the test-results summary embedded in the docs
- `uv run ruff check --fix .` / `uv run ruff format .` — lint/format
- `uv run pyrefly check` — type check
- `uv run pre-commit run --all-files` — run all pre-commit hooks manually
- `uv build` — build the sdist + wheel into `dist/`

## Conventions

- Docstrings: numpy style (Parameters/Returns/Examples sections); doctest examples in docstrings are run as tests.
- Type hints required on public functions/classes; checked by Pyrefly, not Pyright/Pylance.
- Update `CHANGELOG.md` under `## [Unreleased]` for any notable change; rename that section to the new version + date at release time (alongside the `pyproject.toml` version bump).

## Layout

Importable as `treegraphduals`; everything ships under `src/treegraphduals/`.

- `src/treegraphduals/core/` — tree/graph data structures
- `src/treegraphduals/timeseries/` — time series analysis
- `src/treegraphduals/visualizations/` — plotting
- `src/treegraphduals/agents/` — small LangGraph-based tooling (verification assistant, experiment-exploration loop); intermittent, secondary to the core research

Within the package, cross-subpackage imports are relative (`from ..core.tree import Tree`);
docstring examples use the absolute path a user would type (`from treegraphduals.core import Tree`).