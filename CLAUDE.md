# treegraphduals

Extension of my Master's thesis, *The Horizontal Tunnelability Graph is Dual to Level Set Trees*
(University of Nevada, Reno, Sept 2023): https://scholarworks.unr.edu/handle/11714/10548

A package for working with trees, their duals, graphs, and time series.

## Stack

- Python 3.12+, packaged with `uv`, built with Hatchling (`src/` layout)
- Lint/format: Ruff (`ruff check`, `ruff format`)
- Type checking: Pyrefly
- Tests: pytest + Hypothesis for property-based testing (extensive use planned; none written yet), coverage via `pytest-cov`
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
- `CHANGELOG.md` is edited only by the maintainer, at merge time. Never edit it in a branch or PR; propose an entry in the PR description instead (see below). At release time the maintainer renames `## [Unreleased]` to the new version + date, alongside the `pyproject.toml` version bump.

## Layout

Importable as `treegraphduals`; everything ships under `src/treegraphduals/`.

- `src/treegraphduals/core/` — tree/graph data structures
- `src/treegraphduals/timeseries/` — time series analysis
- `src/treegraphduals/visualizations/` — plotting
- `src/treegraphduals/agents/` — small LangGraph-based tooling (verification assistant, experiment-exploration loop); intermittent, secondary to the core research

Within the package, cross-subpackage imports are relative (`from ..core.tree import Tree`);
docstring examples use the absolute path a user would type (`from treegraphduals.core import Tree`).

## Issues and labels

The plan is the pinned roadmap issue (#50): `gh issue view 50`. Every issue carries:

- `priority: highest | high | medium | low`: **when** to do it.
- `difficulty: max | high | medium | low`: **who** does it.
  - `max`: human work, or blocked on a human decision. Never implement it autonomously. If an issue turns out to need a decision, stop and ask.
  - `high`: strongest model at high (or xhigh) effort.
  - `medium`: mechanical and fully specified.
  - `low`: boilerplate or trivial, non-mathematical fixes.
- `file: <path>`: the file(s) mainly touched. Don't run two agents at once on issues that share a `file:` label.
- `foundational`: do these first. Their body has an "Affects" section listing dependent issues; re-read those issues after a foundational one lands.

## Agent workflow (one agent per issue)

Worker agent:
- One issue per git worktree, branch and PR. Name the branch `issue-<N>-<short-slug>`. The PR body starts with `Closes #<N>`.
- In a fresh worktree run `uv sync` first. Before pushing, run `uv run pre-commit run --all-files` and the full test command above.
- **Tests are the spec.** Never modify, delete, skip or weaken an existing test. That includes expected values, `xfail`/`skip` markers, `@example`s and Hypothesis settings. You may add new tests. One exception: remove a `strict=True` `xfail` marker once your fix makes that test pass, and say so in the PR. If you think an existing test is wrong, leave it alone and explain under a "Test concerns" heading in the PR.
- Stay in scope. Don't fix other issues along the way; list anything you notice under "Out of scope" in the PR.
- Don't settle anything the issue leaves open (public API, mathematical conventions, tie handling, new dependencies). Comment on the issue and stop.
- Never edit `CHANGELOG.md`. Put the proposed entry under a "Changelog" heading in the PR description.
- Never push to `main` and never merge PRs. `main` is protected: PRs need green `lint-and-typecheck` and `test` checks.

Reviewer agent:
- Review from the issue and the diff only, not the worker's reasoning or transcript.
- First check whether the diff touches `tests/`, test markers or Hypothesis settings. If it does, flag that at the top of the review.
- Run the full test suite yourself. For mathematical code, also try to break the change with inputs the tests don't cover, and check it against the thesis definition the issue cites.
- Report a verdict (approve / changes requested), then blocking problems with evidence (commands run and their output), then non-blocking notes.
- At most two worker ↔ reviewer rounds, then hand the PR to the maintainer, including any disagreement.