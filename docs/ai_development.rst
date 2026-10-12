How This Project Is Built with AI
=================================

.. admonition:: Maintainer: write this (introduction)
   :class: warning

   *Placeholder.* Introduce the page in your own voice: why treegraphduals is
   built with AI agents, who this page is for (users, contributors, reviewers
   at pyOpenSci and JOSS), and what you want a reader to take away.

The facts below are kept current with the tooling in the repository. The
working rules live in ``CLAUDE.md`` and ``.claude/``. How AI use is measured is
recorded in :doc:`ADR 0006 <adr/0006-ai-provenance-tracking>`, and the full
numbers are on the :doc:`ai_provenance` page.

The pipeline
------------

.. code-block:: text

   GitHub issue
     labels: priority, difficulty, file, foundational
       │
       │  maintainer runs /work-issue <N>
       ▼
   Worker agent (Claude Code), in its own git worktree
     branch issue-<N>-<slug>
     asks the maintainer about anything the issue leaves open
       │
       │  tests, lint and type checks pass; pre-commit hooks run
       ▼
   Draft pull request
     Closes #N · Changes · Test concerns · Decisions made · AI use
       │
       ▼
   reviewer subagent  ◄──────────────┐
     sees only the issue and the diff │ CHANGES REQUESTED
     verdict posted on the PR         │ (at most two rounds)
       ├──────────────────────────────┘
       │  APPROVE, or round 2 finished
       ▼
   AI provenance comment on the PR
       │
       ▼
   Maintainer review ──► squash merge into main (maintainer only)

- **Issues.** Every issue carries ``priority`` (when), ``difficulty`` (who) and
  ``file:`` labels. Issues labeled ``foundational`` go first. The plan is the
  pinned roadmap issue (#50).
- **Worker.** ``/work-issue <N>`` (``.claude/skills/work-issue/``) runs one
  session per issue. It refuses closed, blocked, already-claimed and
  ``difficulty: max`` issues. It asks the maintainer about anything left open
  before writing code, and it opens a draft PR with a fixed set of sections,
  including a record of AI use.
- **Reviewer.** ``.claude/agents/reviewer.md`` is a separate agent that has no
  edit tools and is not given the worker's reasoning. It checks test integrity
  first, runs the suite itself, tries to break mathematical changes with inputs
  the tests don't cover, and reports Test integrity, Evidence, Spec, Math and
  Standards separately.
- **Hand-over.** The PR is marked ready only after an APPROVE verdict. After two
  rounds, any disagreement goes to the maintainer.
- **Merge.** Only the maintainer merges, after the ``lint-and-typecheck`` and
  ``test`` CI checks pass.

.. admonition:: Maintainer: write this (why this pipeline)
   :class: warning

   *Placeholder.* The design reasoning: why one agent per issue, why an
   adversarial reviewer instead of self-review, why two rounds at most, and
   what the alternatives were.

Guardrails
----------

Each guardrail below is listed with what it does. The reasoning behind each is
yours to write in the placeholder at the end of the section.

- **Branch protection on main, admin bypass off.** ``main`` takes changes only
  through pull requests with green ``lint-and-typecheck`` and ``test`` checks,
  and the rule applies to administrators too.
- **The PreToolUse guard hook.** ``.claude/hooks/guard_github.py`` runs before
  every shell command an agent issues in this repository. It denies pushing to
  ``main``, force-pushing, merging pull requests, and changing repository
  settings, branch protection or rulesets. The maintainer's own terminal is not
  affected.
- **Tests for the guard hook.** ``tests/test_guard_github.py`` runs the hook on
  a table of 55 commands (compound commands, ``bash -c``, ``eval``, heredocs,
  ``git -C``, a bare push from a checkout of ``main``, and more) plus two
  extra checks, in a throwaway repository. A new bypass or false positive gets
  a row in the table before it is fixed.
- **Worktree isolation.** Each issue gets its own git worktree, branch and
  pull request. Two agents never work at once on issues that share a ``file:``
  label.
- **The test-integrity rule.** Existing tests are the specification. Agents
  never edit, delete, skip or weaken them, including expected values,
  ``xfail``/``skip`` markers, ``@example``\ s and Hypothesis settings. The only
  exception is removing a ``strict=True`` ``xfail`` that the fix makes pass. A
  test an agent thinks is wrong goes under "Test concerns" in the PR, and the
  reviewer treats any other change to existing tests as blocking.
- **Independent oracles.** An expected value in a test must come from a source
  independent of the code under test: a hand-worked example, the definition,
  NetworkX, or a brute-force construction. The reviewer treats a tautological
  test of mathematical code as blocking.
- **difficulty: max means human-owned.** Agents never implement these issues on
  their own. The label also marks issues blocked on a decision only the
  maintainer can make.
- **Pre-commit hooks.** Lint, type checks and the citation sync run on every
  commit, and agents never skip them with ``--no-verify``.

.. admonition:: Maintainer: write this (why each guardrail exists)
   :class: warning

   *Placeholder.* For each guardrail above, the failure it prevents and why it
   is enforced mechanically (a hook, branch protection) or by rule.

What stays human
----------------

- **Definitions.** The mathematical definitions follow the thesis and
  :doc:`definitions`. Changing one needs an issue and an ADR.
- **Expected values.** Hand-computed acceptance tests are ``difficulty: max``
  issues (for example #53).
- **Oracles.** The independent reference constructions that the property tests
  compare against.
- **Decisions.** Public API, mathematical conventions, tie handling and new
  dependencies are settled by the maintainer and recorded as ADRs in
  ``docs/adr/``.
- **Merging.** Only the maintainer merges, and the guard hook blocks agents
  from doing it.
- **Review replies.** Communication with pyOpenSci and JOSS editors and
  reviewers is written by the maintainer.

.. admonition:: Maintainer: write this (why these stay human)
   :class: warning

   *Placeholder.* Why these are the parts you keep, and where you draw the line
   between work you delegate and work you own.

Lessons learned
---------------

- **A tautological test enshrined a bug.** The level-set tree builder chained
  minima left to right instead of nesting them by height (#4). An existing test
  passed anyway: its input happened to give the same tree either way, and its
  expected value had been copied from the buggy output (#53). This
  is why expected values for the core math are now hand-computed from the
  definitions (#53) and why the reviewer blocks tautological tests.
- **The guard hook blocked its own pull request.** While #56 was being set up,
  opening its PR was denied: the PR description, passed in a heredoc,
  mentioned ``gh pr merge`` inside Markdown backticks, and the first version
  of the hook parsed it as a command. The parser now understands quotes and
  heredocs, regression cases cover it, and the test table became a pytest suite
  (#71).

.. admonition:: Maintainer: write this (lessons learned)
   :class: warning

   *Placeholder.* What each lesson taught you, how it changed the workflow,
   and any further lessons.

Metrics
-------

The numbers below are generated from the published AI provenance statistics
(#81, :doc:`ADR 0006 <adr/0006-ai-provenance-tracking>`) on every docs build. A
nightly workflow refreshes them. The per-PR table and the method are on the
:doc:`ai_provenance` page.

.. include:: ai_stats.md
   :parser: myst_parser.sphinx_
   :end-before: <!-- end-of-headline -->

How they map onto the questions this page asks:

- **Reviewer rounds per PR:** "Reviewer rounds per reviewed PR".
- **How often the reviewer found real blocking problems:** "Approved in the
  first round". A PR that wasn't approved in round 1 had blocking problems.
- **How often the maintainer still changed things after an APPROVE:**
  "Messages after the reviewer agent approved" and "Agent PRs where the
  maintainer pushed commits".
- **Time from dispatch to merge:** not published yet. The statistics record
  each PR's time open, but no headline figure is rendered.

.. admonition:: Maintainer: write this (reading the metrics)
   :class: warning

   *Placeholder.* How you read these numbers, their limits (see "The coverage
   rule" on the :doc:`ai_provenance` page), and what they say about the
   workflow.
