---
name: work-issue
description: Implement one GitHub issue end to end in an isolated worktree, run the reviewer loop, and hand a ready PR to the maintainer. Run as /work-issue <issue-number>, ideally in its own background session from agent view.
argument-hint: "<issue-number>"
arguments: [issue]
disable-model-invocation: true
---

Work on issue **#$issue** of this repository, following the "Issues and labels"
and "Agent workflow" rules in `CLAUDE.md`. One issue per session.

## 1. Check the issue is ready

Run `gh issue view $issue --json number,title,state,labels,body,comments` and
`gh issue view $issue --json blockedBy,closedByPullRequestsReferences`.

**Stop without changing anything**, and tell the maintainer why, if any of these hold:

- the issue is closed;
- it is labelled `difficulty: max` (human work, or a decision still open);
- it is blocked by an issue that is still open;
- an open PR already targets it (`closedByPullRequestsReferences` is non-empty).

Also list any open PRs that touch the same files as this issue's `file:` labels.
Warn the maintainer about them, but carry on.

## 2. Understand before touching code

- Read the whole issue, including its comments, any "Affects" section, and any
  decision recorded in a comment or an ADR under `docs/adr/`.
- The line links in the issue point at an older commit. Find the current code.
- Read the tests that already cover this area, especially strict-`xfail` tests
  that name this issue: they are the spec.

**If anything is underspecified, ask before writing code.** Use AskUserQuestion,
and give your recommended answer with each question. This covers mathematical
conventions, public API, tie handling, new dependencies, or anything where two
readings of the issue lead to different code. Never guess on these. Only you can
ask: the reviewer subagent cannot.

## 3. Set up the worktree

- Confirm you are in a git worktree under `.claude/worktrees/`, not the main
  checkout (`git rev-parse --show-toplevel`). If you are in the main checkout,
  enter a worktree first with the EnterWorktree tool.
- Name the branch after the issue: `git branch -m issue-$issue-<short-slug>`.
- Run `uv sync`.

## 4. Implement

- Make the smallest change that satisfies the issue's requirements and acceptance
  checklist. Stay in scope, and note anything else you spot for the PR's
  "Out of scope" section.
- Follow the test rules in `CLAUDE.md`. Existing tests are the spec: never edit,
  delete, skip or weaken them. Remove a `strict=True` `xfail` marker only once your
  fix makes that test pass.
- You may add tests. For mathematical behaviour, an expected value must come from
  an independent source (a worked example, the definition, NetworkX), **never from
  running the code you just wrote**.
- Match the surrounding code: numpy docstrings with runnable examples, type hints,
  relative imports inside the package.

## 5. Verify

Run everything, and fix failures before going on:

- `uv run pytest --cov=treegraphduals --cov-report=term --doctest-modules -q`
- `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyrefly check`

## 6. Commit, push, open a draft PR

- Commit with a descriptive message that references #$issue. The pre-commit
  hooks run on every commit: fix what they report, and never skip them with
  `--no-verify`. Don't edit `CHANGELOG.md`.
- `git push -u origin HEAD`
- `gh pr create --draft`, with a title that names the change and a body with these
  sections:
  - `Closes #$issue`
  - **Summary**: what changed and why, in a few sentences.
  - **Changes**: per file.
  - **How verified**: the commands from step 5 and their results.
  - **Test concerns**: any existing test you believe is wrong, and why (or "None").
  - **Out of scope**: things noticed but not done (or "None").
  - **Changelog**: the proposed `CHANGELOG.md` entry. The maintainer adds it at merge.
  - **Decisions made**: every judgement call you made, including answers the
    maintainer gave you in step 2.

## 7. Review loop (at most two rounds)

1. Invoke the `reviewer` subagent. Pass it **only** the issue number, the PR number
   and the worktree path. Don't pass your reasoning or summary, so its review
   stays independent.
2. Post its report on the PR with `gh pr comment`, headed
   `Automated review (round 1)`.
3. If the verdict is CHANGES REQUESTED, address each blocking item. If you
   disagree with one, don't change the code for it; explain why in a PR comment.
   Then repeat step 5, push, and run one more review (round 2), posted the same way.
4. Stop after round 2 whatever the verdict. Leave any unresolved disagreement for
   the maintainer.

## 8. Hand over

- `gh pr ready <pr>` (only if the final verdict is APPROVE; otherwise leave it as a
  draft).
- Tell the maintainer: the PR link, the final verdict, what you decided and why,
  and anything they should look at first.
- **Never merge.** Merging is the maintainer's review step, and a hook blocks
  `gh pr merge` anyway.
