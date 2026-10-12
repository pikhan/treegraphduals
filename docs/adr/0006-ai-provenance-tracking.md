# 0006. AI provenance: what is tracked, how, and where it is published

- **Status:** Proposed
- **Date:** 2026-10-11
- **Issues:** #81, #80, #72

## Context

JOSS and pyOpenSci both require a disclosure of AI use (#80), and the
maintainer wants to see where human effort goes: into prompting and into code.
Most of the evidence can't be reconstructed later. Commits carry no AI trailers,
squash merges collapse branch history, and Claude Code deletes session
transcripts after `cleanupPeriodDays` (30 days by default).

#81 left five decisions to the maintainer. They were settled in the issue
(the maintainer's comment of 2026-10-11, US time) and in the `/work-issue 81` session:

- **(a) Tool.** git-ai would work, but the goal is to see where human effort
  went, not attribution alone. Custom scripts only if git-ai can't do it.
- **(b) Formal approvals.** No bot identity. Commits and PRs stay under the
  maintainer's identity, and the maintainer reviews and merges every PR.
- **(c) Merge strategy.** Keep squash merges. The PR process doesn't change.
- **(d) What to publish.** Aggregates and per-PR numbers. No transcripts.
- **(e) Retention.** The 30-day default is fine.

## Options considered

- **git-ai** (open source, Apache-2.0) records line-level AI authorship in git
  notes through agent hooks. It needs a background daemon and user-level hooks
  installed by `curl | bash`. Keeping notes through GitHub's squash merge needs a
  CI workflow that installs git-ai the same way and pushes with a
  `contents: write` token. That cuts against the workflow hardening of #97, and
  git-ai doesn't measure prompting effort at all.
- **A per-PR count from transcripts and commits.** One stdlib-only script reads
  the transcripts for steering, decides per commit whether an agent made it,
  and blames the PR's added lines. It needs no new tool and no write-token
  workflow beyond the existing pattern, but gives per-PR totals, not per-line
  blame on `main`.

The maintainer chose the per-PR count.

## Decision

`scripts/ai_provenance.py` records, for each PR, one comment that starts with
`<!-- ai-provenance:v1 -->` and holds a JSON record. It is computed on the
maintainer's machine, where the transcripts are. `/work-issue` step 8 runs it,
and a `SessionEnd` hook in `.claude/settings.json` runs it again in the
background when a session closes.

**Sessions → PRs.** A session counts for a PR if it created the PR (a `pr-link`
record), worked on its head branch, or is named in the PR body. Within a
session, a transcript record counts for the PR whose head branch it was on.
Otherwise it counts for the first PR the session linked at or after it, or the
last one if it came after all of them.

**Human steering**, per PR:

- *messages*: user records with `origin.kind == "human"`, excluding interrupts. A
  slash command counts as the `/name args` that was typed.
- *words* and *characters* of those messages, plus words typed into
  AskUserQuestion answers. Picking an offered option counts as an *answer*, not
  as words.
- *active time*: for each message, the time since Claude's previous output; for
  each answer, the time since the question. Each is capped at **15 minutes**, so
  breaks don't count. A session's first message counts zero.
- *wall clock*: the span of the session's records that count for the PR.
- *messages after approval*: messages after the first reviewer hand-back with
  verdict APPROVE.

**Agents**, per PR: models, effort levels and tokens (input including cache
reads and writes; output), deduplicated per API response, from the main and
subagent transcripts; agent time as the length of the union of the intervals
`[t − durationMs, t]` over the `turn_duration` records (within one prompt the
value is sometimes cumulative and sometimes per turn, and the union is right in
both cases); reviewer verdicts.

**Line authorship.** Each commit in `merge-base..head` is *ai* if an agent's
Bash output showed git's `[branch sha]` summary for it, or if its **author
date** falls within one of the agents' Bash calls (±2 s). The author date
survives rebases such as GitHub's "Update branch". Every other commit is
*human*. Each non-blank line the PR adds is blamed over that range. A line from
an *ai* commit is `ai`. A line from a *human* commit is `ai_edited` if that
commit's hunk replaced lines an *ai* commit had added, and otherwise `human`.
Lines are split into `src/`, `tests/`, `docs/` and other. The **coverage rule**
follows: whatever no transcript shows an agent doing counts as human, so
AI-written code must enter only through Claude Code (already a `CLAUDE.md`
rule).

Re-running keeps what an earlier run recorded: sessions whose transcripts have
expired, and commits already classified *ai*.

**Publishing.** The `Stats` workflow (on push to `main`, nightly, and on
demand) reads every merged PR's provenance comment, plus GitHub's data: reviewer
rounds and verdicts from the `Automated review (round N)` comments, formal
reviews, line comments, commits, time open, and which closed `difficulty: max`
issues human-written PRs closed. It writes `ai-stats.json` to the orphan `stats`
branch, the same pattern as the coverage badge. A PR whose final head differs
from the head its record was computed at is flagged `stale`. Each row also
counts the PR's commits by class, so human commits pushed to agent PRs are
published. Only comments by the repository owner (`authorAssociation`
`OWNER`) are read, and a malformed record is ignored, so other people's
comments can't forge or break the numbers. The Read the Docs
build renders the JSON on the *AI provenance* page. Nothing from the transcripts
is published except these numbers.

**Test strength.** mutmut runs nightly in the same workflow on
`src/treegraphduals/core` and `timeseries` (plotting code is excluded). The
score, (killed + timeout + segfault) / (those + survived + suspicious +
no tests), is published with the stats. Deterministic Hypothesis in CI stays
with #51.

## Consequences

- Every agent PR gets a provenance comment. Human PRs without one count as
  human when their AI use section says "None: written by a human".
- PRs merged before this lands can be backfilled with `collect <pr>` while their
  transcripts still exist (30 days).
- Known gaps: a human edit that an agent commits along with its own, a human
  amending an agent's commit, or a human commit authored while an agent's Bash
  call (such as a long test run) was running in a session on the same PR, counts
  as AI. Code pasted from other AI chats
  counts as human. Effort in sessions that never touch a PR (planning, triage)
  isn't attributed to any PR.
- Formal approval records stay impossible (decision b). The review metrics are
  the reviewer agent's rounds and the maintainer's messages after its approval.
- #72's page builds on the generated statistics and this ADR.
