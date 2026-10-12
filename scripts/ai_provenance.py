"""Records where AI and human effort went in each pull request (#81, ADR 0006).

Three kinds of numbers, per PR:

- **Steering**: what the maintainer typed into the Claude Code sessions that
  worked on the PR (messages, words, answers to AskUserQuestion) and an estimate
  of the time spent reading and typing ("active time").
- **Agents**: models, effort levels, tokens, agent working time, and reviewer
  hand-backs, from the same transcripts.
- **Authorship**: every non-blank line the PR adds, classified as ``ai`` (its
  commit was made by an agent in a Claude Code session), ``ai_edited`` (a human
  commit that rewrote lines an agent commit had added) or ``human`` (anything
  else). This is the coverage rule: what no transcript shows counts as human.

The transcripts live only on the maintainer's machine and Claude Code deletes
them after ``cleanupPeriodDays``, so ``collect`` saves the numbers in one marked
comment on the PR. ``aggregate`` (CI) reads those comments plus GitHub's review
data into one stats JSON, and ``render`` (Read the Docs) turns that into tables.

Usage::

    python scripts/ai_provenance.py collect 101 [102 ...] [--dry-run]
    python scripts/ai_provenance.py session-end < hook-payload.json
    python scripts/ai_provenance.py aggregate stats.json [--mutation m.json] [--previous old.json]
    python scripts/ai_provenance.py render stats.json-or-URL docs/ai_stats.md

Only the standard library and the ``git`` and ``gh`` command-line tools are
used, so CI, Read the Docs and the SessionEnd hook can run it without ``uv``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 1
MARKER = "<!-- ai-provenance:v1 -->"
# A gap longer than this between Claude's last output and the next human input
# counts as a break, not as reading and typing (ADR 0006).
ACTIVE_CAP_SECONDS = 15 * 60
CATEGORIES = ("src", "tests", "docs", "other")
CLASSES = ("ai", "ai_edited", "human")
# git's summary line after a commit: "[branch 1a2b3c4] message", also
# "[main (root-commit) 1a2b3c4]" and "[detached HEAD 1a2b3c4]".
COMMIT_SUMMARY = re.compile(r"^\[[^\]\n]* ([0-9a-f]{7,40})\]", re.MULTILINE)
VERDICT = re.compile(r"Verdict\W*(APPROVE|CHANGES REQUESTED)")
REVIEW_HEADING = re.compile(r"\A\W*Automated review \(round (\d+)\)")
JSON_BLOCK = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
SESSION_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


# --------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------


@dataclass
class Event:
    """One thing in a transcript that counts towards a PR.

    ``kind`` is ``prompt`` (a typed message or slash command), ``answer`` (one
    AskUserQuestion call answered), ``turn`` (an agent turn, ``seconds`` long),
    ``usage`` (one model response), ``handback`` (a reviewer report) or
    ``activity`` (any other record, used only for the wall-clock span).
    """

    kind: str
    timestamp: datetime
    branch: str | None
    words: int = 0
    chars: int = 0
    seconds: float = 0.0
    model: str | None = None
    effort: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    verdict: str | None = None


@dataclass
class Session:
    """What one Claude Code session (with its subagents) did."""

    session_id: str
    events: list[Event] = field(default_factory=list)
    # (time of the first link, PR number), sorted by time.
    pr_links: list[tuple[datetime, int]] = field(default_factory=list)
    branches: set[str] = field(default_factory=set)
    # Commits an agent made: SHAs from git's "[branch sha]" summary lines, and the
    # (start, end) of every Bash call, to match against commit times.
    commits: set[str] = field(default_factory=set)
    bash_windows: list[tuple[datetime, datetime]] = field(default_factory=list)


def parse_time(value: str) -> datetime:
    """Parse a transcript or GitHub timestamp such as ``2026-10-12T01:42:26.976Z``."""
    return datetime.fromisoformat(value)


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON records of a transcript, skipping unreadable lines."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def human_text(content: Any) -> str:
    """Return what the human typed, given a human message's ``content``.

    A slash command is stored as ``<command-name>`` and ``<command-args>`` tags;
    it is counted as the ``/name args`` the human typed.

    Examples
    --------
    >>> human_text("<command-name>/work-issue</command-name><command-args>81</command-args>")
    '/work-issue 81'
    >>> human_text([{"type": "text", "text": "fix it"}, {"type": "image"}])
    'fix it'
    """
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    text = str(content or "")
    name = re.search(r"<command-name>(.*?)</command-name>", text, re.DOTALL)
    if name:
        args = re.search(r"<command-args>(.*?)</command-args>", text, re.DOTALL)
        return (
            f"{name.group(1).strip()} {args.group(1).strip() if args else ''}".strip()
        )
    return text


def typed_answer_words(result: dict[str, Any]) -> int:
    """Count the words a human typed into an AskUserQuestion answer.

    Picking an offered option is a click, so it counts zero. A custom ("Other")
    answer and any notes count their words.
    """
    labels = {
        option.get("label")
        for question in result.get("questions") or []
        for option in question.get("options") or []
    }
    words = 0
    for answer in (result.get("answers") or {}).values():
        for part in str(answer).split(", "):
            if part not in labels:
                words += len(part.split())
    for annotation in (result.get("annotations") or {}).values():
        if isinstance(annotation, dict):
            words += len(str(annotation.get("notes") or "").split())
    return words


def parse_session(mains: Iterable[Path], subagents: Iterable[Path] = ()) -> Session:
    """Read one session's transcript(s) and its subagents' transcripts.

    A session that moved between working directories (into a worktree and back)
    can leave a transcript in more than one project directory: pass them all.
    """
    mains = list(mains)
    session = Session(session_id=mains[0].stem)
    links: dict[int, datetime] = {}
    for path, is_main in [
        *((p, True) for p in mains),
        *((p, False) for p in subagents),
    ]:
        last_output: datetime | None = None
        asked: dict[str, datetime] = {}
        commands: dict[str, tuple[datetime, str]] = {}
        seen_responses: set[str] = set()
        for record in iter_records(path):
            kind = record.get("type")
            stamp = record.get("timestamp")
            if (
                kind == "pr-link"
                and is_main
                and isinstance(record.get("prNumber"), int)
            ):
                when = parse_time(stamp) if stamp else datetime.min.replace(tzinfo=UTC)
                links.setdefault(record["prNumber"], when)
                continue
            if not stamp:
                continue
            when = parse_time(stamp)
            branch = record.get("gitBranch")
            if branch:
                session.branches.add(branch)
            message = record.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            origin = record.get("origin")
            origin = origin if isinstance(origin, dict) else {}
            session.events.append(Event("activity", when, branch))

            if kind == "assistant" and isinstance(message, dict):
                last_output = when
                for block in content if isinstance(content, list) else []:
                    if block.get("type") != "tool_use":
                        continue
                    if block.get("name") == "AskUserQuestion":
                        asked[block.get("id")] = when
                    elif block.get("name") == "Bash":
                        command = str((block.get("input") or {}).get("command", ""))
                        commands[block.get("id")] = (when, command)
                response = message.get("id")
                usage = message.get("usage")
                if response and response not in seen_responses and usage:
                    seen_responses.add(response)
                    session.events.append(
                        Event(
                            "usage",
                            when,
                            branch,
                            model=message.get("model"),
                            effort=record.get("effort"),
                            input_tokens=int(usage.get("input_tokens") or 0)
                            + int(usage.get("cache_read_input_tokens") or 0)
                            + int(usage.get("cache_creation_input_tokens") or 0),
                            output_tokens=int(usage.get("output_tokens") or 0),
                        )
                    )
            elif kind == "system" and record.get("subtype") == "turn_duration":
                last_output = when
                if is_main:
                    seconds = float(record.get("durationMs") or 0) / 1000
                    session.events.append(Event("turn", when, branch, seconds=seconds))
            elif kind == "user" and is_main and origin.get("kind") == "human":
                text = human_text(content)
                if text.startswith("[Request interrupted"):
                    continue
                gap = (when - last_output).total_seconds() if last_output else 0.0
                session.events.append(
                    Event(
                        "prompt",
                        when,
                        branch,
                        words=len(text.split()),
                        chars=len(text),
                        seconds=min(max(gap, 0.0), ACTIVE_CAP_SECONDS),
                    )
                )
            elif kind == "user" and is_main and origin.get("kind") == "peer":
                if origin.get("name") == "reviewer":
                    found = VERDICT.findall(str(origin.get("body") or content or ""))
                    session.events.append(
                        Event(
                            "handback",
                            when,
                            branch,
                            verdict=found[0] if found else None,
                        )
                    )
            elif kind == "user" and isinstance(content, list):
                result = record.get("toolUseResult")
                for block in content:
                    if (
                        not isinstance(block, dict)
                        or block.get("type") != "tool_result"
                    ):
                        continue
                    tool_id = block.get("tool_use_id")
                    if tool_id in asked and is_main and isinstance(result, dict):
                        gap = (when - asked.pop(tool_id)).total_seconds()
                        session.events.append(
                            Event(
                                "answer",
                                when,
                                branch,
                                words=typed_answer_words(result),
                                seconds=min(max(gap, 0.0), ACTIVE_CAP_SECONDS),
                            )
                        )
                    elif tool_id in commands:
                        started, command = commands.pop(tool_id)
                        session.bash_windows.append((started, when))
                        if "git" not in command:
                            continue
                        output = (
                            result.get("stdout")
                            if isinstance(result, dict)
                            else block.get("content")
                        )
                        session.commits.update(
                            COMMIT_SUMMARY.findall(str(output or ""))
                        )
    session.pr_links = sorted((when, number) for number, when in links.items())
    session.events.sort(key=lambda event: event.timestamp)
    return session


def find_sessions(
    projects: Path, repo_dir: Path
) -> dict[str, tuple[list[Path], list[Path]]]:
    """Map each session in this repository to its transcripts and subagent transcripts.

    Claude Code names a project directory after the session's working directory
    with every non-alphanumeric character replaced by ``-``. Worktrees under
    ``.claude/worktrees/`` therefore share the main checkout's prefix.
    """
    prefix = re.sub(r"[^A-Za-z0-9]", "-", str(repo_dir))
    found: dict[str, tuple[list[Path], list[Path]]] = {}
    worktrees = f"{prefix}--claude-worktrees-"
    for project in sorted(projects.glob(f"{prefix}*")):
        if project.name != prefix and not project.name.startswith(worktrees):
            continue  # a sibling such as treegraphduals-old
        for main in sorted(project.glob("*.jsonl")):
            mains, subs = found.setdefault(main.stem, ([], []))
            mains.append(main)
            subs.extend(sorted((project / main.stem / "subagents").glob("*.jsonl")))
    return found


# --------------------------------------------------------------------------
# Attributing events to pull requests
# --------------------------------------------------------------------------


def assign_pr(
    when: datetime,
    branch: str | None,
    links: list[tuple[datetime, int]],
    branch_prs: dict[str, int],
    default: int | None,
) -> int | None:
    """Decide which PR one transcript event counts towards.

    An event on a PR's head branch counts for that PR. Otherwise it counts for
    the first PR the session linked at or after the event (so the prompt that
    started the work counts for the PR it produced), or for the last one linked
    if it came after all of them. A session that never linked a PR counts
    everything for ``default``.

    Examples
    --------
    >>> from datetime import datetime, UTC
    >>> t = lambda h: datetime(2026, 10, 1, h, tzinfo=UTC)
    >>> links = [(t(2), 7), (t(4), 9)]
    >>> [assign_pr(t(h), "main", links, {}, None) for h in (1, 3, 5)]
    [7, 9, 9]
    >>> assign_pr(t(5), "issue-7-x", links, {"issue-7-x": 7}, None)
    7
    """
    if branch in branch_prs:
        return branch_prs[branch]
    for linked_at, number in links:
        if linked_at >= when:
            return number
    if links:
        return links[-1][1]
    return default


def busy_seconds(turns: list[tuple[datetime, float]]) -> float:
    """Return the length of the union of the turn intervals ``[end - seconds, end]``.

    A ``turn_duration`` record ends a turn and gives its length. Within one
    prompt the value is sometimes cumulative (each record covers the earlier
    ones) and sometimes per turn, so neither a sum nor a maximum is right. The
    union of the intervals is right in both cases.

    Examples
    --------
    >>> from datetime import datetime, timedelta, UTC
    >>> t = lambda s: datetime(2026, 10, 1, tzinfo=UTC) + timedelta(seconds=s)
    >>> busy_seconds([(t(20), 20), (t(100), 100)])  # cumulative: [0, 20] in [0, 100]
    100.0
    >>> busy_seconds([(t(13), 13), (t(190), 7), (t(249), 3)])  # separate turns
    23.0
    """
    total = 0.0
    covered_until: datetime | None = None
    for start, end in sorted((end - timedelta(seconds=s), end) for end, s in turns):
        if covered_until is not None and start < covered_until:
            start = covered_until
        if end > start:
            total += (end - start).total_seconds()
        covered_until = end if covered_until is None else max(covered_until, end)
    return total


def summarize(events: list[Event]) -> dict[str, Any]:
    """Total up the events one session contributed to one PR.

    ``messages_after_approval`` counts human prompts sent after the first
    reviewer hand-back whose verdict was APPROVE: steering that the automated
    review did not settle.
    """
    prompts = [e for e in events if e.kind == "prompt"]
    answers = [e for e in events if e.kind == "answer"]
    approved = min(
        (
            e.timestamp
            for e in events
            if e.kind == "handback" and e.verdict == "APPROVE"
        ),
        default=None,
    )
    models: dict[str, dict[str, int]] = {}
    for e in events:
        if e.kind == "usage" and e.model and not e.model.startswith("<"):
            row = models.setdefault(
                e.model, {"responses": 0, "input_tokens": 0, "output_tokens": 0}
            )
            row["responses"] += 1
            row["input_tokens"] += e.input_tokens
            row["output_tokens"] += e.output_tokens
    span = (events[-1].timestamp - events[0].timestamp).total_seconds() if events else 0
    return {
        "first": events[0].timestamp.isoformat() if events else None,
        "last": events[-1].timestamp.isoformat() if events else None,
        "human_messages": len(prompts),
        "human_words": sum(e.words for e in prompts) + sum(e.words for e in answers),
        "human_chars": sum(e.chars for e in prompts),
        "answers": len(answers),
        "active_seconds": round(sum(e.seconds for e in prompts + answers)),
        "wall_clock_seconds": round(span),
        "agent_seconds": round(
            busy_seconds([(e.timestamp, e.seconds) for e in events if e.kind == "turn"])
        ),
        "messages_after_approval": sum(
            1 for e in prompts if approved is not None and e.timestamp > approved
        ),
        "reviewer_handbacks": [e.verdict for e in events if e.kind == "handback"],
        "efforts": sorted({e.effort for e in events if e.kind == "usage" and e.effort}),
        "models": models,
    }


def session_contribution(
    session: Session,
    pr: int,
    head_branch: str | None,
    branch_prs: dict[str, int],
    linked_by_body: bool,
) -> dict[str, Any] | None:
    """Summarize what ``session`` did for ``pr``, or return None if it didn't touch it."""
    linked = (
        linked_by_body
        or any(number == pr for _, number in session.pr_links)
        or (head_branch is not None and head_branch in session.branches)
    )
    if not linked:
        return None
    default = pr if not session.pr_links else None
    mine = [
        e
        for e in session.events
        if assign_pr(e.timestamp, e.branch, session.pr_links, branch_prs, default) == pr
    ]
    return summarize(mine) if mine else None


# --------------------------------------------------------------------------
# Line authorship
# --------------------------------------------------------------------------


def git(*args: str, cwd: Path = ROOT) -> str:
    """Run git and return its standard output."""
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def category(path: str) -> str:
    """Return the line-count category of a repository path.

    Examples
    --------
    >>> [category(p) for p in ("src/a.py", "tests/t.py", "docs/x.md", "README.md")]
    ['src', 'tests', 'docs', 'other']
    """
    top = path.split("/", 1)[0]
    return top if top in CATEGORIES else "other"


def blame(rev_range: str, path: str, cwd: Path, lines: str | None = None) -> list[dict]:
    """Blame ``path`` over ``rev_range`` and return one dict per line.

    Each dict has ``sha``, ``line`` (the line number in that commit), ``path``
    (the file name in that commit), ``boundary`` and ``text``.
    """
    args = ["blame", "--line-porcelain"]
    if lines:
        args += ["-L", lines]
    out = git(*args, rev_range, "--", path, cwd=cwd)
    result: list[dict] = []
    current: dict[str, Any] = {}
    for raw in out.splitlines():
        if raw.startswith("\t"):
            current["text"] = raw[1:]
            result.append(current)
            current = {}
        elif not current:
            sha, orig, *_ = raw.split(" ")
            current = {"sha": sha, "line": int(orig), "boundary": False, "path": path}
        elif raw == "boundary":
            current["boundary"] = True
        elif raw.startswith("filename "):
            current["path"] = raw[len("filename ") :]
    return result


def ai_edited_lines(
    commit: str, base: str, classes: dict[str, str], cwd: Path
) -> set[tuple[str, int]]:
    """Return the (path, line) pairs that a human ``commit`` wrote over AI lines.

    A hunk of ``commit`` that removes at least one line an agent commit added
    after ``base``, and adds lines in its place, marks those added lines.
    """
    marked: set[tuple[str, int]] = set()
    # Fixed prefixes and unquoted paths, whatever the user's git config says.
    # git ends a "---"/"+++" path containing a space with a tab.
    diff = git(
        "-c", "core.quotePath=false", "diff", "-U0", "--no-color", "--no-ext-diff",
        "--src-prefix=a/", "--dst-prefix=b/", f"{commit}^", commit, cwd=cwd,
    )  # fmt: skip
    old_path = new_path = None
    for raw in diff.splitlines():
        if raw.startswith("--- "):
            old_path = None if raw == "--- /dev/null" else raw[6:].rstrip("\t")
        elif raw.startswith("+++ "):
            new_path = None if raw == "+++ /dev/null" else raw[6:].rstrip("\t")
        elif raw.startswith("@@") and old_path and new_path:
            hunk = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", raw)
            if not hunk:
                continue
            old_start, old_count = int(hunk[1]), int(hunk[2] or 1)
            new_start, new_count = int(hunk[3]), int(hunk[4] or 1)
            if not old_count or not new_count:
                continue
            replaced = blame(
                f"{base}..{commit}^", old_path, cwd, f"{old_start},+{old_count}"
            )
            if any(
                not row["boundary"] and classes.get(row["sha"]) == "ai"
                for row in replaced
            ):
                marked.update(
                    (new_path, n) for n in range(new_start, new_start + new_count)
                )
    return marked


def made_by_agent(
    sha: str,
    authored: datetime,
    ai_shas: set[str],
    windows: list[tuple[datetime, datetime]],
) -> bool:
    """Decide whether an agent made a commit.

    It did if its SHA appeared in git's ``[branch sha]`` summary in an agent's
    Bash output, or if its author date falls while one of the agent's Bash calls
    was running (``git commit -q`` and scripts print no summary). The author
    date, unlike the committer date, survives a rebase, such as GitHub's "Update
    branch" button. Commit times have one-second resolution, hence the slack.

    Examples
    --------
    >>> from datetime import datetime, timedelta, UTC
    >>> t = datetime(2026, 10, 1, 12, tzinfo=UTC)
    >>> made_by_agent("ab12cd3", t, set(), [(t - timedelta(seconds=5), t)])
    True
    >>> made_by_agent("ab12cd3", t, {"ab12cd3"}, [])
    True
    >>> made_by_agent("ab12cd3", t, set(), [(t + timedelta(minutes=1), t + timedelta(minutes=2))])
    False
    """
    if any(sha.startswith(short) for short in ai_shas):
        return True
    slack = 2.0
    return any(
        (start - authored).total_seconds() <= slack
        and (authored - end).total_seconds() <= slack
        for start, end in windows
    )


def classify_commits(
    base: str,
    head: str,
    ai_shas: set[str],
    windows: list[tuple[datetime, datetime]],
    known: dict[str, str],
    cwd: Path,
) -> dict[str, str]:
    """Classify every commit in ``base..head`` as ``ai`` or ``human``.

    See `made_by_agent`. A commit an earlier run classified as ``ai`` stays
    ``ai``, so the record survives the transcripts being deleted.
    """
    classes = {}
    log = git("log", "--format=%H %aI", f"{base}..{head}", cwd=cwd)
    for line in log.splitlines():
        sha, stamp = line.split(" ", 1)
        agent = made_by_agent(sha, parse_time(stamp), ai_shas, windows)
        classes[sha] = "ai" if agent or known.get(sha) == "ai" else "human"
    return classes


def count_lines(
    base: str, head: str, classes: dict[str, str], cwd: Path
) -> dict[str, dict[str, int]]:
    """Count the non-blank lines ``base..head`` adds, by category and class."""
    counts = {cat: dict.fromkeys(CLASSES, 0) for cat in CATEGORIES}
    edited: dict[str, set[tuple[str, int]]] = {}
    numstat = git("diff", "--numstat", "--no-renames", "-z", base, head, cwd=cwd)
    for entry in numstat.split("\0"):
        parts = entry.split("\t")
        if len(parts) != 3 or parts[0] in ("-", "0"):
            continue  # binary file, or nothing added
        path = parts[2]
        for row in blame(f"{base}..{head}", path, cwd):
            if row["boundary"] or not row["text"].strip():
                continue
            cls = classes.get(row["sha"], "human")
            if cls == "human":
                if row["sha"] not in edited:
                    edited[row["sha"]] = ai_edited_lines(row["sha"], base, classes, cwd)
                if (row["path"], row["line"]) in edited[row["sha"]]:
                    cls = "ai_edited"
            counts[category(path)][cls] += 1
    return counts


# --------------------------------------------------------------------------
# GitHub
# --------------------------------------------------------------------------


def gh(*args: str, stdin: str | None = None, cwd: Path = ROOT) -> str:
    """Run the GitHub CLI and return its standard output."""
    return subprocess.run(
        ["gh", *args], cwd=cwd, input=stdin, check=True, capture_output=True, text=True
    ).stdout


def gh_json(*args: str, cwd: Path = ROOT) -> Any:
    """Run the GitHub CLI and parse its JSON output."""
    return json.loads(gh(*args, cwd=cwd) or "null")


def trusted(comment: dict[str, Any]) -> bool:
    """Return whether a comment is the repository owner's.

    Agents post as the maintainer, so provenance records and automated reviews
    come only from the owner. Anyone else's comment is ignored, so it can't
    forge or break the published numbers. ``gh pr list --json`` spells the
    field ``authorAssociation``, the REST API ``author_association``.

    Examples
    --------
    >>> trusted({"authorAssociation": "OWNER"}), trusted({"author_association": "NONE"})
    (True, False)
    """
    association = comment.get("authorAssociation") or comment.get("author_association")
    return association == "OWNER"


def valid_record(record: Any) -> bool:
    """Return whether ``record`` has the shape `aggregate` and `collect` rely on."""
    numeric = (int, float)
    try:
        return (
            record["schema"] == SCHEMA
            and isinstance(record["sessions"], dict)
            and isinstance(record.get("commits", {}), dict)
            and all(
                isinstance(record["lines"][cat][cls], int)
                for cat in CATEGORIES
                for cls in CLASSES
            )
            and all(
                isinstance(value, numeric)
                for key, value in record["totals"].items()
                if key not in ("models", "efforts", "reviewer_handbacks")
            )
        )
    except (KeyError, TypeError, AttributeError):
        return False


def provenance_from_body(body: str) -> dict[str, Any] | None:
    """Extract the JSON record from a provenance comment body, if it is a valid one."""
    if not body.startswith(MARKER):
        return None
    match = JSON_BLOCK.search(body)
    if not match:
        return None
    try:
        record = json.loads(match[1])
    except ValueError:
        return None
    return record if valid_record(record) else None


def provenance_comment(
    comments: Iterable[dict[str, Any]],
) -> tuple[dict | None, dict | None]:
    """Return the first trusted comment with a valid record, and the record."""
    for comment in comments:
        if trusted(comment):
            record = provenance_from_body(comment.get("body") or "")
            if record is not None:
                return comment, record
    return None, None


def find_comment(repo: str, pr: int, cwd: Path) -> tuple[int | None, dict | None]:
    """Return the id and record of the PR's provenance comment, if any."""
    pages = gh_json(
        "api", "--paginate", "--slurp", f"repos/{repo}/issues/{pr}/comments", cwd=cwd
    )
    comment, record = provenance_comment(c for page in pages or [] for c in page)
    return (comment["id"] if comment else None), record


def totals(sessions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Add up per-session summaries into the PR's totals."""
    keys = (
        "human_messages",
        "human_words",
        "human_chars",
        "answers",
        "active_seconds",
        "wall_clock_seconds",
        "agent_seconds",
        "messages_after_approval",
    )
    result: dict[str, Any] = {
        key: sum(s.get(key, 0) for s in sessions.values()) for key in keys
    }
    models: dict[str, dict[str, int]] = {}
    for s in sessions.values():
        for model, row in s.get("models", {}).items():
            into = models.setdefault(model, dict.fromkeys(row, 0))
            for key, value in row.items():
                into[key] = into.get(key, 0) + value
    result["models"] = models
    result["efforts"] = sorted(
        {e for s in sessions.values() for e in s.get("efforts", [])}
    )
    result["reviewer_handbacks"] = [
        v for s in sessions.values() for v in s.get("reviewer_handbacks", [])
    ]
    return result


def comment_body(record: dict[str, Any]) -> str:
    """Render a provenance record as the PR comment."""
    t = record["totals"]
    lines = record["lines"]
    all_lines = {cls: sum(lines[cat][cls] for cat in CATEGORIES) for cls in CLASSES}
    added = sum(all_lines.values())
    share = f"{100 * all_lines['ai'] / added:.0f}%" if added else "n/a"
    rows = "\n".join(
        f"| `{cat}/` | {lines[cat]['ai']} | {lines[cat]['ai_edited']} | {lines[cat]['human']} |"
        if cat != "other"
        else f"| other | {lines[cat]['ai']} | {lines[cat]['ai_edited']} | {lines[cat]['human']} |"
        for cat in CATEGORIES
    )
    models = ", ".join(f"`{m}`" for m in t["models"]) or "none recorded"
    commits = Counter(record.get("commits", {}).values())
    return f"""{MARKER}
### AI provenance

Recorded automatically by `scripts/ai_provenance.py` from {len(record["sessions"])} \
Claude Code session(s) and the PR's commits (ADR 0006). Lines that no transcript \
shows an agent committing count as human.

- **Human steering:** {t["human_messages"]} message(s), {t["human_words"]} word(s), \
{t["answers"]} question(s) answered, about {t["active_seconds"] / 60:.0f} min active \
({t["wall_clock_seconds"] / 3600:.1f} h wall clock); \
{t["messages_after_approval"]} message(s) after the reviewer approved.
- **Agents:** {models}; {t["agent_seconds"] / 60:.0f} min of agent turns; \
reviewer verdicts: {", ".join(str(v) for v in t["reviewer_handbacks"]) or "none"}.
- **Commits:** {commits["ai"]} by agents, {commits["human"]} by a human.
- **Added lines:** {added} non-blank, {share} AI-written.

| | AI | AI, edited by human | Human |
|---|---:|---:|---:|
{rows}

<details><summary>Data</summary>

```json
{json.dumps(record, indent=1, sort_keys=True)}
```

</details>
"""


def projects_dir() -> Path:
    """Return the directory holding Claude Code's per-project transcripts."""
    config = os.environ.get("CLAUDE_CONFIG_DIR") or str(Path.home() / ".claude")
    return Path(config) / "projects"


def repo_toplevel(cwd: Path) -> Path:
    """Return the main checkout of the repository (not a worktree)."""
    common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=cwd).strip()
    )
    return common.parent


def collect(pr: int, cwd: Path = ROOT, dry_run: bool = False) -> dict[str, Any]:
    """Compute one PR's provenance record and save it as the PR's comment."""
    repo = gh_json("repo", "view", "--json", "nameWithOwner", cwd=cwd)["nameWithOwner"]
    info = gh_json(
        "pr",
        "view",
        str(pr),
        "--json",
        "headRefName,headRefOid,baseRefName,body",
        cwd=cwd,
    )
    git("fetch", "--quiet", "origin", f"pull/{pr}/head", info["baseRefName"], cwd=cwd)
    head = info["headRefOid"]
    base = git("merge-base", head, f"origin/{info['baseRefName']}", cwd=cwd).strip()
    comment_id, old = find_comment(repo, pr, cwd)
    old = old or {}

    body_sessions = set(SESSION_ID.findall(info.get("body") or ""))
    sessions = [
        parse_session(mains, subs)
        for mains, subs in find_sessions(projects_dir(), repo_toplevel(cwd)).values()
    ]
    heads: dict[int, str] = {pr: info["headRefName"]}
    for session in sessions:
        for _, number in session.pr_links:
            if number not in heads:
                try:
                    heads[number] = gh_json(
                        "pr", "view", str(number), "--json", "headRefName", cwd=cwd
                    )["headRefName"]
                except subprocess.CalledProcessError:
                    pass  # unknown head branch: time order alone attributes it
    branch_prs = {branch: number for number, branch in heads.items()}

    per_session = dict(old.get("sessions", {}))  # keeps expired transcripts' numbers
    ai_shas: set[str] = set()
    windows: list[tuple[datetime, datetime]] = []
    for session in sessions:
        summary = session_contribution(
            session,
            pr,
            info["headRefName"],
            branch_prs,
            session.session_id in body_sessions,
        )
        if summary is not None:
            per_session[session.session_id] = summary
            ai_shas |= session.commits
            windows += session.bash_windows
    classes = classify_commits(
        base, head, ai_shas, windows, old.get("commits", {}), cwd
    )
    record = {
        "schema": SCHEMA,
        "pr": pr,
        "head_sha": head,
        "base_sha": base,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sessions": per_session,
        "commits": classes,
        "lines": count_lines(base, head, classes, cwd),
        "totals": totals(per_session),
    }
    body = comment_body(record)
    if dry_run:
        print(body)
        return record
    if comment_id is None:
        method, endpoint = "POST", f"repos/{repo}/issues/{pr}/comments"
    else:
        method, endpoint = "PATCH", f"repos/{repo}/issues/comments/{comment_id}"
    payload = json.dumps({"body": body})
    gh("api", "-X", method, endpoint, "--input", "-", stdin=payload, cwd=cwd)
    return record


def session_end(payload: dict[str, Any]) -> None:
    """SessionEnd hook: collect, in the background, every PR the session touched.

    Claude Code gives SessionEnd hooks little time, so this only starts a
    detached process and returns. Its output goes to ``ai-provenance.log`` in
    the repository's git directory.
    """
    cwd = Path(payload.get("cwd") or ROOT)
    if not cwd.is_dir():  # a worktree removed on exit
        cwd = ROOT
    transcript = payload.get("transcript_path")
    if not transcript:
        return
    log = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=cwd).strip()
    )
    with (log / "ai-provenance.log").open("a") as out:
        subprocess.Popen(
            [sys.executable, __file__, "collect-session", transcript],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def collect_session(transcript: Path, cwd: Path) -> None:
    """Collect every PR that one session linked or whose branch it worked on."""
    numbers = {number for _, number in parse_session([transcript]).pr_links}
    try:
        current = gh_json("pr", "view", "--json", "number", cwd=cwd)
        numbers.add(current["number"])
    except subprocess.CalledProcessError:
        pass  # the current branch has no PR
    print(
        f"{datetime.now(UTC).isoformat()} session {transcript.stem}: PRs {sorted(numbers)}"
    )
    for number in sorted(numbers):
        try:
            collect(number, cwd)
        except subprocess.CalledProcessError as error:
            print(f"  PR {number}: {error} {error.stderr}")


# --------------------------------------------------------------------------
# Aggregate (CI) and render (docs)
# --------------------------------------------------------------------------


def review_rounds(comments: list[dict[str, Any]]) -> list[str | None]:
    r"""Return the verdict of each ``Automated review (round N)`` comment, in order.

    Only the owner's comments count (see `trusted`).

    Examples
    --------
    >>> review = "## Automated review (round 1)\n**Verdict: APPROVE**"
    >>> review_rounds([{"body": review, "authorAssociation": "OWNER"},
    ...                {"body": review, "authorAssociation": "NONE"},
    ...                {"body": "unrelated", "authorAssociation": "OWNER"}])
    ['APPROVE']
    """
    rounds = []
    for comment in comments:
        body = comment.get("body") or ""
        if trusted(comment) and REVIEW_HEADING.match(body):
            found = VERDICT.search(body)
            rounds.append(found[1] if found else None)
    return rounds


def hours_between(start: str | None, end: str | None) -> float | None:
    """Return the hours between two timestamps, or None if either is missing."""
    if not start or not end:
        return None
    return round((parse_time(end) - parse_time(start)).total_seconds() / 3600, 2)


def pr_row(pr: dict[str, Any], details: dict[str, Any], max_issues: set[int]) -> dict:
    """Build the stats row for one merged PR from GitHub's data.

    ``pr`` comes from ``gh pr list --json``, ``details`` from the REST
    ``pulls/<n>`` endpoint (``commits`` and ``review_comments`` counts), and
    ``max_issues`` are the closed ``difficulty: max`` issues.
    """
    comments = pr.get("comments") or []
    _, record = provenance_comment(comments)
    body = pr.get("body") or ""
    if record is not None:
        ai_lines = sum(
            record["lines"][c]["ai"] + record["lines"][c]["ai_edited"]
            for c in CATEGORIES
        )
        author = "human" if ai_lines == 0 and not record["sessions"] else "ai-assisted"
    elif re.search(r"(?m)^None: written by a human", body):
        author = "human"
    else:
        author = "unrecorded"
    closes = [i["number"] for i in pr.get("closingIssuesReferences") or []]
    return {
        "number": pr["number"],
        "title": pr["title"],
        "created_at": pr["createdAt"],
        "merged_at": pr["mergedAt"],
        "hours_open": hours_between(pr["createdAt"], pr["mergedAt"]),
        "commits": int(details.get("commits") or 0),
        "formal_reviews": len(pr.get("reviews") or []),
        "review_comments": int(details.get("review_comments") or 0),
        "reviewer_rounds": review_rounds(comments),
        "author": author,
        "closes_difficulty_max": sorted(set(closes) & max_issues),
        "stale": record is not None and record.get("head_sha") != pr.get("headRefOid"),
        "provenance": None
        if record is None
        else {
            "lines": record["lines"],
            "totals": record["totals"],
            # Commits by class: the "human" ones are commits the maintainer
            # pushed to the PR outside any Claude Code session.
            "commits": dict(Counter(record.get("commits", {}).values())),
        },
    }


def aggregate_rows(rows: list[dict[str, Any]], mutation: dict | None) -> dict[str, Any]:
    """Compute the headline totals over the per-PR rows."""
    recorded = [r for r in rows if r["provenance"]]
    lines = {cat: dict.fromkeys(CLASSES, 0) for cat in CATEGORIES}
    steering: Counter[str] = Counter()
    models: dict[str, dict[str, int]] = {}
    for row in recorded:
        for cat in CATEGORIES:
            for cls in CLASSES:
                lines[cat][cls] += row["provenance"]["lines"][cat][cls]
        t = row["provenance"]["totals"]
        for key in (
            "human_messages",
            "human_words",
            "answers",
            "active_seconds",
            "wall_clock_seconds",
            "agent_seconds",
            "messages_after_approval",
        ):
            steering[key] += t.get(key, 0)
        for model, usage in t.get("models", {}).items():
            into = models.setdefault(model, dict.fromkeys(usage, 0))
            for key, value in usage.items():
                into[key] = into.get(key, 0) + value
    reviewed = [r for r in rows if r["reviewer_rounds"]]
    return {
        "merged_prs": len(rows),
        "recorded_prs": len(recorded),
        "stale_prs": sum(1 for r in rows if r["stale"]),
        "human_prs": sum(1 for r in rows if r["author"] == "human"),
        "lines": lines,
        "steering": dict(steering),
        "models": models,
        "review": {
            "prs_reviewed": len(reviewed),
            "reviewer_rounds": sum(len(r["reviewer_rounds"]) for r in reviewed),
            "approved_first_round": sum(
                1 for r in reviewed if r["reviewer_rounds"][0] == "APPROVE"
            ),
            "formal_reviews": sum(r["formal_reviews"] for r in rows),
            "review_comments": sum(r["review_comments"] for r in rows),
            "human_commits_on_agent_prs": sum(
                r["provenance"]["commits"].get("human", 0)
                for r in recorded
                if r["author"] == "ai-assisted"
            ),
            "agent_prs_with_human_commits": sum(
                1
                for r in recorded
                if r["author"] == "ai-assisted"
                and r["provenance"]["commits"].get("human", 0)
            ),
        },
        "difficulty_max_closed_by_human_prs": sorted(
            {
                i
                for r in rows
                if r["author"] == "human"
                for i in r["closes_difficulty_max"]
            }
        ),
        "mutation": mutation,
    }


def aggregate(mutation: dict | None, cwd: Path = ROOT) -> dict[str, Any]:
    """Build the stats JSON from every merged PR (needs ``gh`` with a token)."""
    repo = gh_json("repo", "view", "--json", "nameWithOwner", cwd=cwd)["nameWithOwner"]
    # Not `commits`: with it, a page of 100 PRs exceeds GraphQL's node limit.
    # The per-PR REST call below gives the counts instead.
    fields = (
        "number,title,body,createdAt,mergedAt,headRefOid,reviews,comments,"
        "closingIssuesReferences"
    )
    prs = gh_json(
        "pr", "list", "--state", "merged", "--limit", "1000", "--json", fields, cwd=cwd
    )
    closed_max = ["--state", "closed", "--label", "difficulty: max", "--limit", "1000"]
    max_issues = {
        i["number"]
        for i in gh_json("issue", "list", *closed_max, "--json", "number", cwd=cwd)
    }
    rows = []
    for pr in sorted(prs, key=lambda p: p["number"]):
        details = gh_json("api", f"repos/{repo}/pulls/{pr['number']}", cwd=cwd)
        rows.append(pr_row(pr, details, max_issues))
    return {
        "schema": SCHEMA,
        "repository": repo,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "totals": aggregate_rows(rows, mutation),
        "prs": rows,
    }


def mutation_summary(path: Path) -> dict[str, Any]:
    """Turn mutmut's ``export-cicd-stats`` JSON into the stats JSON's entry.

    The score is the share of judged mutants the tests caught. A timeout or a
    segfault counts as caught: the mutant broke the code badly enough to hang or
    crash it. A mutant no test reaches (``no_tests``) counts as not caught.
    Skipped mutants are left out.
    """
    stats = json.loads(path.read_text())
    keys = ("killed", "timeout", "segfault", "survived", "suspicious", "no_tests")
    counts = {key: int(stats.get(key, 0)) for key in keys}
    judged = sum(counts.values())
    caught = counts["killed"] + counts["timeout"] + counts["segfault"]
    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **counts,
        "total": int(stats.get("total", judged)),
        "score": round(100 * caught / judged, 1) if judged else None,
    }


def render(stats: dict[str, Any] | None) -> str:
    """Render the stats JSON as the Markdown included by ``docs/ai_provenance.rst``."""
    if not stats:
        return "_No statistics have been published yet._\n"
    t = stats["totals"]
    s = t["steering"]
    all_lines = {
        cls: sum(t["lines"][cat][cls] for cat in CATEGORIES) for cls in CLASSES
    }
    added = sum(all_lines.values())

    def pct(n: int, d: int) -> str:
        return f"{100 * n / d:.0f}%" if d else "n/a"

    tests = t["lines"]["tests"]
    review = t["review"]
    reviewed = review["prs_reviewed"]
    headline = [
        ("Added lines written by agents", f"{pct(all_lines['ai'], added)} of {added}"),
        ("Agent lines later edited by a human", all_lines["ai_edited"]),
        ("Lines written by a human", all_lines["human"]),
        ("Test lines written by agents", pct(tests["ai"], sum(tests.values()))),
        (
            "Human messages to agents",
            f"{s.get('human_messages', 0)} ({s.get('human_words', 0)} words)",
        ),
        ("Human active time", f"{s.get('active_seconds', 0) / 3600:.1f} h"),
        ("Agent working time", f"{s.get('agent_seconds', 0) / 3600:.1f} h"),
        (
            "Messages after the reviewer agent approved",
            s.get("messages_after_approval", 0),
        ),
        (
            "Reviewer rounds per reviewed PR",
            f"{review['reviewer_rounds'] / reviewed:.1f}" if reviewed else "n/a",
        ),
        ("Approved in the first round", pct(review["approved_first_round"], reviewed)),
        ("PRs written by a human", t["human_prs"]),
        (
            "Agent PRs where the maintainer pushed commits",
            (
                f"{review.get('agent_prs_with_human_commits', 0)} "
                f"({review.get('human_commits_on_agent_prs', 0)} commits)"
            ),
        ),
    ]
    mutation = t.get("mutation")
    if mutation and mutation.get("score") is not None:
        headline.append(
            (
                f"Mutation score (nightly, {mutation['run_at'][:10]})",
                f"{mutation['score']}% of {mutation['total']} mutants",
            )
        )
    out = [
        (
            f"_Generated {stats['generated_at']} from {t['merged_prs']} merged pull "
            f"requests ({t['recorded_prs']} with a provenance record)._"
        ),
        "",
        "| | |",
        "|---|---:|",
        *(f"| {label} | {value} |" for label, value in headline),
        "",
        "### Per pull request",
        "",
        (
            "| PR | AI | AI, edited | Human | Human commits | Messages | Active min "
            "| Reviewer rounds |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in reversed(stats["prs"]):
        p = row["provenance"]
        if p is None:
            note = "human-written" if row["author"] == "human" else "not recorded"
            cells = [note, "", "", "", "", ""]
        else:
            n = {c: sum(p["lines"][cat][c] for cat in CATEGORIES) for c in CLASSES}
            pt = p["totals"]
            cells = [n["ai"], n["ai_edited"], n["human"]]
            cells.append(p.get("commits", {}).get("human", 0))
            cells.append(pt["human_messages"])
            cells.append(f"{pt['active_seconds'] / 60:.0f}")
        rounds = ", ".join(str(v) for v in row["reviewer_rounds"]) or "none"
        out.append(" | ".join([f"| #{row['number']}", *map(str, cells), f"{rounds} |"]))
    return "\n".join(out) + "\n"


def load_stats(source: str) -> dict[str, Any] | None:
    """Load stats JSON from a path or URL; return None if it isn't available."""
    try:
        if source.startswith("https://"):
            with urllib.request.urlopen(source, timeout=30) as response:
                return json.loads(response.read())
        return json.loads(Path(source).read_text())
    except (OSError, ValueError) as error:
        print(f"ai_provenance: no stats from {source}: {error}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect", help="compute and save PRs' provenance comments")
    p.add_argument("prs", nargs="+", type=int)
    p.add_argument("--dry-run", action="store_true", help="print instead of posting")
    sub.add_parser("session-end", help="SessionEnd hook (payload on stdin)")
    p = sub.add_parser("collect-session", help="collect the PRs one session touched")
    p.add_argument("transcript", type=Path)
    p = sub.add_parser("aggregate", help="write the stats JSON (CI)")
    p.add_argument("out", type=Path)
    p.add_argument("--mutation", type=Path, help="mutmut export-cicd-stats JSON")
    p.add_argument("--previous", help="previous stats JSON (path or URL)")
    p = sub.add_parser("render", help="render stats JSON as Markdown (docs)")
    p.add_argument("source", help="stats JSON path or URL")
    p.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    if args.command == "collect":
        for number in args.prs:
            collect(number, Path.cwd(), dry_run=args.dry_run)
    elif args.command == "session-end":
        try:
            session_end(json.load(sys.stdin))
        except Exception as error:  # noqa: BLE001  a hook must never block exiting
            print(f"ai_provenance session-end: {error}", file=sys.stderr)
    elif args.command == "collect-session":
        collect_session(args.transcript, Path.cwd())
    elif args.command == "aggregate":
        if args.mutation and args.mutation.exists():
            mutation = mutation_summary(args.mutation)
        else:
            previous = load_stats(args.previous) if args.previous else None
            mutation = (previous or {}).get("totals", {}).get("mutation")
        stats = aggregate(mutation, Path.cwd())
        args.out.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    elif args.command == "render":
        args.out.write_text(render(load_stats(args.source)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
