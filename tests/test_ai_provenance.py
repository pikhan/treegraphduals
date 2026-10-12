"""
Test suite for ``scripts/ai_provenance.py`` (#81, ADR 0006).

The transcripts here are built by hand in the format Claude Code writes
(``~/.claude/projects/<dir>/<session>.jsonl``), and the git history in a
throwaway repository, so every expected count below is worked out from the
fixture itself, not taken from a run of the script.

Run with:
    pytest tests/test_ai_provenance.py
"""

import doctest
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ai_provenance.py"
if not SCRIPT.exists():  # the sdist ships tests/ but not scripts/
    pytest.skip("scripts/ai_provenance.py is not available", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("ai_provenance", SCRIPT)
assert _spec is not None and _spec.loader is not None
ap = importlib.util.module_from_spec(_spec)
sys.modules["ai_provenance"] = ap  # dataclasses look their module up here
_spec.loader.exec_module(ap)

T0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


def at(seconds: float) -> str:
    """Timestamp ``seconds`` after T0, in the transcript's format."""
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def human(seconds: float, text: str, branch: str = "main") -> dict:
    """Build a human-typed message record."""
    return {
        "type": "user",
        "timestamp": at(seconds),
        "gitBranch": branch,
        "origin": {"kind": "human"},
        "message": {"role": "user", "content": text},
    }


def assistant(
    seconds: float, msg_id: str, blocks: list, branch: str = "main", **usage
) -> dict:
    """Build an assistant response record; ``usage`` takes ``inp`` and ``out`` tokens."""
    return {
        "type": "assistant",
        "timestamp": at(seconds),
        "gitBranch": branch,
        "effort": "high",
        "message": {
            "id": msg_id,
            "role": "assistant",
            "model": "claude-test",
            "content": blocks,
            "usage": {
                "input_tokens": usage.get("inp", 0),
                "output_tokens": usage.get("out", 0),
            },
        },
    }


def tool_result(
    seconds: float, tool_id: str, result: dict, branch: str = "main"
) -> dict:
    """Build a tool result record carrying ``toolUseResult``."""
    return {
        "type": "user",
        "timestamp": at(seconds),
        "gitBranch": branch,
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": "x"}
            ],
        },
        "toolUseResult": result,
    }


def turn(seconds: float, duration_ms: int, branch: str = "main") -> dict:
    """Build a turn_duration record."""
    return {
        "type": "system",
        "subtype": "turn_duration",
        "timestamp": at(seconds),
        "gitBranch": branch,
        "durationMs": duration_ms,
    }


def write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write records as a JSONL transcript and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


ASK = {
    "questions": [
        {
            "question": "Q1?",
            "options": [{"label": "Yes (Recommended)"}, {"label": "No"}],
        },
        {"question": "Q2?", "options": [{"label": "A"}, {"label": "B"}]},
    ],
    "answers": {"Q1?": "Yes (Recommended)", "Q2?": "neither, use C instead"},
    "annotations": {"Q1?": {"notes": "keep it short"}},
}

# One session, worked out by hand:
#   t=0     prompt "/work-issue 7" (2 words, 13 chars); first prompt, active 0
#   t=10    assistant asks a question (AskUserQuestion, id q1)
#   t=1210  the answer: 1200 s later, capped at 900; typed words: 4 ("neither,
#           use C instead") + 3 notes = 7
#   t=1220  assistant runs `git commit -q` (Bash b1), result at t=1225
#   t=1230  turn_duration 1230000 ms (prompt 1)
#   t=1290  prompt "please also add docs" (4 words): 60 s after the turn ended
#   t=1300  assistant response m3, split over two records (counted once)
#   t=1310  turn_duration 20000 ms, then the reviewer hands back APPROVE
#   t=1400  turn_duration 110000 ms for the same prompt, covering [1290, 1400],
#           which contains [1290, 1310]: agent time is 1230 + 110 s
#   t=9000  prompt "thanks" (1 word), gap 7600 s capped at 900; after approval
#   t=9001  "[Request interrupted by user]" is not counted
SESSION = [
    {"type": "pr-link", "prNumber": 7, "timestamp": at(1226)},
    human(
        0, "<command-name>/work-issue</command-name>\n<command-args>7</command-args>"
    ),
    assistant(
        10,
        "m1",
        [{"type": "tool_use", "id": "q1", "name": "AskUserQuestion", "input": {}}],
        inp=100,
        out=10,
    ),
    tool_result(1210, "q1", ASK),
    assistant(
        1220,
        "m2",
        [
            {
                "type": "tool_use",
                "id": "b1",
                "name": "Bash",
                "input": {"command": "git commit -qm x"},
            }
        ],
        inp=200,
        out=20,
    ),
    tool_result(1225, "b1", {"stdout": "", "stderr": ""}),
    turn(1230, 1_230_000),
    human(1290, "please also add docs", branch="issue-7-x"),
    assistant(
        1300,
        "m3",
        [{"type": "thinking", "thinking": ""}],
        branch="issue-7-x",
        inp=300,
        out=30,
    ),
    assistant(
        1300,
        "m3",
        [{"type": "text", "text": "ok"}],
        branch="issue-7-x",
        inp=300,
        out=30,
    ),
    turn(1310, 20_000, branch="issue-7-x"),
    {
        "type": "user",
        "timestamp": at(1320),
        "gitBranch": "issue-7-x",
        "isMeta": True,
        "origin": {
            "kind": "peer",
            "name": "reviewer",
            "body": "## Verdict: APPROVE\n...",
        },
        "message": {"role": "user", "content": "Another Claude session sent a message"},
    },
    turn(1400, 110_000, branch="issue-7-x"),
    human(9000, "thanks", branch="issue-7-x"),
    human(9001, "[Request interrupted by user]", branch="issue-7-x"),
]

# The session's reviewer subagent ran one Bash call that printed a commit summary.
SUBAGENT = [
    assistant(
        1500,
        "s1",
        [
            {
                "type": "tool_use",
                "id": "b2",
                "name": "Bash",
                "input": {"command": "git commit -m y"},
            }
        ],
        branch="issue-7-x",
        inp=50,
        out=5,
    ),
    tool_result(
        1504,
        "b2",
        {"stdout": "[issue-7-x 1a2b3c4] y\n 1 file changed"},
        branch="issue-7-x",
    ),
]


@pytest.fixture
def session(tmp_path: Path):
    """Return the hand-built session, parsed with its subagent transcript."""
    main = write_jsonl(tmp_path / "sess-1.jsonl", SESSION)
    sub = write_jsonl(tmp_path / "sess-1" / "subagents" / "agent-1.jsonl", SUBAGENT)
    return ap.parse_session([main], [sub])


def test_session_links_and_commit_evidence(session):
    """PR links, commit SHAs and Bash windows come out of both transcripts."""
    assert session.session_id == "sess-1"
    assert session.pr_links == [(T0 + timedelta(seconds=1226), 7)]
    assert session.commits == {"1a2b3c4"}
    assert session.bash_windows == [
        (T0 + timedelta(seconds=1220), T0 + timedelta(seconds=1225)),
        (T0 + timedelta(seconds=1500), T0 + timedelta(seconds=1504)),
    ]


def test_session_summary_matches_hand_count(session):
    """Every steering and agent number matches the hand count above SESSION."""
    summary = ap.summarize(session.events)
    assert summary["human_messages"] == 3
    assert summary["human_words"] == 2 + 4 + 1 + 7
    assert summary["human_chars"] == len("/work-issue 7") + len(
        "please also add docs"
    ) + len("thanks")
    assert summary["answers"] == 1
    assert summary["active_seconds"] == 0 + 900 + 60 + 900
    assert summary["agent_seconds"] == 1230 + 110
    assert summary["messages_after_approval"] == 1
    assert summary["reviewer_handbacks"] == ["APPROVE"]
    # First to last record: the interrupt at t=9001 is a record, though not a prompt.
    assert summary["wall_clock_seconds"] == 9001
    assert summary["models"] == {
        "claude-test": {"responses": 4, "input_tokens": 650, "output_tokens": 65}
    }
    assert summary["efforts"] == ["high"]


def test_session_contribution_splits_by_pr(session):
    """A session counts only for the PRs it linked or worked on."""
    # Everything precedes or follows the only link (#7), so all of it is #7's.
    whole = ap.session_contribution(session, 7, "issue-7-x", {"issue-7-x": 7}, False)
    assert whole is not None and whole["human_messages"] == 3
    # A session that never mentions #8 contributes nothing to it.
    assert (
        ap.session_contribution(session, 8, "issue-8-y", {"issue-8-y": 8}, False)
        is None
    )
    # Named in #8's PR body: the work on #7's branch still counts for #7, and the
    # first prompt (on main, before #7 was linked) counts for #7 too.
    assert (
        ap.session_contribution(session, 8, "issue-8-y", {"issue-7-x": 7}, True) is None
    )


def test_separate_turns_within_a_prompt_add_up(tmp_path: Path):
    """Per-turn durations in one prompt are summed, not maxed (seen in real data)."""
    # One prompt, then turns of 13 s, 7 s and 3 s ending at 13, 190 and 249 s:
    # the intervals [0, 13], [183, 190] and [246, 249] are disjoint, total 23 s.
    records = [
        human(0, "go"),
        turn(13, 13_000),
        turn(190, 7_000),
        turn(249, 3_000),
    ]
    session = ap.parse_session([write_jsonl(tmp_path / "s.jsonl", records)])
    assert ap.summarize(session.events)["agent_seconds"] == 23


def test_messages_before_any_approval_are_not_post_approval(tmp_path: Path):
    """A CHANGES REQUESTED hand-back doesn't start the post-approval count."""
    records = [
        human(0, "go"),
        {
            "type": "user",
            "timestamp": at(50),
            "origin": {
                "kind": "peer",
                "name": "reviewer",
                "body": "**Verdict: CHANGES REQUESTED**",
            },
            "message": {"role": "user", "content": "x"},
        },
        human(100, "fix it"),
    ]
    summary = ap.summarize(
        ap.parse_session([write_jsonl(tmp_path / "s.jsonl", records)]).events
    )
    assert summary["messages_after_approval"] == 0
    assert summary["reviewer_handbacks"] == ["CHANGES REQUESTED"]


@given(
    gaps=st.lists(st.integers(min_value=0, max_value=5 * 3600), min_size=1, max_size=8)
)
def test_active_time_is_each_gap_capped(tmp_path_factory, gaps):
    """Active time is each gap since Claude's last output, capped (ADR 0006)."""
    # Definition (ADR 0006): a prompt's active time is the time since Claude's
    # previous output, capped at 15 minutes; the first prompt has none.
    records, clock = [], 0
    for i, gap in enumerate(gaps):
        clock += gap
        records.append(human(clock, f"prompt {i}"))
        clock += 1
        records.append(assistant(clock, f"m{i}", [{"type": "text", "text": "ok"}]))
    path = write_jsonl(tmp_path_factory.mktemp("h") / "s.jsonl", records)
    summary = ap.summarize(ap.parse_session([path]).events)
    assert summary["active_seconds"] == sum(min(g, 900) for g in gaps[1:])


def test_typed_answer_words_counts_only_typing():
    """Clicking an offered option counts no words; custom answers and notes do."""
    assert ap.typed_answer_words(ASK) == 7
    clicked = {**ASK, "answers": {"Q1?": "No", "Q2?": "A"}, "annotations": {}}
    assert ap.typed_answer_words(clicked) == 0
    # A label containing ", " is still one click (seen in a real transcript).
    comma = {
        "questions": [
            {"question": "Q?", "options": [{"label": "Active time, 15-min cap"}]}
        ],
        "answers": {"Q?": "Active time, 15-min cap"},
    }
    assert ap.typed_answer_words(comma) == 0
    # A multi-select answer joins two clicked labels with ", ".
    multi = {**ASK, "answers": {"Q2?": "A, B"}, "annotations": {}}
    assert ap.typed_answer_words(multi) == 0


def test_find_sessions_merges_moved_transcripts(tmp_path: Path):
    """One session's transcripts in a worktree's and the main project dir merge."""
    repo = Path("/work/repo")
    main_dir = tmp_path / "-work-repo"
    wt_dir = tmp_path / "-work-repo--claude-worktrees-issue-7"
    write_jsonl(main_dir / "abc.jsonl", [human(0, "a")])
    write_jsonl(wt_dir / "abc.jsonl", [human(5, "b")])
    write_jsonl(main_dir / "abc" / "subagents" / "agent-1.jsonl", [])
    write_jsonl(tmp_path / "-work-other" / "zzz.jsonl", [human(0, "c")])
    write_jsonl(tmp_path / "-work-repo-old" / "yyy.jsonl", [human(0, "d")])
    found = ap.find_sessions(tmp_path, repo)
    assert list(found) == ["abc"]
    mains, subs = found["abc"]
    assert sorted(mains) == sorted([main_dir / "abc.jsonl", wt_dir / "abc.jsonl"])
    assert subs == [main_dir / "abc" / "subagents" / "agent-1.jsonl"]


# ---------------------------------------------------------------------------
# Line authorship, in a throwaway repository
# ---------------------------------------------------------------------------


def run_git(repo: Path, *args: str, when: datetime | None = None) -> str:
    """Run git in ``repo`` with a fixed identity and, optionally, a fixed date."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    if when is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when.isoformat()
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit_files(repo: Path, files: dict[str, str], when: datetime) -> str:
    """Write ``files`` and commit them at ``when``; return the new SHA."""
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "c", when=when)
    return run_git(repo, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path: Path):
    r"""Build base -> AI commit -> human commit, with hand-countable lines.

    base:   src/m.py = "old\n"
    AI:     src/m.py = "old\na1\na2\n\na3\n"   (3 non-blank lines added, 1 blank)
            tests/t.py = "t1\nt2\n"              (2 lines)
    human:  src/m.py: a2 -> "a2 changed" (rewrites an AI line), adds "h1" at the end
            docs/d.md = "d1\n"                     (1 line)

    Final added non-blank lines: src a1, a3 (ai); "a2 changed" (ai_edited);
    h1 (human); tests t1, t2 (ai); docs d1 (human).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-q", "-b", "main")
    base = commit_files(repo, {"src/m.py": "old\n"}, T0)
    ai = commit_files(
        repo,
        {"src/m.py": "old\na1\na2\n\na3\n", "tests/t.py": "t1\nt2\n"},
        T0 + timedelta(hours=1),
    )
    hum = commit_files(
        repo,
        {"src/m.py": "old\na1\na2 changed\n\na3\nh1\n", "docs/d.md": "d1\n"},
        T0 + timedelta(hours=2),
    )
    return repo, base, ai, hum


def test_commits_classified_by_agent_bash_windows(history):
    """A commit authored during an agent's Bash call is AI; others are human."""
    repo, base, ai, hum = history
    window = (T0 + timedelta(hours=1, seconds=-3), T0 + timedelta(hours=1, seconds=3))
    classes = ap.classify_commits(base, hum, set(), [window], {}, repo)
    assert classes == {ai: "ai", hum: "human"}
    # A summary-line SHA is enough on its own, and an earlier record is kept.
    assert ap.classify_commits(base, hum, {ai[:7]}, [], {}, repo)[ai] == "ai"
    assert ap.classify_commits(base, hum, set(), [], {ai: "ai"}, repo)[ai] == "ai"


def test_line_counts_match_hand_count(history):
    """Line classes match the hand count in the ``history`` docstring."""
    repo, base, ai, hum = history
    counts = ap.count_lines(base, hum, {ai: "ai", hum: "human"}, repo)
    assert counts == {
        "src": {"ai": 2, "ai_edited": 1, "human": 1},
        "tests": {"ai": 2, "ai_edited": 0, "human": 0},
        "docs": {"ai": 0, "ai_edited": 0, "human": 1},
        "other": {"ai": 0, "ai_edited": 0, "human": 0},
    }


def test_ai_edited_detection_handles_spaces_in_paths(tmp_path: Path):
    """A path with a space (git ends its diff header with a tab) still works."""
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-q", "-b", "main")
    base = commit_files(repo, {"docs/my notes.md": "x\n"}, T0)
    ai = commit_files(
        repo, {"docs/my notes.md": "x\nai line\n"}, T0 + timedelta(hours=1)
    )
    hum = commit_files(
        repo, {"docs/my notes.md": "x\nai line, fixed\n"}, T0 + timedelta(hours=2)
    )
    counts = ap.count_lines(base, hum, {ai: "ai", hum: "human"}, repo)
    assert counts["docs"] == {"ai": 0, "ai_edited": 1, "human": 0}


def test_all_human_history_has_no_ai_lines(history):
    """With no AI commits, every added line is human."""
    repo, base, ai, hum = history
    counts = ap.count_lines(base, hum, {ai: "human", hum: "human"}, repo)
    assert sum(c["ai"] + c["ai_edited"] for c in counts.values()) == 0
    assert counts["src"]["human"] == 4


# ---------------------------------------------------------------------------
# Comments, aggregation and rendering
# ---------------------------------------------------------------------------


def make_record(
    lines_src_ai: int = 3, human_messages: int = 2, commits: dict | None = None
) -> dict:
    """Build a minimal provenance record."""
    lines = {cat: {"ai": 0, "ai_edited": 0, "human": 0} for cat in ap.CATEGORIES}
    lines["src"]["ai"] = lines_src_ai
    lines["tests"]["human"] = 1
    return {
        "schema": 1,
        "pr": 7,
        "head_sha": "h" * 40,
        "sessions": {"s": {}},
        "commits": commits if commits is not None else {},
        "lines": lines,
        "totals": ap.totals(
            {
                "s": {
                    "human_messages": human_messages,
                    "human_words": 10,
                    "active_seconds": 120,
                    "agent_seconds": 600,
                    "models": {
                        "claude-test": {
                            "responses": 1,
                            "input_tokens": 5,
                            "output_tokens": 2,
                        }
                    },
                    "reviewer_handbacks": ["APPROVE"],
                }
            }
        ),
    }


def owner(body: str) -> dict:
    """Build a comment by the repository owner, as ``gh pr list --json`` has it."""
    return {"body": body, "authorAssociation": "OWNER"}


def test_comment_round_trips_its_record():
    """The comment embeds its record losslessly, behind the marker."""
    record = make_record()
    body = ap.comment_body(record)
    assert body.startswith(ap.MARKER)
    assert ap.provenance_from_body(body) == record
    assert ap.provenance_from_body("## Automated review (round 1)") is None


def test_malformed_records_are_ignored():
    """A broken or wrongly shaped record is skipped, not a crash."""
    assert ap.provenance_from_body(ap.MARKER + "\n```json\n{not json\n```") is None
    assert ap.provenance_from_body(ap.MARKER + '\n```json\n{"schema": 1}\n```') is None
    bad = make_record()
    bad["lines"]["src"]["human"] = "lots"
    body = f"{ap.MARKER}\n```json\n{json.dumps(bad)}\n```"
    assert ap.provenance_from_body(body) is None
    # Empty or incomplete totals would make render() fail, so they are rejected.
    for totals in (
        {},
        {k: v for k, v in make_record()["totals"].items() if k != "answers"},
    ):
        record = {**make_record(), "totals": totals}
        body = f"{ap.MARKER}\n```json\n{json.dumps(record)}\n```"
        assert ap.provenance_from_body(body) is None


def test_only_the_owners_comments_count():
    """Another account's provenance record or review comment is ignored."""
    forged = make_record(lines_src_ai=0)
    forged["sessions"] = {}
    forged["lines"]["src"]["human"] = 1_000_000
    real = make_record()
    comments = [
        {"body": ap.comment_body(forged), "authorAssociation": "NONE"},
        {"body": "## Automated review (round 1)\n**Verdict: APPROVE**"},
        owner(ap.MARKER + "\n```json\n{not json\n```"),
        owner(ap.comment_body(real)),
    ]
    assert ap.provenance_comment(comments) == (comments[3], real)
    row = ap.pr_row(github_pr(7, comments), {}, set())
    assert row["author"] == "ai-assisted"
    assert row["reviewer_rounds"] == []
    assert row["provenance"]["lines"]["src"]["human"] == 0


def test_review_rounds_reads_each_round_once():
    """Only comments headed ``Automated review (round N)`` count as rounds."""
    comments = [
        owner("## Automated review (round 1)\n\n### Verdict: CHANGES REQUESTED\n"),
        owner("I disagree with the Automated review (round 1) item 2"),
        owner("## Automated review (round 2)\n\n**Verdict: APPROVE**\n"),
    ]
    assert ap.review_rounds(comments) == ["CHANGES REQUESTED", "APPROVE"]


def github_pr(
    number: int, comments: list[dict], body: str = "", head: str = "h" * 40
) -> dict:
    """Build a merged PR as ``gh pr list --json`` returns it."""
    return {
        "number": number,
        "title": f"PR {number}",
        "body": body,
        "createdAt": "2026-10-01T10:00:00Z",
        "mergedAt": "2026-10-01T13:30:00Z",
        "headRefOid": head,
        "reviews": [],
        "comments": comments,
        "closingIssuesReferences": [{"number": 40}],
    }


def test_pr_rows_and_totals():
    """Per-PR rows, totals and the rendered page match hand counts."""
    recorded = github_pr(
        7,
        [
            owner("## Automated review (round 1)\n**Verdict: CHANGES REQUESTED**"),
            owner("## Automated review (round 2)\n**Verdict: APPROVE**"),
            owner(
                ap.comment_body(
                    make_record(commits={"a" * 40: "ai", "b" * 40: "human"})
                )
            ),
        ],
    )
    by_human = github_pr(8, [], body="## AI use\n\nNone: written by a human.\n")
    stale = github_pr(
        9,
        [owner(ap.comment_body(make_record(1, 1, commits={"c" * 40: "ai"})))],
        head="x" * 40,
    )
    rows = [
        ap.pr_row(recorded, {"commits": 2, "review_comments": 0}, {40}),
        ap.pr_row(by_human, {"commits": 1, "review_comments": 2}, {40}),
        ap.pr_row(stale, {"commits": 3, "review_comments": 0}, set()),
    ]
    assert [r["author"] for r in rows] == ["ai-assisted", "human", "ai-assisted"]
    assert rows[0]["hours_open"] == 3.5
    assert [r["commits"] for r in rows] == [2, 1, 3]
    assert rows[0]["provenance"]["commits"] == {"ai": 1, "human": 1}
    assert rows[0]["reviewer_rounds"] == ["CHANGES REQUESTED", "APPROVE"]
    assert [r["stale"] for r in rows] == [False, False, True]
    assert rows[1]["closes_difficulty_max"] == [40]

    t = ap.aggregate_rows(rows, None)
    assert (t["merged_prs"], t["recorded_prs"], t["stale_prs"], t["human_prs"]) == (
        3,
        2,
        1,
        1,
    )
    assert t["lines"]["src"]["ai"] == 3 + 1
    assert t["lines"]["tests"]["human"] == 2
    assert t["steering"]["human_messages"] == 2 + 1
    assert t["review"] == {
        "prs_reviewed": 1,
        "reviewer_rounds": 2,
        "approved_first_round": 0,
        "formal_reviews": 0,
        "review_comments": 2,
        "human_commits_on_agent_prs": 1,
        "agent_prs_with_human_commits": 1,
    }
    assert t["difficulty_max_closed_by_human_prs"] == [40]

    stats = {"generated_at": "2026-10-02T00:00:00+00:00", "totals": t, "prs": rows}
    page = ap.render(stats)
    assert "| Added lines written by agents | 67% of 6 |" in page  # 4 AI of 4+2
    assert "| Agent PRs where the maintainer pushed commits | 1 (1 commit) |" in page
    assert "| #8 | human-written |  |  |  |  |  | none |" in page
    # PR 7: 3 AI lines, 0 edited, 1 human line, 1 human commit, 2 messages, 2 min.
    assert "| #7 | 3 | 0 | 1 | 1 | 2 | 2 | CHANGES REQUESTED, APPROVE |" in page


def test_script_doctests():
    """The script's own doctest examples pass."""
    # pytest's --doctest-modules covers src/ and tests/ only, not scripts/.
    result = doctest.testmod(ap, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0
    assert result.failed == 0


def test_render_without_stats():
    """Without published stats the page says so."""
    assert "No statistics" in ap.render(None)


def test_render_ends_headline_with_and_without_stats():
    """docs/ai_development.rst includes ai_stats.md up to the headline marker."""
    page = (ap.ROOT / "docs" / "ai_development.rst").read_text()
    assert f":end-before: {ap.HEADLINE_END}" in page
    assert ap.render(None).count(ap.HEADLINE_END) == 1

    by_human = github_pr(8, [], body="## AI use\n\nNone: written by a human.\n")
    rows = [ap.pr_row(by_human, {"commits": 1, "review_comments": 0}, set())]
    stats = {"generated_at": "2026-10-02T00:00:00+00:00", "prs": rows}
    stats["totals"] = ap.aggregate_rows(rows, None)
    headline, rest = ap.render(stats).split(ap.HEADLINE_END)
    assert "| Added lines written by agents |" in headline
    assert "### Per pull request" in rest


def test_mutation_summary(tmp_path: Path):
    """Timeouts and segfaults count as caught, skips don't count: 7 of 10 is 70%."""
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "killed": 5,
                "survived": 3,
                "timeout": 1,
                "segfault": 1,
                "skipped": 4,
                "total": 14,
            }
        )
    )
    m = ap.mutation_summary(path)
    assert (m["killed"], m["survived"], m["total"], m["score"]) == (5, 3, 14, 70.0)
