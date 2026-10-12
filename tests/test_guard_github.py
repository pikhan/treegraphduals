"""
Test suite for the ``.claude/hooks/guard_github.py`` PreToolUse hook.

Each case runs the hook as a subprocess with a Claude Code hook payload on stdin
and checks whether it denies the command. The table started as the scratch check
used while the hook was built (#56, #71). Any future false positive or bypass
gets a row here first, then the fix.

Run with:
    pytest tests/test_guard_github.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard_github.py"

# Placeholders for the working directory of each case, filled in by the fixture:
# REPO is a throwaway repository on a feature branch, MAINWT a worktree of it on main.
REPO = "REPO"
MAINWT = "MAINWT"

CASES = [
    ("DENY", "git push origin main", REPO),
    ("DENY", "git push origin HEAD:main", REPO),
    ("DENY", "git push origin :main", REPO),
    ("DENY", "git push origin +issue-4:issue-4", REPO),
    ("DENY", "git push --force origin issue-4", REPO),
    ("DENY", "git push --force-with-lease", REPO),
    ("DENY", "git push --all origin", REPO),
    ("DENY", "git push -f", REPO),
    ("DENY", "git push", MAINWT),
    ("DENY", "git push origin HEAD", MAINWT),
    ("DENY", "cd {MAINWT} && git push", REPO),
    ("DENY", "git -C {MAINWT} push", REPO),
    ("DENY", "git push origin refs/heads/master", REPO),
    ("DENY", "git push --delete origin main", REPO),
    ("DENY", "bash -c 'git push origin main'", REPO),
    ("DENY", "eval git push origin main", REPO),
    ("DENY", "echo $(git push origin main)", REPO),
    ("DENY", "FOO=1 git push origin main", REPO),
    ("DENY", "uv run pytest && git push origin main", REPO),
    ("DENY", "gh pr merge 57 --squash", REPO),
    ("DENY", "gh pr merge --auto --squash", REPO),
    ("DENY", "gh repo edit --delete-branch-on-merge", REPO),
    (
        "DENY",
        "gh api -X PUT repos/pikhan/treegraphduals/branches/main/protection --input -",
        REPO,
    ),
    (
        "DENY",
        "gh api repos/pikhan/treegraphduals -X PATCH -F delete_branch_on_merge=true",
        REPO,
    ),
    (
        "DENY",
        "gh api -H 'Accept: application/json' repos/pikhan/treegraphduals -XPATCH -f x=y",
        REPO,
    ),
    (
        "DENY",
        "gh api --method DELETE repos/pikhan/treegraphduals/branches/main/protection",
        REPO,
    ),
    ("DENY", "gh api repos/pikhan/treegraphduals/rulesets -f name=x", REPO),
    ("DENY", "gh api -X PUT repos/pikhan/treegraphduals/pulls/57/merge", REPO),
    (
        "DENY",
        (
            "gh api graphql -f query='mutation { mergePullRequest("
            'input:{pullRequestId:"x"}) { clientMutationId } }\''
        ),
        REPO,
    ),
    ("allow", "git push -u origin issue-11-dfs-cache", REPO),
    ("allow", "git push origin HEAD", REPO),
    ("allow", "git push", REPO),
    ("allow", "git push -d origin issue-11-dfs-cache", REPO),
    ("allow", "git log main..HEAD && git push -u origin setup-agent-workflow", REPO),
    ("allow", "git push origin issue-4-main-fix", REPO),
    ("allow", "git status", REPO),
    ("allow", "git commit -m 'push to main later'", REPO),
    ("allow", "gh pr create --draft --fill", REPO),
    ("allow", "gh pr ready 57", REPO),
    ("allow", "gh pr comment 57 --body x", REPO),
    ("allow", "gh api repos/pikhan/treegraphduals/branches/main/protection", REPO),
    ("allow", "gh api repos/pikhan/treegraphduals/issues/5/comments -f body=hi", REPO),
    ("allow", "gh api graphql -f query='query { viewer { login } }'", REPO),
    ("allow", "gh issue edit 36 --add-blocked-by 4", REPO),
    ("allow", "uv run pytest -q", REPO),
    (
        "allow",
        (
            "cat > body.md <<'EOF'\n- denies `gh pr merge` and `git push origin main`\n"
            "gh pr merge 5\nEOF\ngh pr create --body-file body.md"
        ),
        REPO,
    ),
    ("allow", "gh pr comment 5 --body 'run `gh pr merge` later'", REPO),
    (
        "allow",
        "gh pr create --title x --body 'line one\ngit push origin main\nline three'",
        REPO,
    ),
    ("allow", "uv run pytest 2>&1 | tail -3 && git push -u origin issue-4-x", REPO),
    ("DENY", 'gh pr create --body "$(git push origin main)"', REPO),
    ("DENY", "cat <<EOF\n$(git push origin main)\nEOF", REPO),
    ("DENY", "cat <<'EOF'\nharmless\nEOF\ngit push origin main", REPO),
    ("DENY", "git push origin main 2>&1 | tail -1", REPO),
    ("DENY", "echo ok; gh pr merge 3", REPO),
    ("DENY", "sleep 1 & gh pr merge 3", REPO),
]


def clean_env() -> dict[str, str]:
    """
    Return an environment that isolates git from the developer's setup.

    Drops ``GIT_*`` variables (set, for example, inside git hooks) and ignores the
    global and system git config, so signing or hooks there can't interfere.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def git(*args: str, cwd: Path) -> None:
    """Run a git command for test setup, failing loudly on error."""
    subprocess.run(
        ["git", *args], cwd=cwd, env=clean_env(), check=True, capture_output=True
    )


@pytest.fixture(scope="module")
def repos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """
    Create a throwaway repository on a feature branch and a worktree of it on main.

    Returns a mapping from the ``REPO``/``MAINWT`` placeholders to their paths.
    """
    root = tmp_path_factory.mktemp("guard")
    repo = root / "repo"
    mainwt = root / "wt-main"
    repo.mkdir()
    git("init", "-q", "-b", "setup-agent-workflow", cwd=repo)
    git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "init",
        cwd=repo,
    )
    git("worktree", "add", "-q", "-b", "main", str(mainwt), cwd=repo)
    return {REPO: str(repo), MAINWT: str(mainwt)}


def run_hook(
    command: str, cwd: str, tool: str = "Bash"
) -> subprocess.CompletedProcess[str]:
    """Run the hook with a PreToolUse payload for ``command`` on stdin."""
    payload = {"tool_name": tool, "tool_input": {"command": command}, "cwd": cwd}
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        cwd=cwd,
        env=clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def verdict(result: subprocess.CompletedProcess[str]) -> str:
    """Classify a hook run as ``DENY``, ``allow`` or ``ERR <stderr>``."""
    assert result.returncode == 0, result.stderr
    if '"deny"' in result.stdout:
        return "DENY"
    return "ERR " + result.stderr if result.stderr else "allow"


@pytest.mark.parametrize(
    ("want", "command", "where"),
    CASES,
    ids=[f"{want} in {where}: {command}" for want, command, where in CASES],
)
def test_guard_verdict(
    repos: dict[str, str], want: str, command: str, where: str
) -> None:
    """The hook denies or allows each command as the table says."""
    command = command.replace("{MAINWT}", repos[MAINWT])
    assert verdict(run_hook(command, repos[where])) == want


def test_non_bash_tool_is_allowed(repos: dict[str, str]) -> None:
    """Only Bash calls are checked; other tools pass through."""
    result = run_hook("git push origin main", repos[REPO], tool="Edit")
    assert verdict(result) == "allow"


def test_malformed_json_exits_zero() -> None:
    """Malformed input is allowed (exit 0) rather than blocking every command."""
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input="not json",
        env=clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
