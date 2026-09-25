#!/usr/bin/env python3
"""
PreToolUse hook: keep changes to ``main`` and to repository settings with the maintainer.

Claude Code runs this before every Bash tool call made by Claude or by any agent
working in this repository. The maintainer's own terminal is not affected. It
denies:

- pushing to ``main``/``master`` (explicit refspec, or a bare push from that branch),
  force-pushing, and ``--all`` / ``--mirror`` / ``--prune`` pushes;
- merging pull requests (``gh pr merge``, or the REST/GraphQL merge endpoints);
- changing repository settings, branch protection or rulesets (``gh repo
  edit/delete/rename/archive``, and ``gh api`` writes to those endpoints).

Pushing feature branches, opening and commenting on PRs, and reading anything
stay allowed. If this script fails unexpectedly it allows the call (exit 0 with
no output) rather than blocking every command.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from typing import NoReturn

PROTECTED_BRANCHES = {"main", "master"}
SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|\n]")
GIT_OPTIONS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
FORCE_FLAGS = {"-f", "--force", "--mirror", "--all", "--prune"}
REPO_SETTINGS_COMMANDS = {"edit", "delete", "rename", "archive"}
API_OPTIONS_WITH_VALUE = {
    "-H",
    "--header",
    "-q",
    "--jq",
    "-t",
    "--template",
    "--hostname",
    "-p",
    "--preview",
    "--cache",
}
SETTINGS_ENDPOINT = re.compile(
    r"branches/[^/]+/protection|rulesets|/pulls/\d+/merge|git/refs/heads/(main|master)"
    r"|/(hooks|collaborators|keys|environments|actions/permissions|pages)\b"
)
REPO_ROOT_ENDPOINT = re.compile(r"^/?repos/[^/]+/[^/]+/?$")
GRAPHQL_MUTATION = re.compile(
    r"(?i)mutation[\s\S]*(protection|ruleset|mergePullRequest|updateRepository|"
    r"deleteRepository|archiveRepository)"
)


def deny(reason: str) -> NoReturn:
    """Print a PreToolUse deny decision and exit."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Blocked by .claude/hooks/guard_github.py: {reason} "
                        "Changes to main and to repository settings go through the "
                        "maintainer. Ask them instead of working around this."
                    ),
                }
            }
        )
    )
    sys.exit(0)


def current_branch(cwd: str) -> str:
    """Return the branch checked out in ``cwd`` (empty string if unknown)."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def strip_ref(ref: str) -> str:
    """Reduce a push destination like ``refs/heads/main`` to ``main``."""
    return ref.removeprefix("refs/heads/")


def check_git(args: list[str], cwd: str) -> None:
    """Deny ``git push`` calls that would touch a protected branch or force-push."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] in GIT_OPTIONS_WITH_VALUE:
            if args[i] == "-C" and i + 1 < len(args):
                cwd = args[i + 1]
            i += 2
        else:
            i += 1
    if i >= len(args) or args[i] != "push":
        return
    push_args = args[i + 1 :]
    flags = [a for a in push_args if a.startswith("-")]
    positional = [a for a in push_args if not a.startswith("-")]
    for flag in flags:
        if flag in FORCE_FLAGS or flag.startswith("--force"):
            deny(f"'git push {flag}' is not allowed.")
    refspecs = positional[1:]
    if not refspecs:
        if current_branch(cwd) in PROTECTED_BRANCHES:
            deny("a bare 'git push' from main/master would push to main.")
        return
    for spec in refspecs:
        if spec.startswith("+"):
            deny(f"force refspec '{spec}' is not allowed.")
        source, _, destination = spec.partition(":")
        target = strip_ref(destination or source)
        if target == "HEAD":
            target = current_branch(cwd)
        if target in PROTECTED_BRANCHES:
            deny(f"pushing to '{target}' is not allowed.")


def check_gh(args: list[str], raw: str) -> None:
    """Deny ``gh`` calls that merge PRs or change repository settings."""
    if args[:2] == ["pr", "merge"]:
        deny("merging pull requests is the maintainer's job.")
    if args[:1] == ["repo"] and len(args) > 1 and args[1] in REPO_SETTINGS_COMMANDS:
        deny(f"'gh repo {args[1]}' changes repository settings.")
    if args[:1] != ["api"]:
        return
    method = "GET"
    endpoint = ""
    i = 1
    while i < len(args):
        arg = args[i]
        if arg in {"-X", "--method"} and i + 1 < len(args):
            method = args[i + 1].upper()
            i += 2
            continue
        if arg in API_OPTIONS_WITH_VALUE:
            i += 2
            continue
        if arg.startswith("--method="):
            method = arg.split("=", 1)[1].upper()
        elif arg.startswith("-X") and len(arg) > 2:
            method = arg[2:].upper()
        elif arg in {"-f", "-F", "--field", "--raw-field", "--input"}:
            if method == "GET":
                method = "POST"
            i += 2
            continue
        elif not arg.startswith("-") and not endpoint:
            endpoint = arg
        i += 1
    if endpoint == "graphql":
        if GRAPHQL_MUTATION.search(raw):
            deny("GraphQL mutations on merges, protection or repository settings.")
        return
    if method == "GET":
        return
    if SETTINGS_ENDPOINT.search(endpoint) or REPO_ROOT_ENDPOINT.search(endpoint):
        deny(f"'gh api -X {method} {endpoint}' changes a merge or repository setting.")


def check_segment(segment: str, cwd: str) -> None:
    """Check one simple command (no ``&&``, ``;`` or pipes)."""
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens = tokens[1:]  # leading VAR=value assignments
    while tokens and tokens[0] in {"command", "exec", "time", "nohup", "sudo"}:
        tokens = tokens[1:]
    if not tokens:
        return
    name = tokens[0].rsplit("/", 1)[-1]
    if name == "eval":
        check_command(" ".join(tokens[1:]), cwd)
        return
    if name in {"bash", "sh", "zsh"} and "-c" in tokens:
        idx = tokens.index("-c")
        if idx + 1 < len(tokens):
            check_command(tokens[idx + 1], cwd)
        return
    if name == "git":
        check_git(tokens[1:], cwd)
    elif name == "gh":
        check_gh(tokens[1:], segment)


def check_command(command: str, cwd: str) -> None:
    """Check every simple command inside a (possibly compound) shell command."""
    for inner in re.findall(r"\$\(([^()]*)\)|`([^`]*)`", command):
        check_command(inner[0] or inner[1], cwd)
    for segment in SEGMENT_SPLIT.split(command):
        cd = re.match(r"\s*cd\s+(\S+)\s*$", segment)
        if cd:
            cwd = os.path.join(cwd, os.path.expanduser(cd.group(1)))
            continue
        check_segment(segment, cwd)


def main() -> None:
    """Read the hook input from stdin and deny guarded commands."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    if payload.get("tool_name") != "Bash":
        return
    command = payload.get("tool_input", {}).get("command", "")
    check_command(command, payload.get("cwd", "."))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - never block every command on a bug
        print(f"guard_github.py error (allowing): {exc}", file=sys.stderr)
