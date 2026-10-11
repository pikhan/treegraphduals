"""Keeps every copy of the supported-Python list in sync with pyproject.toml.

Sources of truth (ADR 0001):

- the floor is ``requires-python`` (``>=X.Y``), which spec0-action bumps;
- the full supported list is the ``Programming Language :: Python :: X.Y``
  classifiers. Classifiers below the floor are dropped, and the floor's
  classifier is added if missing. New versions are added by hand (ADR 0001,
  decision 5).

``.python-version`` (the local development interpreter) follows the newest
supported version instead, per ADR 0001 decision 7.

The CI workflows don't hold a copy: they ask this script at run time
(``--floor``, ``--ci-matrix``), so the bot's PR never has to edit a workflow.

Usage::

    python scripts/sync_python_versions.py            # rewrite the copies
    python scripts/sync_python_versions.py --check    # exit 1 if any is stale
    python scripts/sync_python_versions.py --floor    # print e.g. 3.13
    python scripts/sync_python_versions.py --ci-matrix 3.15  # CI test matrix

Only the standard library is used, so CI can run it without installing anything.
"""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CLASSIFIER = re.compile(
    r'^\s*"Programming Language :: Python :: (3\.\d+)",\n', flags=re.MULTILINE
)

# Files holding a version: (path, pattern whose group 1 is the version). Each
# pattern must match exactly once, so rewording a file can't silently drop it.
NEWEST_COPIES = [
    (".python-version", r"\A(3\.\d+)\n\Z"),
]
FLOOR_COPIES = [
    (".readthedocs.yaml", r'^    python: "(3\.\d+)"$'),
    ("README.md", r"^Requires Python (3\.\d+)\+\.$"),
    ("CLAUDE.md", r"^- Python (3\.\d+)\+, packaged with `uv`"),
]


def as_tuple(version: str) -> tuple[int, ...]:
    """Turn ``"3.13"`` into ``(3, 13)`` so versions compare numerically."""
    return tuple(int(part) for part in version.split("."))


def read_floor() -> str:
    """Return the ``X.Y`` floor from ``requires-python``."""
    spec = tomllib.loads(PYPROJECT.read_text())["project"]["requires-python"]
    match = re.fullmatch(r"\s*>=\s*(3\.\d+)\s*", spec)
    if match is None:
        sys.exit(f"requires-python must have the form '>=3.Y', got {spec!r}")
    return match.group(1)


def synced_pyproject(text: str, floor: str) -> str:
    """Drop classifiers below the floor and add the floor's if it's missing."""
    versions = CLASSIFIER.findall(text)
    if not versions:
        sys.exit("pyproject.toml has no 'Programming Language :: Python :: 3.Y'")
    text = CLASSIFIER.sub(
        lambda m: m.group(0) if as_tuple(m.group(1)) >= as_tuple(floor) else "",
        text,
    )
    if floor not in versions:
        anchor = '    "Programming Language :: Python :: 3",\n'
        if anchor not in text:
            sys.exit(f"can't add the {floor} classifier: {anchor.strip()} missing")
        line = f'    "Programming Language :: Python :: {floor}",\n'
        text = text.replace(anchor, anchor + line)
    return text


def supported(floor: str) -> list[str]:
    """Return the supported versions, oldest first, after syncing."""
    versions = CLASSIFIER.findall(synced_pyproject(PYPROJECT.read_text(), floor))
    return sorted(set(versions), key=as_tuple)


def ci_matrix(floor: str, experimental: list[str]) -> dict:
    """Build the CI test matrix: supported versions plus non-blocking extras."""
    blocking = supported(floor)
    extra = [v for v in experimental if v not in blocking]
    return {
        "include": [{"python": v, "experimental": False} for v in blocking]
        + [{"python": v, "experimental": True} for v in extra]
    }


def sync(floor: str) -> dict[Path, str]:
    """Return ``{path: new text}`` for every copy that's out of date."""
    stale = {}
    text = PYPROJECT.read_text()
    if (new := synced_pyproject(text, floor)) != text:
        stale[PYPROJECT] = new
    newest = supported(floor)[-1]
    copies = [(n, p, newest) for n, p in NEWEST_COPIES]
    copies += [(n, p, floor) for n, p in FLOOR_COPIES]
    for name, pattern, version in copies:
        path = ROOT / name
        text = path.read_text()
        matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
        if len(matches) != 1:
            sys.exit(f"{name}: expected one match for {pattern!r}, got {len(matches)}")
        start, end = matches[0].span(1)
        new = text[:start] + version + text[end:]
        if new != text:
            stale[path] = new
    return stale


def main() -> None:
    """Parse arguments and run the requested mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if out of sync")
    mode.add_argument("--floor", action="store_true", help="print the floor")
    mode.add_argument(
        "--ci-matrix",
        nargs="*",
        metavar="EXPERIMENTAL",
        help="print the CI matrix as JSON, with these versions non-blocking",
    )
    args = parser.parse_args()

    floor = read_floor()
    if args.floor:
        print(floor)
    elif args.ci_matrix is not None:
        print(json.dumps(ci_matrix(floor, args.ci_matrix)))
    elif args.check:
        stale = sync(floor)
        for path in stale:
            print(f"{path.relative_to(ROOT)} needs updating")
        if stale:
            sys.exit("run: uv run python scripts/sync_python_versions.py")
    else:
        for path, text in sync(floor).items():
            path.write_text(text)
            print(f"updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
