"""Turns coverage.py's JSON report into a shields.io endpoint badge.

Usage: ``python scripts/coverage_badge.py coverage.json coverage-badge.json``.
CI publishes the output to the ``badges`` branch, and the README renders it via
``https://img.shields.io/endpoint?url=<raw URL of that file>``.
"""

import json
import sys
from pathlib import Path

# (lower bound in percent, shields.io color), checked from the top down.
COLORS = [
    (95, "brightgreen"),
    (90, "green"),
    (80, "yellowgreen"),
    (70, "yellow"),
    (50, "orange"),
    (0, "red"),
]

report_path, badge_path = map(Path, sys.argv[1:3])
# Round down, so the shown number never crosses a color threshold it hasn't met.
percent = int(json.loads(report_path.read_text())["totals"]["percent_covered"])
color = next(name for bound, name in COLORS if percent >= bound)
badge = {
    "schemaVersion": 1,
    "label": "coverage",
    "message": f"{percent}%",
    "color": color,
}
badge_path.write_text(json.dumps(badge) + "\n")
