"""Syncs CITATION.cff's version from pyproject.toml, re-dating it on a version bump.

``date-released`` changes only when the version does, so ordinary edits to
pyproject.toml (dependency floors, metadata) leave the release date alone.
"""

import datetime
import re
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path("pyproject.toml").read_text())
version = pyproject["project"]["version"]

cff_path = Path("CITATION.cff")
text = cff_path.read_text()
current = re.search(r'^version: "?([^"\n]*)"?$', text, flags=re.MULTILINE)
if current is None or current.group(1) != version:
    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    text = re.sub(r"^version: .*$", f'version: "{version}"', text, flags=re.MULTILINE)
    text = re.sub(
        r"^date-released: .*$", f'date-released: "{today}"', text, flags=re.MULTILINE
    )
    cff_path.write_text(text)
