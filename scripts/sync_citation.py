"""Syncs CITATION.cff with current commit's date and version number."""

import datetime
import re
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path("pyproject.toml").read_text())
version = pyproject["project"]["version"]
today = datetime.datetime.now(datetime.UTC).date().isoformat()

cff_path = Path("CITATION.cff")
text = cff_path.read_text()
text = re.sub(r"^version: .*$", f'version: "{version}"', text, flags=re.MULTILINE)
text = re.sub(
    r"^date-released: .*$", f'date-released: "{today}"', text, flags=re.MULTILINE
)
cff_path.write_text(text)
