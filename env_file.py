"""Load KEY=VALUE lines from the repository's git-ignored .env into os.environ.

Values already set in the environment win. Blank lines and '#' comments are skipped.
"""

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env(path=ENV_PATH):
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
