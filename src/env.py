"""Tiny .env file loader — no external dependency needed.

Why hand-rolled instead of `python-dotenv`? So you don't have to `pip install`
anything: this is ~15 lines of plain Python that does the one thing we need.

What it does: reads a file called `.env` in the project root (the folder that
holds paper.py) and copies each `KEY=value` line into the process environment,
so code like `os.environ.get("TELEGRAM_BOT_TOKEN")` can see it.

Important rule: it NEVER overwrites a variable that's already set. That means a
real OS environment variable (or a GitHub Actions secret) always wins over the
file — the .env file is just a convenient default for local runs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("paper.env")

# The project root = one level up from this file's folder (src/ -> project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=value pairs from a .env file into os.environ (if present).

    Silently does nothing if the file doesn't exist — that's the normal case
    for people who set real environment variables or skip Telegram entirely.
    """
    env_path = Path(path) if path else _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # Skip blank lines and comments.
        if not line or line.startswith("#"):
            continue
        # Allow an optional leading "export " (so a .env copied from a shell works).
        if line.startswith("export "):
            line = line[len("export "):].strip()
        # Each real line must look like KEY=value.
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Strip one layer of surrounding quotes, if the user wrote VALUE="..." .
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        # Don't clobber something already set in the real environment.
        if key and key not in os.environ:
            os.environ[key] = value

    log.info("Loaded environment variables from %s", env_path)
