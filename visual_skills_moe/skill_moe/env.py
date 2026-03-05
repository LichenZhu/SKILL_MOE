from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    """
    Minimal .env loader to avoid extra deps.
    - Ignores blank lines and lines starting with '#'.
    - Does not override variables already set in the environment.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key and key not in os.environ:
            os.environ[key] = val
