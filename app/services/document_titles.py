"""Presentation-only display titles for source documents."""
from __future__ import annotations

from pathlib import Path
import re


def display_title(title: str | None, source_path: str | Path | None, identifier: str) -> str:
    """Return a human-readable title without mutating indexed metadata."""
    if title and title.strip():
        return title.strip()
    path = Path(source_path) if source_path else None
    if path and path.suffix.lower() in {".md", ".markdown"}:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
                if heading and heading.group(1).strip():
                    return heading.group(1).strip()
        except OSError:
            pass
    return identifier
