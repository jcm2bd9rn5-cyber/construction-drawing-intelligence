"""Small source-loader abstractions for non-PDF supporting documents."""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class MarkdownSection:
    number: int
    heading: str
    content: str
    start_line: int
    end_line: int


class SourceLoader(Protocol):
    def load(self, path: Path) -> list[MarkdownSection]: ...


class MarkdownLoader:
    """Split Markdown into heading-scoped sections with line provenance."""

    def load(self, path: Path) -> list[MarkdownSection]:
        lines = path.read_text(encoding="utf-8").splitlines()
        sections: list[MarkdownSection] = []
        heading = path.stem
        start_line = 1
        body: list[str] = []

        def flush(end_line: int) -> None:
            content = "\n".join(body).strip()
            if content:
                sections.append(
                    MarkdownSection(
                        number=len(sections) + 1,
                        heading=heading,
                        content=content,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )

        for line_number, line in enumerate(lines, 1):
            match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if match:
                flush(line_number - 1)
                heading = match.group(1)
                start_line = line_number
                body = [line]
            else:
                body.append(line)
        flush(len(lines))
        return sections
