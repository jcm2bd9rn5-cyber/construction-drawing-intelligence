"""Create the local schema and ingest the bundled synthetic Markdown samples."""
from __future__ import annotations

import subprocess
import sys

from app.core.config import PROJECT_ROOT, get_settings


def main() -> None:
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=PROJECT_ROOT, check=True)
    root = get_settings().data_root
    sources = sorted((root / "drawings").glob("*.md")) + sorted((root / "changes").glob("*.md"))
    if not sources:
        raise SystemExit(f"No synthetic demo documents found under {root}")
    subprocess.run(
        [sys.executable, "-m", "scripts.ingest", *map(str, sources), "--no-vision", "--no-embed"],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
