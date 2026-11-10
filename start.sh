#!/bin/sh
# Start the local application on port 8000.
set -eu
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run --with 'fastapi>=0.115' --with 'uvicorn[standard]>=0.30' --with 'pydantic>=2.8' --with 'pydantic-settings>=2.4' --with 'sqlalchemy>=2.0' uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
