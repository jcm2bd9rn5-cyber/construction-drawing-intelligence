# Construction Drawing Intelligence

## Overview

Construction Drawing Intelligence is a local prototype for searching construction drawings and technical records with natural-language queries. Its grounded QA flow keeps the answer connected to evidence that can be traced back to a source page or Markdown section.

The repository includes only a small, fictional, text-based dataset. It contains no records for a real company or building.

## Features

- PDF and Markdown ingestion
- Document versioning by source identity and content hash
- Hybrid lexical, metadata, and optional vector search
- PostgreSQL and pgvector support, with SQLite for lightweight local startup
- Grounded QA with evidence and citation validation
- Bounding-box source highlighting for PDF pages
- Adaptive Vision extraction for sparse or scanned pages
- Conservative abstention when evidence is missing or contradictory

## Architecture

```text
PDF / Markdown
      |
      v
Ingestion -> versioned documents/pages/chunks -> PostgreSQL + pgvector
                                                   |
Natural-language query -> lexical + metadata + vector retrieval
                                                   |
                                                   v
                                  grounding and citation validation
                                                   |
                                                   v
                                    answer + evidence + source viewer
```

Gemini adapters are optional. Without a key, ingestion uses available PDF text and Markdown, search uses lexical and metadata signals, and QA uses the conservative local evidence path. No Gemini client is created when `GEMINI_API_KEY` is empty.

## Tech Stack

- Python 3.12+
- FastAPI
- PostgreSQL
- SQLAlchemy and Alembic
- pgvector
- Gemini (optional)
- PyMuPDF

## Getting Started

The quickest no-cost path uses SQLite and the bundled synthetic Markdown data:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m scripts.bootstrap
./start.sh
```

Open <http://localhost:8000>. The API schema is available at <http://localhost:8000/docs>.

For PostgreSQL and pgvector:

```bash
docker compose up -d db
cp .env.example .env
python -m alembic upgrade head
python -m scripts.bootstrap
./start.sh
```

The Compose credentials are local development defaults. Do not reuse them in a hosted or production environment.

## Gemini API

Gemini features require each user to supply their own API key. Copy `.env.example` to `.env` and set:

```dotenv
GEMINI_API_KEY=
```

The repository does not contain a key, does not expose one to frontend JavaScript, and does not configure a hosted endpoint. When the key is absent:

- the application still starts;
- no Gemini request is sent;
- lexical and metadata search remain available;
- Markdown and PDF text-layer ingestion remain available;
- Vision extraction, embeddings, and generated answers are disabled or use the conservative local fallback.

When a key is present, normal ingestion may use Vision and embeddings, and QA may use generated answers. These operations use the configured user's quota and may incur charges. `scripts.bootstrap` explicitly disables Vision and embeddings so that the bundled demo setup stays offline even if a key exists in the shell.

## Demo Data

Everything under `samples/demo-building/` is fictional and was written for this public demo. It includes four Markdown drawing summaries and one change record. It is not copied from a real drawing set.

Useful queries include:

- `受変電設備`
- `杭の工法と支持層`
- `変更資料CHG-001の内容は？`

## Grounding Design

Each evidence item receives a server-side Evidence ID. Generated claims may refer only to IDs supplied in the prompt, and the server rebuilds display citations from its own evidence objects. Numeric and relationship checks reject unsupported values or connections. When retrieval cannot support the requested fact, the QA flow abstains instead of filling the gap from model knowledge.

PDF evidence can retain normalized bounding-box coordinates. The viewer renders the source page and highlights the bounding rectangle associated with a selected citation.

See [docs/EVALUATION.md](docs/EVALUATION.md) for a public-data evaluation approach.

## Tests

Tests block construction of a live Gemini client and use injected fakes for model behavior:

```bash
env -u GEMINI_API_KEY -u GOOGLE_API_KEY pytest
```

The PostgreSQL integration test runs when `DATABASE_URL` points to an available PostgreSQL/pgvector database; otherwise it is skipped.

## Limitations

- OCR and Vision extraction can misread drawings.
- Grounding controls reduce but cannot eliminate hallucinations.
- Bounding boxes are displayed as enclosing rectangles rather than exact polygons.
- The prototype has no production authentication or multi-tenancy.
- The bundled dataset is intentionally small and synthetic.
- Table structure extraction remains heuristic.

## Future Work

- Authentication and tenant isolation
- Asynchronous ingestion and job queues
- Human review workflows
- Stronger table-structure reconstruction
- Production observability and cost controls
- Model and embedding migration tooling

## Licensing and Attribution

No project-level license is included yet. Before public release, the repository owner should confirm that they have the right to license all first-party code and then select a license; MIT is a possible candidate, not a legal conclusion. Third-party UI attribution is recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and preserved in the source comments.
