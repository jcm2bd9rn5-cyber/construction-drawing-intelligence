import logging

import pymupdf
import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Document, DocumentVersion, Page, PageChunk
from app.repositories.chunks import current_chunks, vector_candidates
from app.services.ingestion import TextChunk, ingest_pdf


class IntegrationVision:
    def extract(self, image: bytes):
        assert image.startswith(b"\x89PNG")
        return [TextChunk("架空の1階設備資料", [0.1, 0.1, 0.9, 0.1, 0.9, 0.2, 0.1, 0.2], "gemini_vision")]


class IntegrationEmbedder:
    def documents(self, contents, title=""):
        return [[0.01] * 768 for _ in contents]


def test_real_postgres_pgvector_ingestion_path(tmp_path, caplog):
    engine = create_engine(get_settings().database_url, pool_pre_ping=True)
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL DATABASE_URL is not configured")
    try:
        connection = engine.connect()
    except SQLAlchemyError as exc:
        pytest.skip(f"PostgreSQL is unavailable: {exc}")

    source = tmp_path / "IT-001_vector.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(source)
    outer_transaction = connection.begin()
    try:
        with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
            job = ingest_pdf(session, source, vision=IntegrationVision(), embedder=IntegrationEmbedder())
            assert job.status == "COMPLETED"
            matching = [row for row in current_chunks(session) if row[0].drawing_number == "IT-001"]
            assert len(matching) == 1
            assert len(matching[0][3].embedding) == 768
            caplog.set_level(logging.INFO, logger="app.repositories.chunks")
            candidates = vector_candidates(session, [0.01] * 768, limit=5)
            assert candidates and candidates[0][0].id == matching[0][3].id
            assert connection.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
    finally:
        outer_transaction.rollback()
        connection.close()
        engine.dispose()

    verify_engine = create_engine(get_settings().database_url)
    try:
        with Session(verify_engine) as session:
            assert session.scalar(select(Document).where(Document.drawing_number == "IT-001")) is None
            assert session.scalar(select(DocumentVersion).where(DocumentVersion.file_path == str(source))) is None
            assert session.scalar(select(func.count(PageChunk.id)).join(Page, PageChunk.page_id == Page.id).join(DocumentVersion, Page.document_version_id == DocumentVersion.id).where(DocumentVersion.file_path == str(source))) == 0
    finally:
        verify_engine.dispose()
