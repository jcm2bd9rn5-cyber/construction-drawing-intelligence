from pathlib import Path
import pymupdf
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.core.database import Base
from app.models.document import Document, DocumentVersion, Page, PageChunk, ProcessingJob
from app.services.ingestion import (
    _cache_path,
    ingest_document,
    ingest_pdf,
    needs_vision,
    reprocess_vision_page,
    TextChunk,
    source_identity,
)
from app.repositories.chunks import current_chunks, vector_candidates
from app.ai.client import DailyQuotaExhausted, QuotaDetails

class FakeEmbedder:
    def document(self, content, title):
        return [1.0, 0.0]

    def documents(self, contents, title=""):
        return [[1.0, 0.0] for _ in contents]

def _pdf(path: Path, text: str):
    doc = pymupdf.open()
    page = doc.new_page()
    if text: page.insert_text((50, 50), text)
    doc.save(path)

def test_ingestion_is_idempotent_and_stores_page_chunks(tmp_path):
    path = tmp_path / "T01-01_test.pdf"
    _pdf(path, "A drawing note about mechanical equipment " * 4)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert ingest_pdf(session, path, embedder=FakeEmbedder()).status == "COMPLETED"
        assert ingest_pdf(session, path, embedder=FakeEmbedder()).status == "COMPLETED"
        assert session.scalar(select(DocumentVersion).where(DocumentVersion.file_path == str(path.resolve()))).content_hash
        chunks = current_chunks(session)
        assert len(chunks) == 1
        assert chunks[0][2].page_number == 1
        assert chunks[0][3].bbox and len(chunks[0][3].bbox) == 8
        assert vector_candidates(session, [1, 0])[0][1] == 1.0

def test_latest_version_wins_and_old_version_remains(tmp_path):
    old = tmp_path / "T01-01_test.pdf"
    new = tmp_path / "T01-01_test_rev1.pdf"
    _pdf(old, "old detail")
    _pdf(new, "new detail")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        ingest_pdf(session, old)
        ingest_pdf(session, new)
        document = session.scalar(select(Document).where(Document.drawing_number == "T01-01"))
        assert session.get(DocumentVersion, document.current_version_id).version == "1.1"
        assert len(session.scalars(select(DocumentVersion)).all()) == 2
        assert len(current_chunks(session)) == 1

def test_scanned_pdf_vision_fallback_only_when_sparse(monkeypatch, tmp_path):
    class FakeVision:
        calls = 0
        def extract(self, image):
            self.calls += 1
            return [TextChunk("読取済みの注記", None, "gemini_vision")]
    path = tmp_path / "T01-02_scan.pdf"
    _pdf(path, "")
    monkeypatch.setattr(
        "app.services.ingestion._cache_path",
        lambda digest, page: tmp_path / f"{digest}-{page}.json",
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    vision = FakeVision()
    with Session(engine) as session:
        job = ingest_pdf(session, path, vision=vision)
        assert vision.calls == 1
        chunk = session.scalar(select(PageChunk))
        assert chunk.content.endswith("読取済みの注記")
        assert chunk.metadata_json["extraction_provenance"] == "vision"
        assert chunk.metadata_json["vision_text"] == "読取済みの注記"
        assert job.vision_pages == 1 and job.chunks_created == 1 and job.searchable
    assert needs_vision([TextChunk("a" * 90, None)]) is True
    assert source_identity(Path("A01-01_title_rev1.pdf")) == ("A01-01", "1.1")


def test_text_sufficient_does_not_call_vision(tmp_path):
    class UnexpectedVision:
        def extract(self, _image):
            raise AssertionError("Vision must not be called for sufficient text")

    path = tmp_path / "T01-03_text.pdf"
    _pdf(path, "searchable drawing text " * 8)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        job = ingest_pdf(session, path, vision=UnexpectedVision())
        assert job.text_pages == 1
        assert job.vision_pages == 0
        assert job.chunks_created == 1


def test_zero_chunks_is_warning_and_leaves_no_search_graph(tmp_path):
    path = tmp_path / "T01-04_empty.pdf"
    _pdf(path, "")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        job = ingest_pdf(session, path)
        assert job.status == "COMPLETED_WITH_WARNINGS"
        assert job.searchable is False and job.chunks_created == 0
        assert job.warnings_json
        assert session.scalars(select(ProcessingJob)).all() == [job]
        assert session.scalars(select(Document)).all() == []
        assert session.scalars(select(DocumentVersion)).all() == []
        assert session.scalars(select(Page)).all() == []
        assert session.scalars(select(PageChunk)).all() == []


def test_failure_rolls_back_graph_and_retry_is_idempotent(monkeypatch, tmp_path):
    class FailingVision:
        def extract(self, _image):
            raise RuntimeError("503 unavailable")

    class SuccessfulVision:
        def extract(self, _image):
            return [TextChunk("Vision evidence", None, "gemini_vision")]

    path = tmp_path / "T01-05_retry.pdf"
    _pdf(path, "")
    monkeypatch.setattr(
        "app.services.ingestion._cache_path",
        lambda digest, page: tmp_path / f"{digest}-{page}.json",
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        failed = ingest_pdf(session, path, vision=FailingVision())
        assert failed.status == "COMPLETED_WITH_WARNINGS" and failed.pages_failed == 1
        assert "503" in failed.warnings_json[0]
        assert len(session.scalars(select(Document)).all()) == 1
        assert len(session.scalars(select(DocumentVersion)).all()) == 1
        assert session.scalar(select(Page)).status == "ERROR"
        assert session.scalars(select(PageChunk)).all() == []

        succeeded = ingest_pdf(
            session,
            path,
            vision=SuccessfulVision(),
            embedder=FakeEmbedder(),
        )
        repeated = ingest_pdf(
            session,
            path,
            vision=SuccessfulVision(),
            embedder=FakeEmbedder(),
        )
        assert succeeded.status == repeated.status == "COMPLETED"
        assert len(session.scalars(select(Document)).all()) == 1
        assert len(session.scalars(select(DocumentVersion)).all()) == 1
        assert len(session.scalars(select(Page)).all()) == 1
        chunks = session.scalars(select(PageChunk)).all()
        assert len(chunks) == 1 and chunks[0].embedding == [1.0, 0.0]
        assert len(session.scalars(select(ProcessingJob)).all()) == 3


def test_page_failure_isolated_and_retryable(monkeypatch, tmp_path):
    class ThreePageVision:
        calls = 0

        def extract(self, _image):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("page 2 OCR failed")
            return [TextChunk(f"page {self.calls} evidence", None, "gemini_vision")]

    path = tmp_path / "T01-07_pages.pdf"
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    doc.save(path)
    monkeypatch.setattr(
        "app.services.ingestion._cache_path",
        lambda digest, page: tmp_path / "vision-cache" / f"{digest}-{page}.json",
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    vision = ThreePageVision()
    with Session(engine) as session:
        partial = ingest_pdf(session, path, vision=vision, embedder=FakeEmbedder())
        assert partial.status == "COMPLETED_WITH_WARNINGS"
        assert partial.searchable is True and partial.pages_failed == 1
        pages = session.scalars(select(Page).order_by(Page.page_number)).all()
        assert [page.status for page in pages] == ["COMPLETED", "ERROR", "COMPLETED"]
        assert len(session.scalars(select(PageChunk)).all()) == 2

        completed = ingest_pdf(session, path, vision=vision, embedder=FakeEmbedder())
        assert completed.status == "COMPLETED"
        assert completed.pages_failed == 0
        assert len(session.scalars(select(PageChunk)).all()) == 3


def test_markdown_ingestion_persists_sections_and_versions(tmp_path):
    path = tmp_path / "CHG-001_照明変更.md"
    path.write_text(
        "# 変更概要\n\n給湯器を更新する。\n\n## 影響図面\n\nE-001を確認する。\n",
        encoding="utf-8",
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        first = ingest_document(session, path, embedder=FakeEmbedder())
        assert first.status == "COMPLETED"
        version = session.scalar(select(DocumentVersion))
        assert version.source_type == "markdown"
        pages = session.scalars(select(Page).order_by(Page.page_number)).all()
        assert len(pages) == 2
        assert pages[0].summary == "変更概要"
        assert pages[0].chunks[0].metadata_json["provenance"] == "section"
        assert len(session.scalars(select(PageChunk)).all()) == 2

        repeated = ingest_document(session, path, embedder=FakeEmbedder())
        assert repeated.status == "COMPLETED"
        assert len(session.scalars(select(DocumentVersion)).all()) == 1

        path.write_text(path.read_text(encoding="utf-8") + "\n追記。", encoding="utf-8")
        changed = ingest_document(session, path, embedder=FakeEmbedder())
        assert changed.status == "COMPLETED"
        assert len(session.scalars(select(DocumentVersion)).all()) == 2
        assert session.scalar(select(Document).where(Document.drawing_number == "CHG-001")).current_version_id


def test_daily_quota_defers_and_reuses_cached_vision(monkeypatch, tmp_path):
    class CountingVision:
        def __init__(self):
            self.calls = 0

        def extract(self, _image):
            self.calls += 1
            return [TextChunk("cached Vision evidence", None, "gemini_vision")]

    class QuotaEmbedder:
        embedding_api_requests = 0
        embedding_batches = 0

        def documents(self, _contents, title=""):
            self.embedding_api_requests += 1
            self.embedding_batches += 1
            raise DailyQuotaExhausted(
                "embedding",
                QuotaDetails(
                    retry_after_seconds=48330,
                    metric="embed_content_free_tier_requests",
                    daily=True,
                ),
                RuntimeError("daily quota"),
                1,
            )

    class SuccessfulEmbedder:
        embedding_api_requests = 0
        embedding_batches = 0

        def documents(self, contents, title=""):
            self.embedding_api_requests += 1
            self.embedding_batches += 1
            return [[1.0, 0.0] for _content in contents]

    cache_dir = tmp_path / "vision-cache"
    monkeypatch.setattr(
        "app.services.ingestion._cache_path",
        lambda digest, page: cache_dir / f"{digest}-{page}.json",
    )
    path = tmp_path / "T01-06_quota.pdf"
    _pdf(path, "")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    vision = CountingVision()
    with Session(engine) as session:
        deferred = ingest_pdf(
            session, path, embedder=QuotaEmbedder(), vision=vision
        )
        assert deferred.status == "DEFERRED"
        assert deferred.error_type == "DAILY_QUOTA_EXHAUSTED"
        assert deferred.http_status == 429
        assert deferred.retry_after_seconds == 48330
        assert deferred.embedding_api_requests == 1
        assert session.scalars(select(Document)).all() == []

        completed = ingest_pdf(
            session, path, embedder=SuccessfulEmbedder(), vision=vision
        )
        assert completed.status == "COMPLETED"
        assert vision.calls == 1
        chunk = session.scalar(select(PageChunk))
        assert chunk.content.endswith("cached Vision evidence")
        assert chunk.metadata_json["extraction_provenance"] == "vision"


def test_vision_cache_key_contains_prompt_and_model_version():
    path = _cache_path("a" * 64, 3)
    assert "page-3" in path.name
    assert "prompt-" in path.name
    assert "model-" in path.name


def test_targeted_vision_reprocesses_only_one_page(monkeypatch, tmp_path):
    class TargetVision:
        def extract(self, _image):
            return [TextChunk("新しい表の関係 行A | 300kVA | 消防", None, "gemini_vision")]

    path = tmp_path / "T01-08_targeted.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((50, 50), "old page one evidence " * 6)
    document.new_page().insert_text((50, 50), "untouched page two evidence " * 6)
    document.save(path)
    monkeypatch.setattr(
        "app.services.ingestion._cache_path",
        lambda digest, page: tmp_path / "vision-cache" / f"{digest}-{page}.json",
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        ingest_pdf(session, path, embedder=FakeEmbedder())
        before = {
            page.page_number: [chunk.content for chunk in page.chunks]
            for page in session.scalars(select(Page).order_by(Page.page_number)).all()
        }
        result = reprocess_vision_page(
            session,
            path,
            1,
            vision=TargetVision(),
            embedder=FakeEmbedder(),
        )
        session.commit()
        assert result["accepted"] is True
        assert result["vision_calls"] == 1
        assert result["embedding_api_requests"] == 0
        pages = session.scalars(select(Page).order_by(Page.page_number)).all()
        assert before[2] == [chunk.content for chunk in pages[1].chunks]
        assert any("新しい表の関係" in chunk.content for chunk in pages[0].chunks)
        assert all(
            chunk.metadata_json["vision_prompt_version"]
            for chunk in pages[0].chunks
        )


def test_targeted_vision_empty_result_preserves_existing_page(monkeypatch, tmp_path):
    class EmptyVision:
        def extract(self, _image):
            return []

    path = tmp_path / "T01-09_targeted_empty.pdf"
    _pdf(path, "existing evidence " * 8)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        ingest_pdf(session, path, embedder=FakeEmbedder())
        before = [chunk.content for chunk in session.scalars(select(PageChunk)).all()]
        result = reprocess_vision_page(session, path, 1, vision=EmptyVision())
        assert result["accepted"] is False
        assert [chunk.content for chunk in session.scalars(select(PageChunk)).all()] == before
