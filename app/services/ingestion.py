"""Atomic, idempotent PDF ingestion into versioned page evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

import pymupdf
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.ai.client import DailyQuotaExhausted
from app.core.config import PROJECT_ROOT, get_settings
from app.models.document import Document, DocumentVersion, Page, PageChunk, ProcessingJob
from app.services.evidence_quality import assess_evidence_quality, quality_metadata


@dataclass(frozen=True)
class TextChunk:
    content: str
    bbox: list[float] | None
    extraction_method: str = "text_layer"


@dataclass
class IngestionOutcome:
    pages_created: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    embeddings_created: int = 0
    vision_pages: int = 0
    text_pages: int = 0
    searchable: bool = False
    warnings: list[str] = field(default_factory=list)
    error_type: str | None = None
    http_status: int | None = None
    retry_after_seconds: float | None = None
    quota_metric: str | None = None
    attempts: int = 0
    embedding_api_requests: int = 0
    embedding_batches: int = 0


class NoSearchableContent(Exception):
    pass


def normalized_bbox(
    rect: tuple[float, float, float, float], width: float, height: float
) -> list[float]:
    x0, y0, x1, y1 = rect
    return [
        max(0, min(1, x0 / width)),
        max(0, min(1, y0 / height)),
        max(0, min(1, x1 / width)),
        max(0, min(1, y0 / height)),
        max(0, min(1, x1 / width)),
        max(0, min(1, y1 / height)),
        max(0, min(1, x0 / width)),
        max(0, min(1, y1 / height)),
    ]


def text_layer_chunks(page: pymupdf.Page, max_chars: int = 1200) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for block in page.get_text("blocks"):
        content = re.sub(r"\s+", " ", block[4]).strip()
        if not content:
            continue
        bbox = normalized_bbox(
            tuple(block[:4]), page.rect.width, page.rect.height
        )
        for start in range(0, len(content), max_chars):
            chunks.append(TextChunk(content[start : start + max_chars], bbox))
    return chunks


def needs_vision(
    chunks: list[TextChunk],
    min_chars: int = 80,
    quality_threshold: float = 0.42,
) -> bool:
    """Return whether extraction is too weak to be the page's only evidence."""
    text = "".join(chunk.content for chunk in chunks)
    useful = sum(char.isalnum() for char in text)
    return useful < min_chars or assess_evidence_quality(text).score < quality_threshold


def source_identity(path: Path) -> tuple[str, str]:
    stem = path.stem
    number = stem.split("_", 1)[0]
    return number, "1.1" if "_rev1" in stem else "1.0"


def _image_bytes(page: pymupdf.Page) -> bytes:
    return page.get_pixmap(
        matrix=pymupdf.Matrix(1.5, 1.5), alpha=False
    ).tobytes("png")


def _enriched_evidence_text(document: Document, version: DocumentVersion, page: Page, text: str) -> str:
    """Put stable drawing context next to extracted text for lexical/vector QA.

    The actual source text remains in ``raw_text`` and metadata; this prefix is
    an indexable envelope, not a replacement OCR result.
    """
    context = [
        f"図面番号: {document.drawing_number}",
        f"図面名: {document.title}",
    ]
    if document.floor:
        context.append(f"階: {document.floor}")
    if document.discipline:
        context.append(f"分野: {document.discipline}")
    context.extend((f"Revision: {version.version}", f"ページ: {page.page_number}", "抽出本文:"))
    return "\n".join(context) + "\n" + text.strip()


def _finish_job(
    session: Session,
    job_id: int,
    status: str,
    outcome: IngestionOutcome,
    error: str | None = None,
) -> ProcessingJob:
    job = session.get(ProcessingJob, job_id)
    if job is None:
        raise RuntimeError(f"Processing job {job_id} disappeared")
    job.status = status
    job.error = error
    job.pages_created = outcome.pages_created
    job.pages_failed = outcome.pages_failed
    job.chunks_created = outcome.chunks_created
    job.embeddings_created = outcome.embeddings_created
    job.vision_pages = outcome.vision_pages
    job.text_pages = outcome.text_pages
    job.searchable = outcome.searchable
    job.warnings_json = outcome.warnings
    job.error_type = outcome.error_type
    job.http_status = outcome.http_status
    job.retry_after_seconds = outcome.retry_after_seconds
    job.quota_metric = outcome.quota_metric
    job.attempts = outcome.attempts
    job.embedding_api_requests = outcome.embedding_api_requests
    job.embedding_batches = outcome.embedding_batches
    session.commit()
    session.refresh(job)
    return job


def _existing_outcome(
    session: Session, version: DocumentVersion, embed_requested: bool
) -> IngestionOutcome:
    pages = session.scalars(
        select(Page).where(Page.document_version_id == version.id)
    ).all()
    rows = session.execute(
        select(Page, PageChunk)
        .join(PageChunk, PageChunk.page_id == Page.id)
        .where(Page.document_version_id == version.id)
    ).all()
    chunks = [chunk for _page, chunk in rows]
    outcome = IngestionOutcome(
        pages_created=len(pages),
        pages_failed=sum(page.status != "COMPLETED" for page in pages),
        chunks_created=len(chunks),
        embeddings_created=sum(chunk.embedding is not None for chunk in chunks),
        vision_pages=len({
            page.id
            for page, chunk in rows
            if chunk.metadata_json
            and chunk.metadata_json.get("extraction_method") == "gemini_vision"
        }),
        text_pages=len({
            page.id
            for page, chunk in rows
            if not chunk.metadata_json
            or chunk.metadata_json.get("extraction_method") == "text_layer"
        }),
        searchable=bool(chunks),
    )
    if embed_requested and outcome.embeddings_created < outcome.chunks_created:
        outcome.warnings.append("Existing chunks are missing one or more embeddings")
    return outcome


def _cache_path(digest: str, page_number: int) -> Path:
    cache_dir = get_settings().vision_cache_dir
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    settings = get_settings()
    safe_prompt = re.sub(r"[^A-Za-z0-9_.-]+", "_", settings.vision_prompt_version)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", settings.gemini_model)
    return cache_dir / (
        f"{digest}-page-{page_number}-prompt-{safe_prompt}-model-{safe_model}.json"
    )


def _load_vision_cache(digest: str, page_number: int) -> list[TextChunk] | None:
    path = _cache_path(digest, page_number)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            TextChunk(
                item["content"], item.get("bbox"), item.get("extraction_method", "gemini_vision")
            )
            for item in payload
            if item.get("content", "").strip()
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_vision_cache(digest: str, page_number: int, chunks: list[TextChunk]) -> None:
    path = _cache_path(digest, page_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            [
                {
                    "content": chunk.content,
                    "bbox": chunk.bbox,
                    "extraction_method": chunk.extraction_method,
                }
                for chunk in chunks
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _ingest_searchable_graph(
    session: Session,
    path: Path,
    digest: str,
    *,
    embedder=None,
    vision=None,
) -> IngestionOutcome:
    existing = session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.file_path == str(path),
            DocumentVersion.content_hash == digest,
        )
    )
    if existing is not None:
        outcome = _existing_outcome(session, existing, embedder is not None)
        if outcome.searchable and outcome.pages_failed == 0:
            document = session.get(Document, existing.document_id)
            if document is not None and document.current_version_id != existing.id:
                document.current_version_id = existing.id
            return outcome
        # A previous run may have published a partially searchable version.
        # Remove that version transactionally so a retry can reprocess the
        # failed pages without violating the source-hash idempotency key.
        document = session.get(Document, existing.document_id)
        if document is not None and document.current_version_id == existing.id:
            document.current_version_id = None
        session.delete(existing)
        session.flush()

    number, version_label = source_identity(path)
    document = session.scalar(
        select(Document).where(Document.drawing_number == number)
    )
    if document is None:
        document = Document(
            drawing_number=number,
            title=path.stem.split("_", 1)[-1],
            document_type="drawing",
            discipline="",
            floor="",
        )
        session.add(document)
        session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=version_label,
        file_path=str(path),
        source_type="pdf",
        content_hash=digest,
    )
    session.add(version)
    session.flush()

    outcome = IngestionOutcome()
    settings = get_settings()
    pending_chunks: list[tuple[Page, TextChunk, dict]] = []
    with pymupdf.open(path) as source:
        for page_number, pdf_page in enumerate(source, start=1):
            chunks: list[TextChunk] = []
            page_error: str | None = None
            try:
                text_chunks = text_layer_chunks(pdf_page)
                raw_text = "\n".join(chunk.content for chunk in text_chunks)
                text_quality = assess_evidence_quality(
                    raw_text,
                    good_threshold=settings.evidence_quality_good_threshold,
                    vision_threshold=settings.evidence_quality_vision_threshold,
                )
                sparse = needs_vision(
                    text_chunks,
                    settings.vision_min_text_chars,
                    settings.evidence_quality_vision_threshold,
                )
                chunks = list(text_chunks)
                vision_text = ""
                provenance = "text"
                if sparse and vision is not None:
                    vision_chunks = _load_vision_cache(digest, page_number)
                    if not vision_chunks:
                        vision_chunks = vision.extract(_image_bytes(pdf_page))
                        if vision_chunks:
                            _save_vision_cache(digest, page_number, vision_chunks)
                    if vision_chunks:
                        vision_text = "\n".join(chunk.content for chunk in vision_chunks)
                        # Keep the text layer as an auditable primary record;
                        # Vision supplements weak pages rather than erasing it.
                        chunks.extend(vision_chunks)
                        provenance = "text+vision" if text_chunks else "vision"
                        outcome.vision_pages += 1
                    else:
                        outcome.warnings.append(
                            f"page {page_number}: Vision returned no text; using text layer"
                        )
                elif sparse:
                    outcome.warnings.append(
                        f"page {page_number}: insufficient text and Vision is disabled"
                    )
                if not chunks:
                    outcome.warnings.append(
                        f"page {page_number}: no searchable chunks were extracted"
                    )
            except DailyQuotaExhausted:
                raise
            except Exception as exc:
                page_error = str(exc)

            if page_error is not None:
                outcome.pages_failed += 1
                outcome.warnings.append(f"page {page_number}: isolated failure: {page_error}")
                failed_page = Page(
                    document_version_id=version.id,
                    page_number=page_number,
                    raw_text=raw_text if 'raw_text' in locals() else "",
                    summary="",
                    status="ERROR",
                    error=page_error,
                )
                session.add(failed_page)
                session.flush()
                outcome.pages_created += 1
                continue
            if not chunks:
                continue
            if text_chunks:
                outcome.text_pages += 1

            page = Page(
                document_version_id=version.id,
                page_number=page_number,
                raw_text=raw_text,
                summary="",
            )
            session.add(page)
            session.flush()
            outcome.pages_created += 1
            effective_quality = assess_evidence_quality(
                "\n".join(chunk.content for chunk in chunks),
                good_threshold=settings.evidence_quality_good_threshold,
                vision_threshold=settings.evidence_quality_vision_threshold,
            )
            for chunk in chunks:
                original = chunk.content if chunk.extraction_method == "text_layer" else ""
                vision_value = chunk.content if chunk.extraction_method != "text_layer" else vision_text
                effective = _enriched_evidence_text(document, version, page, chunk.content)
                pending_chunks.append((
                    page,
                    TextChunk(effective, chunk.bbox, chunk.extraction_method),
                    {
                        **quality_metadata(effective_quality),
                        "source_quality_score": text_quality.score,
                        "source_quality_class": text_quality.quality_class,
                        "extraction_provenance": provenance,
                        "vision_prompt_version": settings.vision_prompt_version,
                        "vision_model": settings.gemini_model,
                        "original_text": original,
                        "vision_text": vision_value,
                        "effective_text": effective,
                    },
                ))

    vectors: list[list[float] | None]
    if embedder is not None and pending_chunks:
        try:
            vectors = embedder.documents(
                [chunk.content for _page, chunk, _metadata in pending_chunks], title=document.title
            )
            if len(vectors) != len(pending_chunks):
                raise ValueError(
                    f"Embedding response count mismatch: requested {len(pending_chunks)}, "
                    f"received {len(vectors)}"
                )
        except DailyQuotaExhausted:
            raise
        except Exception as bulk_exc:
            # A malformed response or one failing batch should not discard
            # successfully extracted pages. Retry by page and isolate only the
            # page whose embedding still fails.
            outcome.warnings.append(
                f"document embedding failed; retrying per page: {bulk_exc}"
            )
            vectors = [None] * len(pending_chunks)
            for page in list(dict.fromkeys(page for page, _chunk, _metadata in pending_chunks)):
                page_items = [
                    (index, chunk)
                    for index, (candidate, chunk, _metadata) in enumerate(pending_chunks)
                    if candidate.id == page.id
                ]
                try:
                    page_vectors = embedder.documents(
                        [chunk.content for _index, chunk in page_items],
                        title=document.title,
                    )
                    if len(page_vectors) != len(page_items):
                        raise ValueError(
                            f"Embedding response count mismatch on page {page.page_number}"
                        )
                    for (index, _chunk), vector in zip(page_items, page_vectors):
                        vectors[index] = vector
                except DailyQuotaExhausted:
                    raise
                except Exception as page_exc:
                    page.status = "ERROR"
                    page.error = str(page_exc)
                    outcome.pages_failed += 1
                    outcome.warnings.append(
                        f"page {page.page_number}: isolated embedding failure: {page_exc}"
                    )
    else:
        vectors = [None] * len(pending_chunks)

    outcome.chunks_created = len(pending_chunks)
    outcome.embeddings_created = sum(vector is not None for vector in vectors)
    for (page, chunk, metadata), vector in zip(pending_chunks, vectors):
        session.add(
            PageChunk(
                page_id=page.id,
                content=chunk.content,
                embedding=vector,
                bbox=chunk.bbox,
                metadata_json={
                    "document_id": document.id,
                    "document_version_id": version.id,
                    "page_number": page.page_number,
                    "extraction_method": chunk.extraction_method,
                    **metadata,
                },
            )
        )

    if outcome.chunks_created == 0:
        if outcome.pages_failed == 0:
            raise NoSearchableContent("No searchable chunks were extracted from the PDF")
        outcome.searchable = False
        return outcome
    if embedder is None:
        outcome.warnings.append(
            "Embeddings are disabled; evidence is available to lexical search only"
        )
    outcome.searchable = True
    current = (
        session.get(DocumentVersion, document.current_version_id)
        if document.current_version_id is not None
        else None
    )
    if current is None or version_label >= current.version:
        document.current_version_id = version.id
    return outcome


def reprocess_vision_page(
    session: Session,
    path: Path,
    page_number: int,
    *,
    vision,
    embedder=None,
) -> dict:
    """Replace one page's evidence with a fresh, versioned Vision extraction.

    This intentionally does not call ``ingest_pdf``: targeted recovery must not
    re-embed the other pages in a large document.  Existing page chunks are
    deleted only after Vision and the optional page-only embedding succeed.
    """
    path = path.resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    version = session.scalar(
        select(DocumentVersion).where(DocumentVersion.content_hash == digest)
    )
    if version is None:
        raise LookupError(f"No ingested document version matches {path}")
    document = session.get(Document, version.document_id)
    page = session.scalar(
        select(Page).where(
            Page.document_version_id == version.id,
            Page.page_number == page_number,
        )
    )
    if document is None or page is None:
        raise LookupError(f"No ingested page {page_number} matches {path}")

    old_chunks = session.scalars(
        select(PageChunk).where(PageChunk.page_id == page.id).order_by(PageChunk.id)
    ).all()
    old_evidence = [
        {
            "chunk_id": chunk.id,
            "content": chunk.content,
            "effective_text": (chunk.metadata_json or {}).get("effective_text", chunk.content),
            "vision_text": (chunk.metadata_json or {}).get("vision_text", ""),
            "extraction_provenance": (chunk.metadata_json or {}).get("extraction_provenance", "text"),
        }
        for chunk in old_chunks
    ]

    settings = get_settings()
    with pymupdf.open(path) as source:
        if page_number < 1 or page_number > len(source):
            raise ValueError(f"Page {page_number} is outside PDF range 1..{len(source)}")
        pdf_page = source[page_number - 1]
        text_chunks = text_layer_chunks(pdf_page)
        raw_text = "\n".join(chunk.content for chunk in text_chunks)
        vision_chunks = vision.extract(_image_bytes(pdf_page))
    if not vision_chunks:
        return {
            "accepted": False,
            "reason": "vision_returned_no_text",
            "drawing_number": document.drawing_number,
            "page": page_number,
            "vision_calls": 1,
            "embedding_api_requests": 0,
            "old_chunks": old_evidence,
            "new_chunks": [],
        }

    _save_vision_cache(digest, page_number, vision_chunks)
    chunks = [*text_chunks, *vision_chunks]
    vision_text = "\n".join(chunk.content for chunk in vision_chunks)
    provenance = "text+vision" if text_chunks else "vision"
    quality = assess_evidence_quality(
        "\n".join(chunk.content for chunk in chunks),
        good_threshold=settings.evidence_quality_good_threshold,
        vision_threshold=settings.evidence_quality_vision_threshold,
    )
    text_quality = assess_evidence_quality(
        raw_text,
        good_threshold=settings.evidence_quality_good_threshold,
        vision_threshold=settings.evidence_quality_vision_threshold,
    )
    pending: list[tuple[TextChunk, dict]] = []
    for chunk in chunks:
        effective = _enriched_evidence_text(document, version, page, chunk.content)
        pending.append((
            TextChunk(effective, chunk.bbox, chunk.extraction_method),
            {
                **quality_metadata(quality),
                "source_quality_score": text_quality.score,
                "source_quality_class": text_quality.quality_class,
                "extraction_provenance": provenance,
                "vision_prompt_version": settings.vision_prompt_version,
                "vision_model": settings.gemini_model,
                "original_text": chunk.content if chunk.extraction_method == "text_layer" else "",
                "vision_text": vision_text if chunk.extraction_method == "text_layer" else chunk.content,
                "effective_text": effective,
            },
        ))

    initial_embedding_requests = getattr(embedder, "embedding_api_requests", 0)
    if embedder is not None:
        vectors = embedder.documents(
            [chunk.content for chunk, _metadata in pending], title=document.title
        )
        if len(vectors) != len(pending):
            raise ValueError(
                f"Targeted embedding response count mismatch: requested {len(pending)}, "
                f"received {len(vectors)}"
            )
    else:
        vectors = [None] * len(pending)

    session.execute(delete(PageChunk).where(PageChunk.page_id == page.id))
    session.flush()
    page.raw_text = raw_text
    page.status = "COMPLETED"
    page.error = None
    new_evidence = []
    for (chunk, metadata), vector in zip(pending, vectors):
        created = PageChunk(
            page_id=page.id,
            content=chunk.content,
            embedding=vector,
            bbox=chunk.bbox,
            metadata_json={
                "document_id": document.id,
                "document_version_id": version.id,
                "page_number": page_number,
                "extraction_method": chunk.extraction_method,
                **metadata,
            },
        )
        session.add(created)
        new_evidence.append({
            "content": chunk.content,
            "effective_text": metadata["effective_text"],
            "vision_text": metadata["vision_text"],
            "extraction_provenance": metadata["extraction_provenance"],
        })
    session.flush()
    return {
        "accepted": True,
        "reason": "targeted_vision_reextraction",
        "drawing_number": document.drawing_number,
        "page": page_number,
        "document_hash": digest,
        "vision_prompt_version": settings.vision_prompt_version,
        "vision_model": settings.gemini_model,
        "vision_calls": 1,
        "embedding_api_requests": max(
            0, getattr(embedder, "embedding_api_requests", 0) - initial_embedding_requests
        ),
        "old_chunks": old_evidence,
        "new_chunks": new_evidence,
        "old_chunk_count": len(old_evidence),
        "new_chunk_count": len(new_evidence),
        "embedding_count": sum(vector is not None for vector in vectors),
    }


def ingest_pdf(
    session: Session, path: Path, *, embedder=None, vision=None
) -> ProcessingJob:
    path = path.resolve()
    job = ProcessingJob(status="PROCESSING", source=str(path), warnings_json=[])
    session.add(job)
    session.commit()
    job_id = job.id
    outcome = IngestionOutcome()
    initial_embedding_stats = (
        getattr(embedder, "embedding_api_requests", 0),
        getattr(embedder, "embedding_batches", 0),
    )

    def record_embedding_stats() -> None:
        outcome.embedding_api_requests = max(
            0,
            getattr(embedder, "embedding_api_requests", 0)
            - initial_embedding_stats[0],
        )
        outcome.embedding_batches = max(
            0,
            getattr(embedder, "embedding_batches", 0) - initial_embedding_stats[1],
        )

    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        outcome = _ingest_searchable_graph(
            session, path, digest, embedder=embedder, vision=vision
        )
        record_embedding_stats()
        session.commit()
        status = "COMPLETED_WITH_WARNINGS" if outcome.warnings else "COMPLETED"
        return _finish_job(session, job_id, status, outcome)
    except NoSearchableContent as exc:
        session.rollback()
        record_embedding_stats()
        outcome.warnings = [str(exc)]
        outcome.searchable = False
        return _finish_job(
            session, job_id, "COMPLETED_WITH_WARNINGS", outcome
        )
    except DailyQuotaExhausted as exc:
        session.rollback()
        record_embedding_stats()
        outcome.searchable = False
        outcome.error_type = "DAILY_QUOTA_EXHAUSTED"
        outcome.http_status = exc.http_status or 429
        outcome.retry_after_seconds = exc.retry_after_seconds
        outcome.quota_metric = exc.quota_metric
        outcome.attempts = exc.attempts
        outcome.warnings.append(
            "Embedding daily quota exhausted; database graph was rolled back and "
            "the Vision cache was retained for resume"
        )
        return _finish_job(
            session, job_id, "DEFERRED", outcome, str(exc)
        )
    except Exception as exc:
        session.rollback()
        record_embedding_stats()
        outcome.searchable = False
        outcome.error_type = "GEMINI_API_ERROR" if outcome.embedding_batches else "INGESTION_ERROR"
        outcome.http_status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        outcome.attempts = getattr(exc, "gemini_attempts", 1)
        return _finish_job(session, job_id, "ERROR", outcome, str(exc))


def ingest_document(session: Session, path: Path, *, embedder=None, vision=None) -> ProcessingJob:
    """Dispatch a supported source to its format-specific loader."""
    if path.suffix.lower() == ".md":
        from app.services.markdown_ingestion import ingest_markdown

        return ingest_markdown(session, path, embedder=embedder)
    if path.suffix.lower() == ".pdf":
        return ingest_pdf(session, path, embedder=embedder, vision=vision)
    raise ValueError(f"Unsupported source extension: {path.suffix or '<none>'}")
