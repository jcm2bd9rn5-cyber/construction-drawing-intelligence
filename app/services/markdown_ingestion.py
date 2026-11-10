"""Persist Markdown supporting documents using the same evidence graph."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentVersion, Page, PageChunk, ProcessingJob
from app.services.ingestion import IngestionOutcome, NoSearchableContent, _existing_outcome
from app.services.source_loaders import MarkdownLoader


@dataclass(frozen=True)
class MarkdownIdentity:
    drawing_number: str
    title: str


def markdown_identity(path: Path) -> MarkdownIdentity:
    prefix = path.stem.split("_", 1)[0]
    return MarkdownIdentity(prefix if prefix else path.stem, path.stem)


def ingest_markdown(session: Session, path: Path, *, embedder=None) -> ProcessingJob:
    path = path.resolve()
    if path.suffix.lower() != ".md":
        raise ValueError(f"Markdown loader requires .md: {path}")
    job = ProcessingJob(status="PROCESSING", source=str(path), warnings_json=[])
    session.add(job)
    session.commit()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    outcome = IngestionOutcome()

    def finish(status: str, error: str | None = None) -> ProcessingJob:
        job.status = status
        job.error = error
        job.pages_created = outcome.pages_created
        job.pages_failed = outcome.pages_failed
        job.chunks_created = outcome.chunks_created
        job.embeddings_created = outcome.embeddings_created
        job.searchable = outcome.searchable
        job.warnings_json = outcome.warnings
        job.embedding_api_requests = getattr(embedder, "embedding_api_requests", 0)
        job.embedding_batches = getattr(embedder, "embedding_batches", 0)
        session.commit()
        session.refresh(job)
        return job

    try:
        existing = session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.file_path == str(path),
                DocumentVersion.content_hash == digest,
            )
        )
        if existing is not None:
            outcome = _existing_outcome(session, existing, embedder is not None)
            if outcome.searchable and outcome.pages_failed == 0:
                return finish("COMPLETED_WITH_WARNINGS" if outcome.warnings else "COMPLETED")
            document = session.get(Document, existing.document_id)
            if document is not None and document.current_version_id == existing.id:
                document.current_version_id = None
            session.delete(existing)
            session.flush()

        identity = markdown_identity(path)
        document = session.scalar(
            select(Document).where(Document.drawing_number == identity.drawing_number)
        )
        if document is None:
            document = Document(
                drawing_number=identity.drawing_number,
                title=identity.title,
                document_type="supporting",
                discipline="supporting",
                floor="",
            )
            session.add(document)
            session.flush()
        previous_versions = session.scalars(
            select(DocumentVersion).where(DocumentVersion.document_id == document.id)
        ).all()
        version = DocumentVersion(
            document_id=document.id,
            version=f"1.{len(previous_versions)}",
            file_path=str(path),
            source_type="markdown",
            content_hash=digest,
        )
        session.add(version)
        session.flush()

        sections = MarkdownLoader().load(path)
        if not sections:
            raise NoSearchableContent("No non-empty Markdown sections were found")
        pending: list[tuple[Page, str]] = []
        for section in sections:
            page = Page(
                document_version_id=version.id,
                page_number=section.number,
                raw_text=section.content,
                summary=section.heading,
                status="COMPLETED",
            )
            session.add(page)
            session.flush()
            pending.append((page, section.content))
            outcome.pages_created += 1

        vectors = embedder.documents(
            [content for _page, content in pending], title=document.title
        ) if embedder is not None else [None] * len(pending)
        if len(vectors) != len(pending):
            raise ValueError("Markdown embedding response count mismatch")
        for (page, content), vector in zip(pending, vectors):
            session.add(
                PageChunk(
                    page_id=page.id,
                    content=content,
                    embedding=vector,
                    bbox=None,
                    metadata_json={
                        "document_id": document.id,
                        "document_version_id": version.id,
                        "page_number": page.page_number,
                        "source_type": "markdown",
                        "provenance": "section",
                    },
                )
            )
        outcome.chunks_created = len(pending)
        outcome.embeddings_created = sum(vector is not None for vector in vectors)
        outcome.searchable = True
        if embedder is None:
            outcome.warnings.append("Embeddings are disabled; evidence is available to lexical search only")
        document.current_version_id = version.id
        session.commit()
        return finish("COMPLETED_WITH_WARNINGS" if outcome.warnings else "COMPLETED")
    except NoSearchableContent as exc:
        session.rollback()
        outcome.warnings = [str(exc)]
        return finish("COMPLETED_WITH_WARNINGS")
    except Exception as exc:
        session.rollback()
        return finish("ERROR", str(exc))
