"""Persistence and nearest-neighbor access for page evidence."""
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.document import Document, DocumentVersion, Page, PageChunk
from app.ai.embeddings import cosine_similarity

logger = logging.getLogger(__name__)

def vector_candidates(session: Session, vector: list[float], limit: int = 100) -> list[tuple[PageChunk, float]]:
    stmt = (
        select(PageChunk)
        .join(Page, PageChunk.page_id == Page.id)
        .join(DocumentVersion, Page.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Document.current_version_id == DocumentVersion.id, PageChunk.embedding.is_not(None))
    )
    if session.bind.dialect.name == "postgresql":
        logger.info(
            "Executing pgvector cosine similarity query (dimensions=%d, limit=%d)",
            len(vector),
            limit,
        )
        distance = PageChunk.embedding.cosine_distance(vector)
        rows = session.execute(stmt.add_columns(distance).order_by(distance).limit(limit)).all()
        logger.info("pgvector returned %d vector candidates", len(rows))
        return [(chunk, max(0.0, 1.0 - float(dist))) for chunk, dist in rows]
    rows = session.scalars(stmt).all()
    ranked = [(chunk, cosine_similarity(vector, chunk.embedding)) for chunk in rows]
    return sorted(ranked, key=lambda pair: -pair[1])[:limit]

def current_chunks(session: Session) -> list[tuple[Document, DocumentVersion, Page, PageChunk]]:
    stmt = (
        select(Document, DocumentVersion, Page, PageChunk)
        .join(DocumentVersion, DocumentVersion.document_id == Document.id)
        .join(Page, Page.document_version_id == DocumentVersion.id)
        .join(PageChunk, PageChunk.page_id == Page.id)
        .where(Document.current_version_id == DocumentVersion.id)
        .order_by(Document.drawing_number, Page.page_number, PageChunk.id)
    )
    return list(session.execute(stmt).all())
