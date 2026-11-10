from datetime import datetime
from typing import Any
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, JSON, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import VECTOR
from app.core.database import Base

class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    drawing_number: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(500))
    document_type: Mapped[str] = mapped_column(String(100), default="drawing")
    discipline: Mapped[str] = mapped_column(String(100), default="")
    floor: Mapped[str] = mapped_column(String(100), default="")
    current_version_id: Mapped[int | None] = mapped_column(ForeignKey("document_versions.id"), nullable=True)
    versions: Mapped[list["DocumentVersion"]] = relationship(foreign_keys="DocumentVersion.document_id", cascade="all, delete-orphan")

class DocumentVersion(TimestampMixin, Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("file_path", "content_hash", name="uq_document_versions_source_hash"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version: Mapped[str] = mapped_column(String(32), default="1.0")
    file_path: Mapped[str] = mapped_column(String(1000))
    source_type: Mapped[str] = mapped_column(String(100), default="pdf")
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pages: Mapped[list["Page"]] = relationship(cascade="all, delete-orphan")

class Page(TimestampMixin, Base):
    __tablename__ = "pages"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_version_id: Mapped[int] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"))
    page_number: Mapped[int] = mapped_column(Integer)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    image_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="COMPLETED")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunks: Mapped[list["PageChunk"]] = relationship(cascade="all, delete-orphan")

class PageChunk(TimestampMixin, Base):
    __tablename__ = "page_chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text)
    # SQLite keeps a JSON representation for deterministic tests; PostgreSQL
    # stores the same 768 floats as a pgvector column.
    embedding: Mapped[Any | None] = mapped_column(
        VECTOR(768).with_variant(JSON(), "sqlite"), nullable=True
    )
    bbox: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[Any] = mapped_column("metadata", JSON, default=dict)

class ProcessingJob(TimestampMixin, Base):
    __tablename__ = "processing_jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    source: Mapped[str] = mapped_column(String(1000))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    pages_created: Mapped[int] = mapped_column(Integer, default=0)
    pages_failed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    embeddings_created: Mapped[int] = mapped_column(Integer, default=0)
    vision_pages: Mapped[int] = mapped_column(Integer, default=0)
    text_pages: Mapped[int] = mapped_column(Integer, default=0)
    searchable: Mapped[bool] = mapped_column(Boolean, default=False)
    warnings_json: Mapped[Any] = mapped_column("warnings", JSON, default=list)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retry_after_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    quota_metric: Mapped[str | None] = mapped_column(String(255), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    embedding_api_requests: Mapped[int] = mapped_column(Integer, default=0)
    embedding_batches: Mapped[int] = mapped_column(Integer, default=0)
