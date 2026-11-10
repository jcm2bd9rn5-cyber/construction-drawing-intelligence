from pydantic import BaseModel, Field
from app.schemas.search import SearchResult

class Citation(BaseModel):
    evidence_id: str | None = None
    drawing_number: str
    document_id: int | None = None
    page_id: int | None = None
    chunk_id: int | None = None
    page: int = 1
    quote: str
    bbox: list[float] | None = None
    source_type: str = "drawing"
    version: str | None = None
    title: str = ""
    provenance: str = "text"
    citation_scope: str = "page"

class CitationSnippet(BaseModel):
    evidence_id: str | None = None
    quote: str
    preview: str
    viewer_url: str | None = None
    citation_scope: str

class CitationGroup(BaseModel):
    number: int
    drawing_number: str
    title: str = ""
    version: str | None = None
    page: int
    source_type: str
    source_label: str
    viewer_url: str | None = None
    evidence_ids: list[str]
    snippets: list[CitationSnippet]

class FacetCoverage(BaseModel):
    name: str
    answered: bool
    supported: bool
    evidence_ids: list[str]

class QARequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=20)
    floor: str | None = Field(default=None, max_length=32)
    discipline: str | None = Field(default=None, max_length=32)
    source_type: str | None = Field(default=None, max_length=32)

class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    evidence: list[SearchResult] = []
    display_answer: str = ""
    citation_groups: list[CitationGroup] = []
    facet_coverage: list[FacetCoverage] = []
    facet_coverage_score: float | None = None
    facet_fallback_query: str | None = None
