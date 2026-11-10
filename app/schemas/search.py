from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    floor: str | None = Field(default=None, max_length=32)
    discipline: str | None = Field(default=None, max_length=32)
    source_type: str | None = Field(default=None, max_length=32)

class SearchResult(BaseModel):
    drawing_number: str
    title: str
    floor: str = ""
    discipline: str = ""
    source_type: str = "pdf"
    revision: str | None = None
    page: int = 1
    score: float
    lexical_score: float = 0.0
    metadata_score: float = 0.0
    vector_score: float = 0.0
    semantic_proxy_score: float = 0.0
    fusion_score: float = 0.0
    snippet: str
    search_reason: list[str] = []
    source_path: str | None = None
    document_id: int | None = None
    page_id: int | None = None
    chunk_id: int | None = None
    version: str | None = None
    bbox: list[float] | None = None
    evidence_kind: str = "catalog"

class SearchResponse(BaseModel):
    query: str
    parsed: dict
    results: list[SearchResult]
