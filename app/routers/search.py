from fastapi import APIRouter
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.search_service import search
router = APIRouter(prefix="/api/v1/search", tags=["search"])
@router.post("/", response_model=SearchResponse)
def search_drawings(request: SearchRequest) -> SearchResponse:
    parsed, rows = search(
        request.query,
        request.limit,
        floor=request.floor,
        discipline=request.discipline,
        source_type=request.source_type,
    )
    return SearchResponse(query=request.query, parsed={"positive_terms": parsed.positive_terms, "negative_terms": parsed.negative_terms, "floors": parsed.floors, "drawing_types": parsed.drawing_types, "tokens": parsed.tokens}, results=[SearchResult(**r) for r in rows])
