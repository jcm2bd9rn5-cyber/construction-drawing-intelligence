from fastapi import APIRouter
from app.schemas.qa import QARequest, QAResponse
from app.services.qa_service import answer_question
from app.services.qa_presentation import present_qa
router = APIRouter(prefix="/api/v1/qa", tags=["qa"])
@router.post("/", response_model=QAResponse)
def grounded_qa(request: QARequest) -> QAResponse:
    payload = answer_question(
        request.question,
        request.limit,
        floor=request.floor,
        discipline=request.discipline,
        source_type=request.source_type,
    )
    return QAResponse(**present_qa(payload, request.question))
