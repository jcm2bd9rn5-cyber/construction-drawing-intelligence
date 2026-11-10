from fastapi.testclient import TestClient

from app.main import app
from app.services import search_service
from app.services.qa_service import answer_question, validate_citations


def test_health_and_public_branding():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    html = client.get("/").text
    assert "Construction Drawing Intelligence" in html
    assert "架空デモ資料" in html


def test_synthetic_catalog_supports_key_free_metadata_search():
    search_service.DOCS = None
    _parsed, rows = search_service.search("受変電設備", 5)
    assert rows
    assert rows[0]["drawing_number"] == "E-001"
    assert rows[0]["source_type"] == "markdown"


def test_missing_evidence_abstains_without_model_call():
    result = answer_question("存在しない設備の仕様は？", answerer=False)
    assert result["answer"] == "根拠不足のため回答できません。"
    assert result["citations"] == []


def test_citations_require_server_side_evidence_objects():
    forged = {"citations": [{"evidence_id": "E1", "drawing_number": "X", "page": 1, "quote": "x"}]}
    assert validate_citations(forged) is False
