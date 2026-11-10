from types import SimpleNamespace
from app.services.qa_service import Evidence, _rerank_rows, _suppress_duplicates, resolve_evidence_ids, validate_citations, answer_question, collect_evidence, format_evidence
from app.evaluation.metrics import citation_scores, must_contain_accuracy

def _evidence():
    return [Evidence("E1", "D-101", 2, "給水ポンプは地下2階にあります。", None, 7, 8, 9),
            Evidence("E2", "D-102", 1, "受水槽の容量は18m3です。", None, 10, 11, 12)]

def test_evidence_id_resolution_rejects_unknown_and_deduplicates():
    assert [e.evidence_id for e in resolve_evidence_ids(["E1", "E2", "E1"], _evidence())] == ["E1", "E2"]
    import pytest
    with pytest.raises(ValueError): resolve_evidence_ids(["E99"], _evidence())

def test_citation_fields_are_resolved_from_server_evidence():
    evidence = _evidence()
    citation = {"evidence_id": "E1", "drawing_number": "D-101", "page": 2, "quote": evidence[0].quote}
    assert validate_citations({"citations": [citation]}, evidence)
    assert not validate_citations({"citations": [{**citation, "quote": "捏造"}]}, evidence)

def test_grounded_answer_selects_existing_id_only(monkeypatch):
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (_evidence(), []))
    answerer = SimpleNamespace(answer=lambda q, text: SimpleNamespace(answer="給水ポンプは地下2階。", evidence_ids=["E1"]))
    out = answer_question("給水ポンプはどこ", answerer=answerer)
    assert out["citations"][0]["chunk_id"] == 9
    assert out["citations"][0]["quote"] == _evidence()[0].quote
    bad = SimpleNamespace(answer=lambda q, text: SimpleNamespace(answer="捏造", evidence_ids=["E99"]))
    fallback = answer_question("給水ポンプはどこ", answerer=bad)
    assert fallback["citations"] and fallback["citations"][0]["evidence_id"] == "E1"

def test_multiple_citations_and_empty_evidence(monkeypatch):
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (_evidence(), []))
    answerer = SimpleNamespace(answer=lambda q, text: SimpleNamespace(answer="給水ポンプと受水槽。", evidence_ids=["E1", "E2"]))
    assert len(answer_question("給水ポンプと受水槽", answerer=answerer)["citations"]) == 1
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: ([], []))
    assert answer_question("未記載")['confidence'] == 0

def test_unsubstantiated_numeric_claim_is_rejected(monkeypatch):
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (_evidence(), []))
    bad = SimpleNamespace(answer=lambda q, text: SimpleNamespace(answer="容量は999m3", evidence_ids=["E2"]))
    fallback = answer_question("容量", answerer=bad)
    assert fallback["citations"] and "18m3" in fallback["answer"]

def test_generation_failure_falls_back_to_verified_evidence(monkeypatch):
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (_evidence(), []))
    failing = SimpleNamespace(answer=lambda q, text: (_ for _ in ()).throw(RuntimeError("503 unavailable")))
    out = answer_question("給水ポンプはどこ", answerer=failing)
    assert "地下2階" in out["answer"]
    assert "E1" in [citation["evidence_id"] for citation in out["citations"]]
    assert validate_citations({"citations": out["citations"]}, _evidence())


def test_claims_choose_evidence_ids_without_model_generated_metadata(monkeypatch):
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (_evidence(), []))
    answerer = SimpleNamespace(
        answer=lambda q, text: SimpleNamespace(
            answer="給水ポンプは地下2階、受水槽の容量は18m3です。",
            claims=[
                SimpleNamespace(text="給水ポンプは地下2階", evidence_ids=["E1"]),
                SimpleNamespace(text="容量は18m3", evidence_ids=["E2"]),
            ],
        )
    )
    out = answer_question("給水ポンプと受水槽の容量", answerer=answerer)
    assert [item["evidence_id"] for item in out["citations"]] == ["E1", "E2"]
    assert all("document_id" in item and "quote" in item for item in out["citations"])


def test_unknown_claim_id_falls_back_to_verified_evidence(monkeypatch):
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (_evidence(), []))
    answerer = SimpleNamespace(
        answer=lambda q, text: SimpleNamespace(
            answer="給水ポンプは地下2階。",
            claims=[SimpleNamespace(text="給水ポンプ", evidence_ids=["E999"])]
        )
    )
    out = answer_question("給水ポンプはどこ", answerer=answerer)
    assert out["citations"]
    assert all(item["evidence_id"] in {"E1", "E2"} for item in out["citations"])

def test_valid_but_unsupported_claim_id_uses_deterministic_support_backstop(monkeypatch):
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (_evidence(), []))
    answerer = SimpleNamespace(
        answer=lambda q, text: SimpleNamespace(
            answer="容量は18m3です。",
            claims=[SimpleNamespace(text="容量は18m3", evidence_ids=["E1"])],
        )
    )
    out = answer_question("容量", answerer=answerer)
    assert out["citations"][0]["evidence_id"] == "E2"


def test_generated_claim_keeps_multiple_direct_supports(monkeypatch):
    evidence = [Evidence("E1", "X", 1, "受電電圧は6.6kVです。", None), Evidence("E2", "Y", 1, "受電電圧は6.6kVです。", None)]
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (evidence, []))
    model = SimpleNamespace(answer=lambda q, text: SimpleNamespace(answer="受電電圧は6.6kVです。", claims=[SimpleNamespace(text="受電電圧は6.6kVです。", evidence_ids=["E1", "E2"])]))
    assert [item["evidence_id"] for item in answer_question("受電電圧は？", answerer=model)["citations"]] == ["E1"]


def test_relation_claim_requires_one_direct_evidence_span(monkeypatch):
    evidence = [
        Evidence("E1", "E25", 1, "受信機", None),
        Evidence("E2", "E25", 1, "P型複合盤、防災センターリタ回路", None),
    ]
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (evidence, []))
    model = SimpleNamespace(answer=lambda q, text: SimpleNamespace(
        answer="受信機は防災センターに設置されます。", evidence_ids=["E1", "E2"]
    ))
    out = answer_question("自動火災報知設備の受信機はどこに設置されますか", answerer=model)
    assert out["relation_status"] == "unsupported"
    assert out["citations"] == []


def test_format_evidence_exposes_stable_id_source_page_and_text():
    formatted = format_evidence(_evidence())
    assert "[E1]" in formatted
    assert "source: drawing D-101" in formatted
    assert "page: 2" in formatted
    assert "text:" in formatted

def test_qa_metrics_and_context_bound():
    assert citation_scores(["A", "B"], ["A", "X"]) == {"CitationPrecision": 0.5, "CitationRecall": 0.5, "CitationF1": 0.5}
    assert must_contain_accuracy([["地下", "B2"], ["ポンプ"]], "B2のポンプ") == 1
    assert "[E1]" in format_evidence(_evidence(), 100)


def test_qa_rerank_prefers_high_quality_structured_evidence():
    rows = [
        {"drawing_number": "M01", "title": "給水", "page": 1, "snippet": "給水ポンプの概要です。", "score": 3, "quality_score": .15, "evidence_kind": "pdf_chunk", "document_id": 1, "page_id": 1, "chunk_id": 1},
        {"drawing_number": "M02", "title": "給水", "page": 1, "snippet": "給水ポンプ P-1 の容量は18m3、揚程35mです。", "score": 3, "quality_score": .92, "evidence_kind": "pdf_chunk", "document_id": 2, "page_id": 2, "chunk_id": 2},
    ]
    ranked = _rerank_rows("給水ポンプの容量は？", rows, 2)
    assert ranked[0]["drawing_number"] == "M02"


def test_soft_duplicate_penalty_allows_strong_multichunk_and_cross_document_evidence():
    rows = [
        {"drawing_number": "A01", "document_id": 1, "page_id": 10, "chunk_id": index, "qa_score": 10 - index}
        for index in range(1, 5)
    ] + [{"drawing_number": "M01", "document_id": 2, "page_id": 20, "chunk_id": 8, "qa_score": 8.8}]
    selected = _suppress_duplicates(rows, 4)
    assert len([row for row in selected if row["page_id"] == 10]) == 3
    assert {row["document_id"] for row in selected} == {1, 2}


def test_collect_evidence_merges_conjunctive_subquery_candidates(monkeypatch):
    from app.services.query_parser import parse_query

    calls = []
    search_rows = [
        {"drawing_number": "S-101", "title": "杭工法", "score": 2, "evidence_kind": "catalog"},
        {"drawing_number": "S-102", "title": "支持層", "score": 2, "evidence_kind": "catalog"},
    ]

    def fake_search(query, limit, **_filters):
        return parse_query(query), search_rows

    def fake_candidates(query, limit, **_filters):
        calls.append(query)
        drawing = "S-101" if "工法" in query else "S-102"
        return [{
            "drawing_number": drawing,
            "title": drawing,
            "page": 1,
            "snippet": "杭の工法: 場所打ち" if drawing == "S-101" else "支持層: 砂礫",
            "score": 3,
            "quality_score": .9,
            "document_id": 1 if drawing == "S-101" else 2,
            "page_id": 1 if drawing == "S-101" else 2,
            "chunk_id": 1 if drawing == "S-101" else 2,
            "evidence_kind": "pdf_chunk",
        }]

    monkeypatch.setattr("app.services.qa_service.search", fake_search)
    monkeypatch.setattr("app.services.qa_service.qa_evidence_candidates", fake_candidates)
    evidence, _rows = collect_evidence("杭の工法と支持層を教えてください", limit=2)
    assert calls == [
        "杭の工法と支持層を教えてください",
        "杭の工法",
        "杭の支持層を教えてください",
    ]
    assert {item.drawing_number for item in evidence} == {"S-101", "S-102"}


def test_citation_without_bbox_is_explicitly_page_scoped(monkeypatch):
    monkeypatch.setattr("app.services.qa_service.collect_evidence", lambda q, l: (_evidence(), []))
    answerer = SimpleNamespace(answer=lambda q, text: SimpleNamespace(answer="給水ポンプは地下2階。", evidence_ids=["E1"]))
    citation = answer_question("給水ポンプはどこ", answerer=answerer)["citations"][0]
    assert citation["citation_scope"] == "page"
