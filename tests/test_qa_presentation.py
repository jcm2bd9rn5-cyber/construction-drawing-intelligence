import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.qa_presentation import present_qa, source_label


def citation(eid="E1", page=1, version="1.0", source="pdf", text="EV-3(荷物用)"):
    return {"evidence_id": eid, "drawing_number": "D-103", "page": page, "version": version,
            "source_type": source, "quote": text, "bbox": [.1,.2,.4,.2,.4,.5,.1,.5]}


def test_same_page_groups_snippets_and_remaps_nonconsecutive_evidence_ids():
    payload = {"answer": "用途[E1]。台数[E3][E5]。", "citations": [citation(), citation("E3", text="別の根拠"), citation("E5")]}
    result = present_qa(payload, "用途と台数")
    assert len(result["citation_groups"]) == 1
    assert len(result["citation_groups"][0]["snippets"]) == 2
    assert result["display_answer"] == "用途[1]。台数[1]。"
    assert result["citations"] == payload["citations"]


def test_compound_evidence_markers_are_remapped_and_deduplicated():
    result = present_qa({"answer": "回答[E1, E3]補足[E3、E9]", "citations": [citation(),citation("E3"),citation("E9",page=2)]}, "質問")
    assert result["display_answer"] == "回答[1]補足[1][2]"


@pytest.mark.parametrize("other", [citation("E2",page=2), citation("E2",version="2.0"), citation("E2",source="markdown")])
def test_distinct_pages_versions_and_sources_do_not_merge(other):
    result = present_qa({"answer": "根拠[1][2]", "citations": [citation(),other]}, "質問")
    assert len(result["citation_groups"]) == 2
    assert result["display_answer"] == "根拠[1][2]"


@pytest.mark.parametrize("source,label", [("pdf","PDF / 図面"),("drawing","PDF / 図面"),("markdown","Markdown / 資料"),("change_order","Markdown / 資料"),("manual","Markdown / 資料"),("unknown","ソース未確認")])
def test_source_label_uses_provenance_field_only(source,label):
    assert source_label(source) == label


def test_group_viewer_keeps_version_page_and_each_region_separately():
    result = present_qa({"answer": "回答", "citations": [citation(page=3,version="2.0")]}, "用途")
    group = result["citation_groups"][0]
    assert "page=3&version=2.0" in group["viewer_url"]
    assert "bbox" not in group["viewer_url"]
    assert "bbox=" in group["snippets"][0]["viewer_url"]


def test_markdown_preview_is_short_but_full_evidence_remains_available():
    full = "資料\n" * 500 + "変更前: 従来型\n変更後: 高効率型、ドレン配管新設\n" + "その他\n" * 500
    result = present_qa({"answer": "変更内容", "citations": [citation(source="change_order",text=full)]}, "変更内容は？")
    snippet = result["citation_groups"][0]["snippets"][0]
    assert len(snippet["preview"]) <= 300
    assert "ドレン配管" in snippet["preview"]
    assert snippet["quote"] == full and snippet["viewer_url"] is None


def test_api_preserves_grouping_and_facet_diagnostics(monkeypatch):
    from app.routers import qa
    monkeypatch.setattr(qa, "answer_question", lambda *args, **kw: {"answer":"用途[E1]", "citations":[citation()], "confidence":.65,"evidence":[],
        "facet_coverage":[{"name":"用途","answered":True,"supported":True,"evidence_ids":["E1"]}]})
    response = TestClient(app).post("/api/v1/qa/",json={"question":"用途と台数"})
    assert response.status_code == 200
    result = response.json()
    assert result["display_answer"] == "用途[1]"
    assert result["citation_groups"][0]["source_label"] == "PDF / 図面"
    assert result["citations"][0]["version"] == "1.0"
    assert result["facet_coverage"][0]["supported"] is True
