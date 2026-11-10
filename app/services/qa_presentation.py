"""Page-level presentation without changing chunk-level evaluation citations."""
from __future__ import annotations

import re
from urllib.parse import urlencode, quote

from app.services.qa_facets import evidence_body, CITATION_MARKER


def source_label(source_type: str) -> str:
    return {"pdf": "PDF / 図面", "drawing": "PDF / 図面", "markdown": "Markdown / 資料",
            "change_order": "Markdown / 資料", "manual": "Markdown / 資料"}.get(source_type, "ソース未確認")


def viewer_url(citation: dict, region: bool = True) -> str | None:
    if citation.get("source_type") not in {"pdf", "drawing"}:
        return None
    params = {"page": citation.get("page", 1)}
    if citation.get("version"):
        params["version"] = citation["version"]
    if region and citation.get("bbox"):
        params["bbox"] = ",".join(str(v) for v in citation["bbox"])
    return f"/api/v1/documents/{quote(citation['drawing_number'], safe='')}/viewer?{urlencode(params)}"


def relevant_snippet(text: str, question: str, budget: int = 300) -> str:
    body = evidence_body(text)
    lines = [line.strip(" #|-\t") for line in body.splitlines() if line.strip(" #|-\t")]
    terms = re.findall(r"[一-龥]{2,}|[A-Za-z]+|\d+", question)
    ranked = sorted(enumerate(lines), key=lambda pair: (-sum(t in pair[1] for t in terms)
                    - (2 if re.search(r"変更前|変更後|変更箇所", pair[1]) else 0), pair[0]))
    selected = sorted(ranked[:3])
    snippet = "\n".join(line for _, line in selected)
    return snippet if len(snippet) <= budget else snippet[:budget - 1] + "…"


def present_qa(payload: dict, question: str) -> dict:
    groups, keyed, id_numbers, ordinal_numbers = [], {}, {}, {}
    for ordinal, citation in enumerate(payload.get("citations", []), 1):
        source = citation.get("source_type", "")
        canonical_source = "pdf" if source in {"drawing", "pdf"} else source
        key = (citation["drawing_number"], citation.get("version"), citation.get("page", 1), canonical_source)
        if key not in keyed:
            group = {"number": len(groups) + 1, "drawing_number": key[0], "version": key[1], "page": key[2],
                     "source_type": source, "source_label": source_label(source), "title": citation.get("title", ""),
                     "viewer_url": viewer_url(citation, region=False), "snippets": [], "evidence_ids": []}
            keyed[key] = group
            groups.append(group)
        group = keyed[key]
        eid = citation.get("evidence_id")
        if eid:
            id_numbers[eid] = group["number"]
            group["evidence_ids"].append(eid)
        ordinal_numbers[str(ordinal)] = group["number"]
        if not any(s["quote"] == citation["quote"] for s in group["snippets"]):
            group["snippets"].append({"evidence_id": eid, "quote": citation["quote"],
                                      "preview": relevant_snippet(citation["quote"], question),
                                      "viewer_url": viewer_url(citation),
                                      "citation_scope": "region" if citation.get("bbox") else "page"})
    def remap(match):
        numbers = []
        for token in re.findall(r"E?\d+", match.group(1)):
            number = id_numbers.get(token) if token.startswith("E") else ordinal_numbers.get(token)
            if number and number not in numbers:
                numbers.append(number)
        return "".join(f"[{number}]" for number in sorted(set(numbers)))
    display = CITATION_MARKER.sub(remap, payload["answer"])
    display = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", display)
    if groups and not re.search(r"\[\d+\]", display):
        display += "\n" + " ".join(f"[{group['number']}]" for group in groups)
    return {**payload, "display_answer": display, "citation_groups": groups}
