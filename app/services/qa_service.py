"""Grounded QA. The model selects evidence IDs; the server builds citations."""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from app.core.config import get_settings
from app.services.search_service import search, qa_evidence_candidates, documents
from app.services.query_parser import decompose_query, parse_query
from app.services.qa_facets import Facet, extract_facets, supporting_spans, evidence_spans, validate_coverage, numeric_facts, evidence_body, strip_citation_markers, valid_numeric_tokens, valid_voltage_expressions
from app.services.document_titles import display_title

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QARerankOptions:
    quality_boost: bool = True
    numeric_entity_boost: bool = True
    page_cap: bool = True
    document_cap: bool = True
    document_diversity: bool = True
    duplicate_suppression: bool = True
    deterministic_backstop: bool = True


QA_RERANK_OPTIONS = QARerankOptions()


@contextmanager
def qa_rerank_options(**changes):
    """Temporarily toggle one ranking component for offline ablation tests."""
    global QA_RERANK_OPTIONS
    previous = QA_RERANK_OPTIONS
    QA_RERANK_OPTIONS = QARerankOptions(**{
        field: changes.get(field, getattr(previous, field))
        for field in QARerankOptions.__dataclass_fields__
    })
    try:
        yield QA_RERANK_OPTIONS
    finally:
        QA_RERANK_OPTIONS = previous

@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    drawing_number: str
    page: int
    quote: str
    bbox: list[float] | None
    document_id: int | None = None
    page_id: int | None = None
    chunk_id: int | None = None
    source_type: str = "drawing"
    provenance: str = "text"
    quality_score: float = 0.0
    version: str | None = None
    title: str = ""
    semantic_types: tuple[str, ...] = ()
    grounding_status: str = "text_layer"


TYPED_GROUNDING_FACETS = {
    "電圧", "容量", "台数", "数量", "人数", "定員", "積載量", "重量", "寸法",
    "階数", "部屋数", "戸数", "型式", "型番",
}

QA_STOP_TERMS = {
    "教えてください", "教えて", "教え", "確認したい", "確認", "どの", "どこ", "何", "それぞれ",
    "この建物", "方式", "方法", "用途", "影響", "理由", "起因", "構成", "種類", "納まり",
}
QA_SHORT_TERMS = {
    "空調", "換気", "給水", "排水", "給湯", "消火", "防火", "防水", "笠木", "階", "部屋",
    "容量", "電圧", "電源", "方式", "用途", "種類", "設備", "室名", "駐輪", "ガス", "EV", "R32",
}
QA_STRUCTURED_TERMS = {
    "容量", "台数", "数量", "寸法", "型番", "電圧", "金額", "日付", "階", "GL", "FL", "EL",
    "流量", "揚程", "高さ", "幅", "長さ", "m3", "kVA", "仕様", "品番",
}

FACET_EVIDENCE_TYPES = {
    "電圧": "voltage", "容量": "apparent_power", "台数": "count", "数量": "count",
    "人数": "person_capacity", "定員": "person_capacity", "積載量": "load_capacity",
    "重量": "load_capacity", "寸法": "dimension", "場所": "floor", "設置場所": "floor",
    "型式": "model_number", "型番": "model_number", "変更内容": "generic_text",
}


def evidence_semantic_types(text: str) -> tuple[str, ...]:
    """Infer lightweight, deterministic evidence types from attested text."""
    body = evidence_body(text)
    types: list[str] = []
    patterns = (
        ("voltage", r"\d+(?:[.,]\d+)?\s*(?:kV|V)(?!A)"),
        ("apparent_power", r"\d+(?:[.,]\d+)?\s*kVA"),
        ("power", r"\d+(?:[.,]\d+)?\s*kW"),
        ("current", r"\d+(?:[.,]\d+)?\s*A\b"),
        ("count", r"\d+\s*(?:台|基|個|本|枚|箇所|か所|戸)"),
        ("person_capacity", r"\d+\s*人"),
        ("load_capacity", r"(?:積載|荷重|重量|質量|自重)[^。\n]{0,30}\d+\s*(?:kg|t|トン)"),
        ("dimension", r"\d+(?:\.\d+)?\s*(?:mm|cm|m)(?![A-Za-z0-9²³])"),
        ("floor", r"(?:地下)?\d+\s*階|\bB?[0-9]+F\b"),
        ("room", r"[一-龥ァ-ヶ]+室"),
        ("model_number", r"(?:型式|型番|品番)\s*[:：]?|[A-Z]{2,}[-][A-Z0-9-]+"),
        ("revision", r"(?:Rev(?:ision)?|改訂)\s*[:：]?\s*[A-Za-z0-9.]+"),
        ("date", r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"),
        ("money", r"[¥￥]\s*[\d,]+|\d+(?:\.\d+)?\s*万円"),
    )
    for semantic_type, pattern in patterns:
        if re.search(pattern, body, re.I):
            types.append(semantic_type)
    if re.search(r"(?:機器|設備|ポンプ|発電機|変圧器|受電盤|EV)", body):
        types.append("equipment")
    return tuple(types or ["generic_text"])


def _source_identity(item: Evidence) -> tuple[object, object]:
    """A repeated Vision call for one page is never independent evidence."""
    return (item.document_id or item.drawing_number, item.page_id or item.page)


def _entity_labels(text: str) -> set[str]:
    return {label.upper().replace("－", "-").replace("−", "-")
            for label in re.findall(r"\b(?:TR|EV|DG|UPS)[-−－]?\d+\b", evidence_body(text), re.I)}


def _typed_values(facet: Facet, item: Evidence) -> set[tuple[str, str]]:
    values: set[tuple[str, str]] = set()
    for span in evidence_spans(facet, item):
        for value, unit, _role in numeric_facts(span):
            values.add((value.replace(",", ""), unit.lower()))
    return values


def grounding_status_for(facet: Facet, item: Evidence, evidence: list[Evidence]) -> str:
    """Compute a claim trust status without storing manual audit decisions.

    Text-layer chunks can support typed values. Vision chunks remain available
    to retrieval and citations, but need the same valid value on a distinct
    source page. Competing values for an explicitly matching equipment label
    remain contradicted rather than being settled by a retrieval score.
    """
    if facet.name not in TYPED_GROUNDING_FACETS or item.grounding_status == "text_layer":
        return "TEXT_LAYER"
    values = _typed_values(facet, item)
    if not values:
        return "VISION_UNVERIFIED"
    labels = _entity_labels(item.quote)
    corroborated = False
    conflicting = False
    for other in evidence:
        if other is item or _source_identity(other) == _source_identity(item):
            continue
        other_labels = _entity_labels(other.quote)
        if labels and other_labels and not (labels & other_labels):
            continue
        other_values = _typed_values(facet, other)
        if not other_values:
            continue
        if values & other_values:
            corroborated = True
        elif labels and labels & other_labels:
            conflicting = True
    if conflicting:
        return "CONTRADICTED"
    return "VISION_CORROBORATED" if corroborated else "VISION_UNVERIFIED"


VOLTAGE_ROLES = {
    "RECEIVING_VOLTAGE", "DISTRIBUTION_VOLTAGE", "BUS_VOLTAGE",
    "SECONDARY_VOLTAGE", "CONTROL_VOLTAGE", "GENERIC_VOLTAGE",
}


def _voltage_role(text: str) -> str:
    """Classify only the attested role words, never the numeric value itself."""
    normalized = evidence_body(text).lower()
    if re.search(r"受電|引込|incoming|service|高圧受電", normalized, re.I):
        return "RECEIVING_VOLTAGE"
    if re.search(r"低圧母線|母線|bus", normalized, re.I):
        return "BUS_VOLTAGE"
    if re.search(r"配電|distribution", normalized, re.I):
        return "DISTRIBUTION_VOLTAGE"
    if re.search(r"二次|secondary|出力|output", normalized, re.I):
        return "SECONDARY_VOLTAGE"
    if re.search(r"制御|control", normalized, re.I):
        return "CONTROL_VOLTAGE"
    return "GENERIC_VOLTAGE"


def _voltage_roles_compatible(facet: Facet, item: Evidence) -> bool:
    requested = _voltage_role(facet.query)
    actual = _voltage_role(item.quote)
    if requested == "GENERIC_VOLTAGE":
        return True
    return actual == requested


def _can_support_facet(facet: Facet, item: Evidence, evidence: list[Evidence]) -> bool:
    if facet.name == "電圧":
        if not valid_voltage_expressions(evidence_body(item.quote)):
            return False
        if not _voltage_roles_compatible(facet, item):
            return False
    status = grounding_status_for(facet, item, evidence)
    if get_settings().qa_vision_typed_policy == "balanced":
        return status in {"TEXT_LAYER", "VISION_CORROBORATED"}
    return status == "TEXT_LAYER"


def _atomic_facet_spans(facet: Facet, item: Evidence) -> list[str]:
    """Return only the requested atomic fact, never an adjacent OCR qualifier."""
    body = evidence_body(item.quote)
    if facet.name == "電圧":
        return valid_voltage_expressions(body)
    if facet.name == "容量":
        return list(dict.fromkeys(re.findall(r"\d+(?:[.,]\d+)?\s*kVA\b", body, re.I)))
    return evidence_spans(facet, item)


def answer_plan(question: str) -> list[dict]:
    """Create a deterministic evidence requirement plan before generation."""
    return [
        {"facet": facet.name, "expected_type": FACET_EVIDENCE_TYPES.get(facet.name, "generic_text"), "required": True}
        for facet in extract_facets(question)
    ]


def _wants_structured_fact(question: str) -> bool:
    normalized = question.lower()
    return bool(re.search(r"\d|[0-9０-９]|\b(?:gl|fl|el|kv|kva|mm|cm|m3)\b", normalized) or any(
        term.lower() in normalized for term in QA_STRUCTURED_TERMS
    ))


def _structured_evidence_score(text: str) -> float:
    score = 0.0
    if re.search(r"\d+(?:[.,]\d+)?\s*(?:mm|cm|m3|m²|㎡|kva|kv|v|a|w|hz|台|個|本|人|日|年|円|階)", text, re.I):
        score += 1.25
    if re.search(r"\b(?:GL|FL|EL|No\.?|P[- ]?\d+)\b", text, re.I):
        score += 0.75
    if re.search(r"(?:室|ポンプ|盤|槽|機|器|配管|ダクト|給水|排水|換気|空調)", text):
        score += 0.5
    return score


def _qa_terms(question: str) -> list[str]:
    parsed = parse_query(question)
    terms = []
    for term in parsed.positive_terms:
        if term in QA_STOP_TERMS or len(term) < 2:
            continue
        if len(term) == 2 and term not in QA_SHORT_TERMS:
            continue
        # The query parser intentionally emits Japanese bigrams for lexical
        # search. QA reranking should use meaningful nouns, not fragments such
        # as "上パ" or polite-form suffixes.
        if not re.search(r"[一-龥A-Za-z0-9]", term):
            continue
        if term[-1] in "ぁあかがさざただなはばぱまやらわをんっゃゅょ":
            continue
        if term not in terms:
            terms.append(term)
    return terms


def _floor_mentions(text: str) -> set[str]:
    """Extract concrete floors without treating common/global as a conflict."""
    normalized = text.replace("Ｆ", "F")
    mentions = set()
    mentions.update(f"B{number}" for number in re.findall(r"(?<![A-Za-z])B\s*([0-9]+)(?:F)?", normalized, re.I))
    for basement, number in re.findall(r"(地下\s*)?([0-9]+)\s*(?:階|F)", normalized, re.I):
        mentions.add(("B" if basement else "") + number + ("" if basement else "F"))
    return mentions


def _inferred_floor_score(row: dict, floors: list[str]) -> tuple[float, list[str]]:
    """Score inferred QA floors softly; explicit UI filters remain hard filters."""
    if not floors:
        return 0.0, []
    metadata = " ".join((str(row.get("floor", "")), str(row.get("title", ""))))
    body = str(row.get("snippet", ""))
    score, reasons = 0.0, []
    for floor in floors:
        wanted = floor.upper().replace("階", "F")
        metadata_mentions = _floor_mentions(metadata)
        body_mentions = _floor_mentions(body)
        if wanted in metadata_mentions:
            score += 2.5
            reasons.append(f"qa inferred floor metadata={wanted}")
        elif wanted in body_mentions:
            score += 1.5
            reasons.append(f"qa inferred floor text={wanted}")
        elif (metadata_mentions | body_mentions):
            score -= 3.0
            reasons.append(f"qa inferred floor conflict={wanted}")
        else:
            reasons.append(f"qa inferred floor common={wanted}")
    return score, reasons


def _numeric_role_score(question: str, snippet: str) -> tuple[float, list[str]]:
    """Prefer evidence with the requested numeric semantic role."""
    requested = {facet.name for facet in extract_facets(question)}
    if not requested:
        return 0.0, []
    roles = {role for _value, _unit, role in numeric_facts(snippet)}
    score, reasons = 0.0, []
    for facet, role in (("電圧", "kv"), ("容量", "kva")):
        if facet not in requested:
            continue
        if role in roles:
            score += 3.0
            reasons.append(f"numeric role={role}")
        elif roles:
            score -= 3.0
            reasons.append(f"numeric role mismatch={role}")
    return score, reasons


def _rerank_rows(question: str, rows: list[dict], limit: int) -> list[dict]:
    terms = _qa_terms(question)
    wants_structured = _wants_structured_fact(question)
    inferred_floors = parse_query(question).floors
    ranked = []
    for original_rank, row in enumerate(rows):
        title = str(row.get("title", ""))
        snippet = str(row.get("snippet", ""))
        haystack = " ".join(
            (str(row.get("drawing_number", "")), title, str(row.get("floor", "")), snippet)
        )
        title_hits = [term for term in terms if term in title]
        text_hits = [term for term in terms if term in haystack]
        score = float(row.get("score", 0.0)) * 0.12
        score += len(title_hits) * 3.0 + len(text_hits) * 1.25
        # Quality is persisted during ingestion. Old rows get no free boost;
        # their fallback score comes from the chunk-level candidate path.
        if QA_RERANK_OPTIONS.quality_boost:
            # Keep this deliberately subordinate to query relevance.
            score += max(0.0, min(1.0, float(row.get("quality_score", 0.0)))) * 0.35
        score += max(0.0, float(row.get("vector_score", 0.0))) * 0.8
        if wants_structured and QA_RERANK_OPTIONS.numeric_entity_boost:
            score += _structured_evidence_score(snippet)
        floor_score, floor_reasons = _inferred_floor_score(row, inferred_floors)
        numeric_score, numeric_reasons = _numeric_role_score(question, snippet)
        score += floor_score + numeric_score
        if row.get("evidence_kind") == "pdf_chunk":
            score += 0.5
        enriched = dict(row)
        enriched["qa_match_terms"] = list(dict.fromkeys(text_hits))
        enriched["qa_relevance_reasons"] = floor_reasons + numeric_reasons
        enriched["qa_score"] = round(score, 4)
        ranked.append((score, original_rank, enriched))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    ranked_rows = [item[2] for item in ranked]
    if not QA_RERANK_OPTIONS.duplicate_suppression:
        return ranked_rows[:limit]
    return _suppress_duplicates(ranked_rows, limit)


def _suppress_duplicates(rows: list[dict], limit: int) -> list[dict]:
    """Keep primary evidence diverse with soft repeat penalties.

    A small unseen-document bonus is only applied when the chunks have already
    passed lexical/semantic ranking; it does not pull unrelated drawings into
    the context merely to satisfy a diversity quota. Repeat penalties are soft
    so a clearly stronger fourth chunk can still be selected.
    """
    settings = get_settings()
    selected: list[dict] = []
    pages: dict[tuple[object, object], int] = {}
    document_counts: dict[object, int] = {}
    documents_seen: set[object] = set()
    remaining = list(rows)
    while remaining and len(selected) < limit:
        eligible = remaining
        if QA_RERANK_OPTIONS.document_diversity and len(documents_seen) < min(4, limit):
            unseen = [
                row for row in remaining
                if (row.get("document_id") or row.get("drawing_number")) not in documents_seen
            ]
            if unseen:
                best_score = max(float(row.get("qa_score", 0.0)) for row in remaining)
                diversified = [
                    row for row in unseen
                    if float(row.get("qa_score", 0.0)) >= best_score * 0.10
                ]
                if diversified:
                    eligible = diversified
        choice = max(
            eligible,
            key=lambda row: (
                float(row.get("qa_score", 0.0))
                + (0.75 if QA_RERANK_OPTIONS.document_diversity and (row.get("document_id") or row.get("drawing_number")) not in documents_seen else 0.0)
                - (getattr(settings, "qa_page_repeat_penalty", 0.35) * pages.get((row.get("document_id") or row.get("drawing_number"), row.get("page_id") or row.get("page")), 0) if QA_RERANK_OPTIONS.page_cap else 0.0)
                - (getattr(settings, "qa_document_repeat_penalty", 0.45) * document_counts.get(row.get("document_id") or row.get("drawing_number"), 0) if QA_RERANK_OPTIONS.document_cap else 0.0),
                -int(row.get("chunk_id") or 0),
            ),
        )
        remaining.remove(choice)
        document_key = choice.get("document_id") or choice.get("drawing_number")
        key = (document_key, choice.get("page_id") or choice.get("page"))
        pages[key] = pages.get(key, 0) + 1
        document_counts[document_key] = document_counts.get(document_key, 0) + 1
        documents_seen.add(document_key)
        selected.append(choice)
    return selected


def _merge_candidates(search_rows: list[dict], chunk_rows: list[dict]) -> list[dict]:
    """Prefer direct chunk data, retaining a document search vector score."""
    merged: dict[object, dict] = {}
    for row in search_rows + chunk_rows:
        key = row.get("chunk_id") or (row.get("drawing_number"), row.get("page"), row.get("evidence_kind"))
        current = merged.get(key)
        if current is None:
            merged[key] = dict(row)
            continue
        chosen = dict(current)
        if row.get("quality_score") is not None or row.get("qa_candidate_terms"):
            chosen.update(row)
        chosen["vector_score"] = max(float(current.get("vector_score", 0.0)), float(row.get("vector_score", 0.0)))
        chosen["score"] = max(float(current.get("score", 0.0)), float(row.get("score", 0.0)))
        merged[key] = chosen
    return list(merged.values())


def collect_evidence(
    question: str,
    limit: int = 20,
    *,
    floor: str | None = None,
    discipline: str | None = None,
    source_type: str | None = None,
    candidate_pool: list[Evidence] | None = None,
) -> tuple[list[Evidence], list[dict]]:
    # Retrieve a wider pool, then use QA-specific lexical/entity coverage to
    # reduce noise before assigning stable Evidence IDs.
    search_filters = {
        key: value
        for key, value in {
            "floor": floor,
            "discipline": discipline,
            "source_type": source_type,
        }.items()
        if value is not None
    }
    # Keep one shared query embedding in the normal document search, then add
    # richer chunk candidates without making a second embedding API request.
    _, candidates = search(question, max(limit * 6, 40), **search_filters)
    chunk_limit = max(limit * 32, 256)
    chunk_candidates = qa_evidence_candidates(
        question, chunk_limit, include_vector=False, **search_filters
    )
    # A conjunctive QA question can require separate documents (for example,
    # a method and a support layer). Retrieve each conservative sub-query from
    # the same lexical chunk index, then merge before the existing reranker.
    # This adds no embedding/API call and does not change ordinary search.
    for subquery in decompose_query(question):
        if subquery == question:
            continue
        chunk_candidates.extend(
            qa_evidence_candidates(
                subquery, chunk_limit, include_vector=False, **search_filters
            )
        )
    # Document retrieval remains the broad-recall stage. Chunk reranking may
    # replace a representative excerpt *within* those documents, but may not
    # introduce a lexically noisy document that the hybrid retriever rejected.
    # This keeps QA evidence enrichment from lowering document-level recall.
    retrieved_documents = {row.get("drawing_number") for row in candidates}
    if retrieved_documents:
        chunk_candidates = [
            row for row in chunk_candidates
            if row.get("drawing_number") in retrieved_documents
        ]
    ranked_rows = _rerank_rows(
        question,
        _merge_candidates(candidates, chunk_candidates),
        max(limit * 6, 48),
    )
    evidence = []
    for row in ranked_rows:
        if row.get("evidence_kind") != "pdf_chunk":
            continue
        quote = row["snippet"].strip()
        if quote:
            evidence.append(Evidence(f"E{len(evidence)+1}", row["drawing_number"], row["page"], quote,
                                     row.get("bbox"), row.get("document_id"), row.get("page_id"), row.get("chunk_id"),
                                     source_type=row.get("source_type", "drawing"),
                                     provenance=row.get("extraction_provenance", "text"),
                                     quality_score=float(row.get("quality_score", 0.0)),
                                     version=row.get("version") or row.get("revision"), title=row.get("title", ""),
                                     semantic_types=evidence_semantic_types(quote),
                                     grounding_status=row.get("grounding_status", "text_layer")))
    # Change orders are explicit documents, not citations inferred from their
    # mention inside a drawing. The stem is part of the corpus identity.
    for number in set(re.findall(r"(?:変更指示|No\.?)\s*(\d{4})", question)):
        for path in (get_settings().data_root / "changes").glob(f"{number}_*.md"):
            quote = path.read_text(encoding="utf-8")[:1800]
            evidence.append(Evidence(f"E{len(evidence)+1}", number, 1, quote, None, source_type="change_order",
                                     semantic_types=evidence_semantic_types(quote)))
    parsed = parse_query(question)
    manuals = []
    for path in (get_settings().data_root / "equipment-manuals").glob("*.md"):
        content = path.read_text(encoding="utf-8")
        score = sum(1 for term in parsed.positive_terms if len(term) >= 3 and term in content)
        if score >= 1:
            manuals.append((score, path, content))
    for _, path, content in sorted(manuals, key=lambda row: -row[0])[:2]:
        evidence.append(Evidence(f"E{len(evidence)+1}", path.stem, 1, content[:4000], None,
                                 source_type="manual", title=display_title("", path, path.stem),
                                 semantic_types=evidence_semantic_types(content)))
    # Explicit change-order sources are authoritative and must not be pushed
    # out by generic drawing candidates. Keep the remaining context compact.
    direct = [item for item in evidence if item.source_type == "change_order"]
    rest = [item for item in evidence if item.source_type != "change_order"]
    # Keep the established rank order for the primary context. Facet slots are
    # used by the existing bounded missing-facet recovery path, so they never
    # evict a broad-recall candidate from this first window.
    selected_evidence = direct + rest[: max(0, limit - len(direct))]
    if candidate_pool is not None:
        candidate_pool.extend(evidence)
    return selected_evidence[: max(limit, 8)], ranked_rows[: max(limit, 8)]


def _preserve_facet_evidence(question: str, evidence: list[Evidence], limit: int) -> list[Evidence]:
    """Reserve one supporting slot per requested facet before diversity trim."""
    selected: list[Evidence] = []
    seen: set[tuple[object, object, str]] = set()
    for step in answer_plan(question):
        facet = next((item for item in extract_facets(question) if item.name == step["facet"]), None)
        if facet is None:
            continue
        expected = step["expected_type"]
        candidate = next((item for item in evidence if evidence_spans(facet, item)
                          and (expected == "generic_text" or expected in item.semantic_types)), None)
        if candidate is not None:
            key = (candidate.document_id or candidate.drawing_number, candidate.chunk_id or candidate.page, candidate.quote)
            if key not in seen:
                selected.append(candidate)
                seen.add(key)
    for item in evidence:
        key = (item.document_id or item.drawing_number, item.chunk_id or item.page, item.quote)
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected[:limit]

def format_evidence(evidence: list[Evidence], max_chars: int = 24000) -> str:
    sections = []
    remaining = max_chars
    for item in evidence:
        header = (f'[{item.evidence_id}]\nsource: {item.source_type} {item.drawing_number}\npage: {item.page}\n'
                  f'<EVIDENCE id="{item.evidence_id}" source="{item.source_type}" '
                  f'drawing="{item.drawing_number}" page="{item.page}">\ntext:\n')
        content = item.quote[:max(0, remaining-len(header))]
        if not content:
            if not sections and remaining:
                sections.append(header[:remaining])
            break
        section = header + content + "\n</EVIDENCE>"
        sections.append(section)
        remaining -= len(section)
    return "\n\n".join(sections)

def resolve_evidence_ids(ids: list[str], evidence: list[Evidence]) -> list[Evidence]:
    mapping = {item.evidence_id: item for item in evidence}
    if any(evidence_id not in mapping for evidence_id in ids):
        raise ValueError("Unknown evidence ID in generated answer")
    return [mapping[evidence_id] for evidence_id in dict.fromkeys(ids)]

def _extractive_answer(question: str, evidence: list[Evidence]) -> tuple[str, list[str]]:
    direct_sources = [item for item in evidence if item.source_type == "change_order" and item.drawing_number in question]
    if direct_sources:
        # The requested change instruction is a primary source. Preserve its
        # facts together instead of dropping amounts or schedule conditions.
        relevant = []
        for item in direct_sources:
            lines = [line.strip(" -|#") for line in item.quote.splitlines()]
            facts = [line for line in lines if re.match(r"(?:変更前|変更後|変更箇所|コスト影響|工程影響)\s*[:：]", line)]
            relevant.extend(facts)
        return "。\n".join(relevant)[:3000], [item.evidence_id for item in direct_sources]
    facets = extract_facets(question)
    if len(facets) == 1 and facets[0].name in {"電圧", "容量", "台数", "数量", "人数", "定員", "積載量", "重量", "寸法"}:
        # A single typed fact should not be diluted with nearby but different
        # quantities. Preserve one attested supporting span and its citation.
        for item in evidence:
            spans = evidence_spans(facets[0], item)
            if spans and _can_support_facet(facets[0], item, evidence):
                fact = strip_citation_markers(spans[0]).strip().rstrip("。")
                # Evidence often starts with a room or diagram label. Put the
                # requested fact first without adding information not present.
                if facets[0].name not in fact:
                    fact = f"{facets[0].name}は、{fact}"
                return fact + "。", [item.evidence_id]
        return "取得できた根拠からは、求められた値を確定できません。", []
    parsed = parse_query(question)
    terms = _qa_terms(question)
    candidates = []
    for item in evidence:
        for sentence in re.split(r"[。\n]", evidence_body(item.quote)):
            sentence = sentence.strip(" -|:：")
            if len(sentence) < 5: continue
            score = sum(2 for term in terms if term in sentence)
            if item.source_type == "drawing":
                score += 1
            if score: candidates.append((score, item.evidence_id, sentence))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    chosen = []
    per_source: dict[str, int] = {}
    for candidate in candidates:
        if per_source.get(candidate[1], 0) >= 2:
            continue
        chosen.append(candidate)
        per_source[candidate[1]] = per_source.get(candidate[1], 0) + 1
        if len(chosen) >= 6:
            break
    if not chosen: return "根拠不足のため回答できません。", []
    # Verbatim evidence excerpts remain verifiable; no generated facts are added.
    answer = "。".join(row[2] for row in chosen)[:1500]
    ids = list(dict.fromkeys(row[1] for row in chosen))
    return answer, ids

def _fallback_payload(question: str, evidence: list[Evidence], results: list[dict]) -> dict:
    answer, ids = _extractive_answer(question, evidence)
    selected = resolve_evidence_ids(ids, evidence)
    return _build_response(answer, selected, results)


def _premise_validation(question: str, evidence: list[Evidence]) -> tuple[str, str, str] | None:
    """Return (subject, status) for a user-supplied numeric yes/no premise.

    A number in the question is a hypothesis, never evidence.  We only call a
    premise supported when the same typed value appears in source evidence.
    """
    match = re.search(
        r"(?P<subject>.+?)(?:は|の(?:容量)?は)\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>kVA|kV|kW|V|台|基|人|kg|mm|cm|m|戸|室)\s*(?:ありますか|ですか|ありますでしょうか)\s*[？?]?$",
        question.strip(), re.I,
    )
    if not match:
        return None
    unit = match.group("unit").lower()
    facet_name = {
        "台": "台数", "基": "数量", "人": "人数", "kg": "重量", "mm": "寸法",
        "cm": "寸法", "m": "寸法", "戸": "戸数", "室": "部屋数",
        "kv": "電圧", "v": "電圧", "kva": "容量", "kw": "容量",
    }.get(unit)
    if facet_name is None:
        return None
    subject = match.group("subject").strip(" の")
    facet = Facet(facet_name, f"{subject} {facet_name}", subject)
    expected = f"{match.group('value').replace(',', '')}{unit}"
    supported_values = set()
    for item in evidence:
        if facet.name in TYPED_GROUNDING_FACETS and not _can_support_facet(facet, item, evidence):
            continue
        for span in evidence_spans(facet, item):
            for value, found_unit, _role in numeric_facts(span):
                if found_unit.lower() == unit:
                    supported_values.add(f"{value}{unit}")
    return subject, ("supported" if expected in supported_values else ("contradicted" if supported_values else "unknown")), match.group("value") + match.group("unit")


def _build_response(answer: str, selected: list[Evidence], results: list[dict]) -> dict:
    unique_selected = list({item.evidence_id: item for item in selected}.values())
    citations = [dict(evidence_id=item.evidence_id, drawing_number=item.drawing_number,
                      document_id=item.document_id, page_id=item.page_id, chunk_id=item.chunk_id,
                      page=item.page, quote=item.quote, bbox=item.bbox,
                      citation_scope="region" if item.bbox else "page",
                      source_type=item.source_type, provenance=item.provenance,
                      quality_score=item.quality_score, version=item.version,
                      title=item.title, semantic_types=list(item.semantic_types or evidence_semantic_types(item.quote))) for item in unique_selected]
    return {"answer": answer, "citations": citations,
            "confidence": 0.65 if citations else 0.0, "evidence": results}


def _claim_terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[一-龥々ー]{2,}|[A-Za-z0-9][A-Za-z0-9,./+ー-]*", text)
        if token not in QA_STOP_TERMS
    ]


def _generated_ids(generated, evidence: list[Evidence]) -> list[str]:
    claims = getattr(generated, "claims", None) or []
    if not claims:
        return list(dict.fromkeys(getattr(generated, "evidence_ids", [])))
    mapping = {item.evidence_id: item for item in evidence}
    selected: list[str] = []
    for claim in claims:
        terms = _claim_terms(claim.text)
        scored = []
        for order, evidence_id in enumerate(claim.evidence_ids):
            item = mapping.get(evidence_id)
            if item is None:
                scored.append((0, order, evidence_id))
                continue
            quote = item.quote
            support = sum(3 if len(term) >= 4 else 1 for term in terms if term in quote)
            scored.append((support, order, evidence_id))
        supported = [row for row in scored if row[0] > 0]
        if supported or not QA_RERANK_OPTIONS.deterministic_backstop:
            chosen = supported
        else:
            # If the model supplied valid IDs but none actually supports the
            # claim, select the strongest available Evidence deterministically
            # instead of preserving a plausible-looking wrong citation.
            all_supported = []
            for evidence_order, item in enumerate(evidence):
                support = sum(
                    3 if len(term) >= 4 else 1
                    for term in terms
                    if term in item.quote
                )
                if support:
                    all_supported.append((support, -evidence_order, item.evidence_id))
            if all_supported:
                best_support, negative_order, best_id = max(all_supported)
                chosen = [(best_support, -negative_order, best_id)]
            else:
                chosen = scored
        # A claim cites its strongest direct support. Cross-document questions
        # retain multiple citations through their separate claims/facets.
        for _support, _order, evidence_id in sorted(
            chosen, key=lambda row: (-row[0], row[1])
        )[:3]:
            selected.append(evidence_id)
    return list(dict.fromkeys(selected))


def _answer_from_evidence(question, evidence, results, answerer):
    """Generation and existing citation backstop, shared by complete/partial QA."""
    if answerer:
        try:
            generated = answerer.answer(question, format_evidence(evidence))
            selected = resolve_evidence_ids(_generated_ids(generated, evidence), evidence)
            answer = generated.answer or "。".join(
                claim.text for claim in (getattr(generated, "claims", None) or [])
            )
            if selected:
                cited_text = "\n".join(item.quote for item in selected)
                numbers = set(re.findall(r"\d+(?:[,.]\d+)*", strip_citation_markers(answer)))
                if all(number in cited_text for number in numbers) and numeric_facts(answer) <= numeric_facts(cited_text):
                    return _build_response(answer, selected, results)
                # One unsafe quantity should not replace all valid claims with
                # six broadly related excerpts (and inflate citation count).
                # Retain only independently verified claims from the same
                # model response; this needs no second generation request.
                verified, verified_evidence = [], []
                for claim in (getattr(generated, "claims", None) or []):
                    ids = _generated_ids(SimpleNamespace(claims=[claim]), evidence)
                    supporting = resolve_evidence_ids(ids, evidence)
                    text = "\n".join(item.quote for item in supporting)
                    clean_claim = strip_citation_markers(claim.text).strip()
                    claim_numbers = set(re.findall(r"\d+(?:[,.]\d+)*", clean_claim))
                    if supporting and all(n in text for n in claim_numbers) and numeric_facts(clean_claim) <= numeric_facts(text):
                        verified.append(clean_claim + "".join(f"[{eid}]" for eid in ids))
                        verified_evidence.extend(supporting)
                if verified:
                    logger.info("Dropped unsupported numeric claims; retained independently verified claims")
                    unique = list({item.evidence_id: item for item in verified_evidence}.values())
                    requested = extract_facets(question)
                    warning = "" if requested else "\n一部の数値は取得できた根拠から確定できません。"
                    return _build_response("\n".join(verified) + warning, unique, results)
                logger.warning("Generated answer contained unsupported typed numeric facts; using fallback")
        except Exception as exc:
            logger.warning("Grounded generation unavailable or invalid; using evidence fallback: %s", exc)
    return _fallback_payload(question, evidence, results)


def _prepare_facet_evidence(question, facets, evidence, pool, filters):
    """Recheck the already retrieved pool, then at most one lexical lookup.

    Ordinary search weights, embeddings and the original selected evidence are
    untouched. At most two supplemental chunks fit within the 24k context.
    """
    added = []
    missing = []
    for facet in facets:
        if any(evidence_spans(facet, item) for item in evidence + added):
            continue
        match = next((item for item in pool if evidence_spans(facet, item)), None)
        if match and len(added) < 2:
            added.append(match)
        else:
            missing.append(facet)
    fallback_query = None
    if missing:
        fallback_query = " ".join(dict.fromkeys(facet.query for facet in missing))
        rows = qa_evidence_candidates(fallback_query, 40, include_vector=False, **filters)
        existing = {item.evidence_id for item in pool + evidence}
        next_id = max((int(eid[1:]) for eid in existing), default=0) + 1
        for row in rows:
            if len(added) >= 2:
                break
            if not any((not facet.source_id or facet.source_id == row["drawing_number"]) and supporting_spans(facet, row.get("snippet", "")) for facet in missing):
                continue
            if any(item.chunk_id == row.get("chunk_id") and item.drawing_number == row["drawing_number"] for item in evidence + added):
                continue
            added.append(Evidence(f"E{next_id}", row["drawing_number"], row.get("page", 1),
                                  row["snippet"], row.get("bbox"), row.get("document_id"),
                                  row.get("page_id"), row.get("chunk_id"), row.get("source_type", "pdf"),
                                  row.get("extraction_provenance", "text"), float(row.get("quality_score", 0)),
                                  row.get("version") or row.get("revision"), row.get("title", ""),
                                  grounding_status=row.get("grounding_status", "text_layer")))
            next_id += 1
    # Never cite a supplemental chunk that is truncated out of the LLM context.
    selected = list(evidence)
    for item in added:
        if len(format_evidence(selected, 100000)) + len(item.quote) + 160 <= 24000:
            selected.append(item)
    return selected, fallback_query


def _transformer_capacity_spans(evidence: list[Evidence]) -> list[tuple[Evidence, list[str]]]:
    """Return well-formed kVA evidence explicitly tied to transformers.

    This is post-retrieval claim validation. It intentionally leaves the
    original evidence pool untouched so unrelated capacity candidates remain
    available for the request types they can actually support.
    """
    valid_capacity = re.compile(r"(?<![/A-Za-z0-9])\d+(?:[.,]\d+)?\s*kVA\b", re.I)
    transformer = re.compile(r"(?:TR[-−－]?\d+|変圧器)", re.I)
    incompatible = re.compile(r"(?:発電機|発電設備|非常用発電|\bDG\b|\bUPS\b)", re.I)
    selected = []
    for item in evidence:
        body = evidence_body(item.quote)
        facet = Facet("容量", "変圧器 容量")
        if (not transformer.search(body) or incompatible.search(body)
                or not _can_support_facet(facet, item, evidence)):
            continue
        spans = [span for span in supporting_spans(facet, item.quote)
                 if valid_capacity.search(span)]
        if spans:
            selected.append((item, spans))
    return selected


def _complete_facets(question, payload, facets, evidence):
    """Keep supported generated sentences; repair gaps with bounded excerpts.

    Unknown counts remain unknown, even when capacity or a drawing-title range
    is available. Diagnostics distinguish an explicit abstention from coverage.
    """
    answer = payload["answer"]
    coverage = validate_coverage(facets, answer, evidence)
    parts, selected = [], []
    for facet, check in zip(facets, coverage):
        transformer_capacity = facet.name == "容量" and "変圧器" in question
        if transformer_capacity:
            # For a transformer-capacity request, use attested transformer
            # labels only. This prevents an unrelated generator kVA value (or
            # a malformed OCR token) from filling the requested facet.
            choices = _transformer_capacity_spans(evidence)[:2]
            ids = [item.evidence_id for item, _ in choices]
            snippets = [span for _, spans in choices for span in spans]
        elif facet.name in TYPED_GROUNDING_FACETS:
            # Rebuild typed claims from source spans rather than preserving a
            # generated sentence. This makes requested-facet output atomic
            # (for example, 6.6kV without an unrequested parenthetical).
            choices = [(item, _atomic_facet_spans(facet, item)) for item in evidence
                       if _can_support_facet(facet, item, evidence)]
            choices = [(item, spans) for item, spans in choices if spans and valid_numeric_tokens(" ".join(spans))]
            ids = [item.evidence_id for item, _ in choices[:1]]
            snippets = [span for _, spans in choices[:1] for span in spans[:1]]
        else:
            candidates = [item for item in evidence if _can_support_facet(facet, item, evidence)]
            snippets = supporting_spans(facet, answer) if check["supported"] and candidates else []
            ids = [eid for eid in check["evidence_ids"] if any(item.evidence_id == eid for item in candidates)]
        if snippets:
            # Revalidate each sentence; a supported sentence must not license a
            # second, unsupported number or an unrelated capacity statement.
            snippets = [s for s in snippets if validate_coverage([facet], s, evidence)[0]["supported"]]
        if not snippets:
            choices = [(item, _atomic_facet_spans(facet, item) if facet.name in TYPED_GROUNDING_FACETS else evidence_spans(facet, item)) for item in evidence
                       if _can_support_facet(facet, item, evidence)]
            choices = [(item, spans) for item, spans in choices if spans]
            ids = [item.evidence_id for item, _ in choices[:2]]
            snippets = [span for _, spans in choices[:2] for span in spans]
        if snippets:
            clean = [strip_citation_markers(s).strip() for s in snippets]
            clean = [re.sub(r"^([A-Za-z][A-Za-z0-9]*-\d+)\(([^()]+用)\)$", r"\1は\2として記載されています", s) for s in clean]
            # A validated clause may end at a Japanese comma; finish common
            # connective endings without introducing any new fact.
            clean = [re.sub(r"であり$", "です", re.sub(r"ており$", "ています", s)) for s in clean]
            prefix = "" if any(facet.name in s for s in clean) else f"{facet.name}は、"
            parts.append(prefix + "。".join(dict.fromkeys(clean)) + "。" + "".join(f"[{eid}]" for eid in ids))
            selected.extend(resolve_evidence_ids(ids, evidence))
        else:
            parts.append(f"{facet.name}は、取得できた根拠からは確定できません。")
    unique = list({item.evidence_id: item for item in selected}.values())
    result = _build_response("\n".join(parts), unique, payload["evidence"])
    result["facet_coverage"] = validate_coverage(facets, result["answer"], unique)
    result["facet_coverage_score"] = sum(c["supported"] for c in result["facet_coverage"]) / len(facets)
    return result


def direct_answer_coverage(question: str, answer: str, evidence: list[Evidence]) -> float:
    """Share of requested facets resolved or explicitly declined up front.

    The first paragraph is the field user's direct-answer block. This is a
    diagnostic only: it never changes retrieval, evidence selection, or UI.
    """
    facets = extract_facets(question)
    if not facets:
        return 1.0 if answer.strip() else 0.0
    direct_block = answer.split("\n\n", 1)[0]
    coverage = validate_coverage(facets, direct_block, evidence)
    resolved = 0
    for facet, check in zip(facets, coverage):
        abstained = bool(re.search(
            rf"{re.escape(facet.name)}[^。\n]*?(?:確定できません|根拠不足|回答できません)", direct_block
        ))
        direct_text_support = False
        if facet.name not in {"台数", "数量", "人数", "定員", "積載量", "重量", "寸法", "階数", "部屋数", "戸数", "電圧", "容量"}:
            # Non-numeric requirements (for example change descriptions)
            # often omit the field label in a natural direct answer. Require
            # a substantive verbatim phrase from supported evidence instead.
            for item in evidence:
                for span in evidence_spans(facet, item):
                    phrases = re.findall(r"[一-龥ァ-ヶA-Za-z0-9ー（）()]{4,}", evidence_body(span))
                    if any(phrase not in {facet.name, facet.query} and phrase in direct_block for phrase in phrases):
                        direct_text_support = True
                        break
                if direct_text_support:
                    break
        resolved += bool(check["supported"] or direct_text_support or abstained)
    return resolved / len(facets)


def _complete_type_inventory(question, payload, evidence):
    """Do not omit explicit structured types when a type inventory is asked.

    Unit grammars, not gold labels or drawing IDs, identify values. A type may
    be attested without a reliable A/B/C-to-type mapping, so the supplement
    reports notation only and never infers that mapping or dwelling counts.
    """
    grammars = {"間取り": r"(?<![A-Za-z0-9])\d+(?:S?LDK|DK|K|R)(?![A-Za-z0-9])"}
    if not any(word in question for word in ("種類", "構成", "タイプ", "一覧")):
        return payload
    added = []
    answer = payload["answer"]
    for name, pattern in grammars.items():
        if name not in question:
            continue
        existing = set(re.findall(pattern, answer, re.I))
        for item in evidence:
            values = list(dict.fromkeys(re.findall(pattern, evidence_body(item.quote), re.I)))
            missing = [value for value in values if value not in existing]
            if missing:
                answer += f"\n{name}の表記として、{'、'.join(missing)}も確認できます。[{item.evidence_id}]"
                existing.update(missing)
                added.append(item)
    # The same omission can affect named equipment in a types/list question.
    # Reuse the existing parser vocabulary and require a verbatim occurrence
    # in the selected evidence. Report notation, not inferred installation or
    # an unverified relationship. No new retrieval or answer-model call.
    terms = [term for term in parse_query(question).positive_terms if len(term) >= 3]
    for item in evidence:
        body = evidence_body(item.quote)
        missing = [term for term in terms if term in body and term not in answer]
        # Longer expressions subsume their shorter lexical fragments.
        missing = [term for term in missing if not any(term != other and term in other for other in missing)]
        if missing:
            answer += f"\n資料には「{'、'.join(missing[:3])}」という関連表記もあります。[{item.evidence_id}]"
            added.append(item)
    if not added:
        return payload
    ids = {c["evidence_id"] for c in payload["citations"]} | {item.evidence_id for item in added}
    citations = _build_response(answer, [item for item in evidence if item.evidence_id in ids], payload["evidence"])["citations"]
    return {**payload, "answer": answer, "citations": citations}


def _citation_key(item: Evidence) -> tuple[object, ...]:
    """Deduplicate only the displayed citation identity, never QA evidence."""
    return (item.document_id or item.drawing_number, item.version, item.page,
            item.chunk_id or item.evidence_id)


def _atomic_answer_claims(answer: str) -> list[str]:
    text = strip_citation_markers(answer)
    return [part.strip() for part in re.split(r"[。\n、]+", text)
            if part.strip() and "確定できません" not in part and "確認できません" not in part]


def _direct_claim_score(claim: str, item: Evidence, question: str, evidence: list[Evidence]) -> float:
    """Score direct textual support for *display* citations only.

    The broader evidence list is deliberately untouched. Numeric claims require
    the same value in the source, and typed evidence remains subject to the
    existing provenance/contradiction validation.
    """
    body = evidence_body(item.quote)
    terms = _claim_terms(claim)
    if not terms:
        return 0.0
    matched = [term for term in terms if term in body]
    numeric = set(re.findall(r"\d+(?:[,.]\d+)?\s*(?:kVA|kV|kW|V|台|基|人|kg|mm|cm|m)\b", claim, re.I))
    if numeric and not all(token.replace(" ", "") in body.replace(" ", "") for token in numeric):
        return 0.0
    facets = extract_facets(question)
    typed = [facet for facet in facets if facet.name in TYPED_GROUNDING_FACETS]
    claim_typed = [facet for facet in typed if facet.name in claim]
    if claim_typed and not any(evidence_spans(facet, item) for facet in claim_typed):
        return 0.0
    if typed and any(evidence_spans(facet, item) for facet in typed):
        if not any(_can_support_facet(facet, item, evidence) for facet in typed):
            return 0.0
    if not matched:
        return 0.0
    provenance = {"TEXT_LAYER": 0.35, "VISION_CORROBORATED": 0.2}.get(
        grounding_status_for(typed[0], item, evidence) if typed else "TEXT_LAYER", 0.0
    )
    return sum(3 if len(term) >= 4 else 1 for term in matched) + provenance


def select_display_citations(question: str, payload: dict, evidence: list[Evidence]) -> dict:
    """Select minimal, claim-backed citations after answer validation.

    This is intentionally post-validation: it never changes retrieval,
    candidate evidence, prompt context, or the answer text.
    """
    claims = _atomic_answer_claims(payload.get("answer", ""))
    if not claims:
        return {**payload, "citations": []}
    selected: list[Evidence] = []
    used: set[tuple[object, ...]] = set()
    for claim in claims:
        candidates = []
        for order, item in enumerate(evidence):
            if grounding_status_for(next((f for f in extract_facets(question) if f.name in TYPED_GROUNDING_FACETS), Facet("text", "text")), item, evidence) == "CONTRADICTED":
                continue
            score = _direct_claim_score(claim, item, question, evidence)
            if score:
                candidates.append((score, -order, item))
        if candidates:
            item = max(candidates, key=lambda row: (row[0], row[1]))[2]
            key = _citation_key(item)
            if key not in used:
                selected.append(item)
                used.add(key)
    return {**payload, "citations": _build_response(payload["answer"], selected, payload["evidence"])["citations"]}


def _unsupported_relation(answer: str, evidence: list[Evidence]) -> bool:
    """Reject asserted location/use relations without one direct source span.

    This is deliberately a post-generation validation gate.  It does not alter
    retrieval and never infers a relationship by joining an entity from one
    evidence item to a location from another.
    """
    plain = strip_citation_markers(answer)
    relations = re.findall(
        r"([^。\n]{1,48}?)(?:は|が)([^。\n]{1,48}?)(?:に|へ)(?:設置|配置|設け|収容)(?:され|されて|されます|されてい)",
        plain,
    )
    for entity, target in relations:
        entity = entity.strip(" 、")
        target = target.strip(" 、")
        if not entity or not target:
            continue
        if not any(
            entity in evidence_body(item.quote)
            and target in evidence_body(item.quote)
            and re.search(r"(?:設置|配置|設け|収容)", evidence_body(item.quote))
            for item in evidence
        ):
            return True
    return False


def answer_question(
    question: str,
    limit: int = 10,
    answerer=None,
    *,
    floor: str | None = None,
    discipline: str | None = None,
    source_type: str | None = None,
) -> dict:
    filters = {
        key: value
        for key, value in {
            "floor": floor,
            "discipline": discipline,
            "source_type": source_type,
        }.items()
        if value is not None
    }
    facets = extract_facets(question)
    plan = answer_plan(question)
    # Completeness enforcement is conservative: explicit multi-attribute
    # requests and typed count/capacity requests only; other QA remains intact.
    enforce = len(facets) > 1 or any(f.name in {"台数", "人数", "定員", "積載量", "重量", "部屋数", "階数"} for f in facets)
    pool = []
    if enforce:
        evidence, results = collect_evidence(question, limit, candidate_pool=pool, **filters)
        evidence, fallback_query = _prepare_facet_evidence(question, facets, evidence, pool, filters)
    else:
        evidence, results = collect_evidence(question, limit, **filters)
        fallback_query = None
    query_terms = _qa_terms(question)
    specific_terms = {term for term in query_terms if len(term) >= 3}
    if specific_terms:
        has_ranked_match = not results or any(
            specific_terms.intersection(row.get("qa_match_terms", [])) for row in results
        )
    else:
        has_ranked_match = not results or any(row.get("qa_match_terms") for row in results)
    has_explicit_source = any(item.source_type in {"change_order", "manual"} for item in evidence)
    if not evidence or not query_terms or (not has_ranked_match and not has_explicit_source):
        return {"answer": "根拠不足のため回答できません。", "citations": [], "confidence": 0.0, "evidence": results, "answer_plan": plan}
    premise = _premise_validation(question, evidence)
    if premise and premise[1] != "supported":
        subject, _status, value = premise
        return {
            "answer": f"{subject}が{value}あるとは、取得できた根拠から確認できません。",
            "citations": [], "confidence": 0.0, "evidence": results,
            "answer_plan": plan, "premise_status": _status,
            "direct_answer_coverage": 1.0,
        }
    if answerer is None and get_settings().gemini_api_key:
        from app.ai.gemini import GeminiAnswerer
        answerer = GeminiAnswerer()
    payload = _answer_from_evidence(question, evidence, results, answerer)
    if _unsupported_relation(payload.get("answer", ""), evidence):
        return {
            "answer": "取得できた根拠からは、質問の関係を確定できません。",
            "citations": [], "confidence": 0.0, "evidence": results,
            "answer_plan": plan, "relation_status": "unsupported",
            "direct_answer_coverage": 1.0,
        }
    # Voltage roles (receiving / bus / distribution, etc.) are semantic, so
    # validate them even when the source is a trusted PDF text layer.
    needs_grounding_validation = any(f.name == "電圧" for f in facets) or (
        any(f.name in TYPED_GROUNDING_FACETS for f in facets)
        and any(item.grounding_status != "text_layer" for item in evidence)
    )
    if enforce or needs_grounding_validation:
        payload = _complete_facets(question, payload, facets, evidence)
        if enforce:
            payload["facet_fallback_query"] = fallback_query
    completed = _complete_type_inventory(question, payload, evidence)
    completed = select_display_citations(question, completed, evidence)
    return {
        **completed,
        "answer_plan": plan,
        "direct_answer_coverage": direct_answer_coverage(question, completed["answer"], evidence),
    }

def validate_citations(payload: dict, evidence: list[Evidence] | None = None) -> bool:
    citations = payload.get("citations", [])
    if evidence is None or not citations:
        return False
    mapping = {item.evidence_id: item for item in evidence}
    return all((item := mapping.get(c.get("evidence_id"))) is not None
               and c.get("drawing_number") == item.drawing_number
               and c.get("page") == item.page
               and c.get("quote") == item.quote
               and c.get("source_type", "drawing") == item.source_type for c in citations)
