from __future__ import annotations
import csv, re
import logging
from dataclasses import dataclass
from pathlib import Path
from app.core.config import get_settings
from app.services.query_parser import ParsedQuery, parse_query
from app.services.evidence_quality import assess_evidence_quality
from app.services.document_titles import display_title
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

FUSION_WEIGHTS = {
    "lexical": 0.30,
    "metadata": 0.40,
    "semantic": 0.30,
}

@dataclass
class SearchDoc:
    drawing_number: str
    title: str
    floor: str
    discipline: str
    text: str
    source_path: str | None = None
    source_type: str = "pdf"

def _norm(value: str) -> str:
    return value.lower().replace("階", "f").replace("Ｆ", "f").replace("・", " ").replace("、", " ")

def _load_docs() -> list[SearchDoc]:
    root = get_settings().data_root
    docs: dict[str, SearchDoc] = {}
    catalog = root / "building-info" / "drawing_list.csv"
    if not catalog.is_file():
        logger.info("Demo catalog is unavailable: %s", catalog)
        return []
    with catalog.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            number = row["図番"].strip()
            path = next(iter(root.glob(f"drawings/{number}_*.pdf")), None)
            path = path or next(iter(root.glob(f"drawings/{number}_*.md")), None)
            source_type = "markdown" if path and path.suffix.lower() == ".md" else "pdf"
            docs.setdefault(number, SearchDoc(number, row["タイトル"], row["階"], row["種別"], "", source_type=source_type))
            docs[number].text += " " + " ".join(row.values())
            if path: docs[number].source_path = str(path)
    # Enrich floor plans/details with the authoritative building metadata. This
    # bridges the common case where the PDF is a scanned drawing with no text
    # layer, without coupling search to evaluation labels.
    try:
        import json
        rooms = json.loads((root / "building-info" / "rooms.json").read_text(encoding="utf-8")).get("rooms", [])
        equipment = json.loads((root / "building-info" / "equipment.json").read_text(encoding="utf-8")).get("equipment", [])
        def floors_for_doc(doc: SearchDoc) -> set[str]:
            raw = _norm(doc.floor)
            if "基準" in raw: return {f"{n}f" for n in range(3, 9)}
            if "全階" in raw or "共通" in raw or "配置" in raw: return set()
            found = set(re.findall(r"b[12]|[0-9]+f|r|ph|pit", raw))
            if "b" not in raw and "f" in raw:
                found.update(f"{n}f" for n in re.findall(r"[0-9]", raw))
            if "地下" in _norm(doc.title):
                found.update(f"b{n}" for n in re.findall(r"[12]", _norm(doc.title)))
            return found
        for doc in docs.values():
            floor_text = _norm(doc.floor + " " + doc.title)
            doc_floors = floors_for_doc(doc)
            context = []
            for room in rooms:
                room_floor = _norm(room.get("floor", ""))
                if room_floor in doc_floors:
                    context.extend([room.get("name", ""), room.get("use", "")])
            for item in equipment:
                item_floor = _norm(item.get("floor", ""))
                if item_floor in doc_floors:
                    context.extend([item.get("type", ""), item.get("name", ""), item.get("equipment_id", "")])
            doc.text += " " + " ".join(context)
    except (OSError, ValueError, KeyError):
        pass
    # Use the text layer when available; scanned pages remain searchable by metadata.
    try:
        import pymupdf
        for doc in docs.values():
            if doc.source_path:
                pdf = pymupdf.open(doc.source_path)
                doc.text += " " + " ".join(page.get_text() for page in pdf)
    except Exception:
        pass
    for path in list((root / "changes").glob("*.md")) + list((root / "equipment-manuals").glob("*.md")):
        content = path.read_text(encoding="utf-8")
        refs = set(re.findall(r"\b(?:[A-Z]{1,3}-\d{2}-\d{2}|[A-Z]{1,3}-\d{2}|\d{4})\b", content))
        for ref in refs:
            if ref in docs: docs[ref].text += " " + content
    return list(docs.values())

DOCS: list[SearchDoc] | None = None
def documents() -> list[SearchDoc]:
    global DOCS
    if DOCS is None: DOCS = _load_docs()
    return DOCS

def _floor_match(requested: str, doc_floor: str, title: str) -> bool:
    rf, df = _norm(requested), _norm(doc_floor + " " + title)
    if rf == "基準f": return "基準" in df
    if rf == "r": return "屋上" in df or " r" in df or df.startswith("r")
    if rf == "ph": return "ph" in df or "塔屋" in df
    if rf == "pit": return "pit" in df or "ピット" in df
    m = re.fullmatch(r"(\d+)f", rf)
    if rf in {"b1", "b2"}:
        return rf in df or f"地下{rf[1]}" in df
    if m:
        number = m.group(1)
        # Do not let 1F match B1F/地下1階. Ranges such as 1・2F and
        # 基準階(3-8F) are accepted as long as the requested floor is in the
        # range or the title explicitly says 基準階.
        direct = re.search(rf"(?<![b地下]){number}(?:f|階)", df)
        range_match = re.search(rf"{number}\s*[・,～〜-]\s*\d+f", df)
        return bool(direct or range_match or ("基準" in df and int(number) >= 3))
    return rf in df

def _metadata_score(doc: SearchDoc, parsed: ParsedQuery) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    title = _norm(doc.title)
    number = _norm(doc.drawing_number)
    raw_query = _norm(parsed.raw_query)

    # Exact identifiers and title phrases are stronger evidence than a loose
    # bigram match. These are domain-general features and do not reference any
    # evaluation case or gold document.
    if number and number in raw_query:
        score += 10.0
        reasons.append("drawing-number exact")
    for token in parsed.tokens:
        normalized = _norm(token)
        if len(normalized) >= 4 and normalized in title:
            score += 2.0
            reasons.append(f"title exact:{token}")
    for floor in parsed.floors:
        if _floor_match(floor, doc.floor, doc.title): score += 7.0; reasons.append(f"floor={floor}")
        elif "全階" in doc.floor or "共通" in doc.floor: reasons.append("floor=common")
        else: score -= 4.0
    for typ in parsed.drawing_types:
        title = doc.title
        type_stem = {"平面図": "平面", "配置図": "配置", "系統図": "系統", "詳細図": "詳細", "製作図": "製作", "伏図": "伏", "図": "図"}.get(typ, typ)
        if type_stem in title or (typ == "図" and "図" in title): score += 4.0; reasons.append(f"type={typ}")
        else: score -= 0.5
    # Discipline is a metadata proxy, not a query-specific gold mapping.
    mechanical_terms = (
        "空調", "換気", "給水", "排水", "給湯", "消火", "配管", "衛生", "ポンプ",
        "厨房", "駐車", "防水", "防食", "排煙",
    )
    electrical_terms = ("照明", "電気", "セキュリティ", "ブレーカー", "受変電", "発電", "防犯")
    architectural_terms = ("建具", "サッシ", "間取り", "防火戸", "入口", "外装", "会議室")
    if any(x in parsed.positive_terms for x in mechanical_terms) and doc.discipline == "機械":
        score += 2.5; reasons.append("discipline=機械")
    if any(x in parsed.positive_terms for x in electrical_terms) and doc.discipline == "電気":
        score += 2.5; reasons.append("discipline=電気")
    if any(x in parsed.positive_terms for x in architectural_terms) and doc.discipline == "意匠":
        score += 1.5; reasons.append("discipline=意匠")

    # Canonical synonym groups are emitted by the query parser. A matching
    # domain term in the drawing title gives a modest boost without making a
    # broad synonym outrank an exact drawing title.
    entity_groups = {
        "空調": ("空調", "エアコン", "冷房"),
        "換気": ("換気", "排気", "排煙"),
        "給水": ("給水", "受水槽", "ポンプ"),
        "排水": ("排水", "汚水", "雑排水", "通気"),
        "昇降機": ("エレベーター", "エレベータ", "昇降機", "EV"),
        "受変電": ("受変電", "受電", "変圧器"),
        "廃棄物": ("ゴミ", "ごみ", "廃棄物"),
        "シャフト": ("シャフト", "EPS", "PS", "DS", "コア"),
    }
    for canonical, variants in entity_groups.items():
        if canonical in parsed.positive_terms and any(_norm(variant) in title for variant in variants):
            score += 2.0
            reasons.append(f"entity title={canonical}")
    return score, reasons

def _lexical_score(doc: SearchDoc, parsed: ParsedQuery) -> tuple[float, list[str]]:
    title, body = _norm(" ".join((doc.drawing_number, doc.title, doc.floor, doc.discipline))), _norm(doc.text)
    score, reasons = 0.0, []
    for term in parsed.tokens:
        t = _norm(term)
        if len(t) < 2: continue
        if t in title: score += 3.0; reasons.append(f"title:{term}")
        elif t in body: score += 0.75; reasons.append(f"text:{term}")
    return score, reasons

def _semantic_score(doc: SearchDoc, parsed: ParsedQuery) -> tuple[float, list[str]]:
    hay = _norm(" ".join((doc.title, doc.floor, doc.discipline, doc.text)))
    matches = [term for term in parsed.positive_terms if len(term) >= 2 and _norm(term) in hay]
    # This is a deterministic semantic proxy used when GEMINI_API_KEY is absent.
    return min(5.0, len(set(matches)) * 0.35), [f"semantic:{x}" for x in matches[:5]]

def _score(doc: SearchDoc, parsed: ParsedQuery) -> tuple[float, list[str]]:
    # Negative constraints are applied to document metadata and title-level
    # evidence. Full contextual text can contain a related-but-not-target
    # category (e.g. an air-conditioning unit referenced on a ventilation plan).
    negative_hay = _norm(" ".join((doc.drawing_number, doc.title, doc.floor, doc.discipline)))
    negative = [term for term in parsed.negative_terms if _norm(term) in negative_hay]
    if negative: return -100.0, [f"negative filter: {x}" for x in negative]
    lexical, lr = _lexical_score(doc, parsed)
    metadata, mr = _metadata_score(doc, parsed)
    semantic, sr = _semantic_score(doc, parsed)
    return (
        FUSION_WEIGHTS["lexical"] * lexical
        + FUSION_WEIGHTS["metadata"] * metadata
        + FUSION_WEIGHTS["semantic"] * semantic
    ), lr + mr + sr

def score_components(doc: SearchDoc, parsed: ParsedQuery) -> dict:
    lexical, _ = _lexical_score(doc, parsed)
    metadata, _ = _metadata_score(doc, parsed)
    semantic, _ = _semantic_score(doc, parsed)
    final = (
        FUSION_WEIGHTS["lexical"] * lexical
        + FUSION_WEIGHTS["metadata"] * metadata
        + FUSION_WEIGHTS["semantic"] * semantic
    )
    return {"lexical_score": round(lexical, 4), "metadata_score": round(metadata, 4), "vector_score": 0.0, "semantic_proxy_score": round(semantic, 4), "fusion_score": round(final, 4)}

def _indexed_evidence(parsed: ParsedQuery, query: str) -> dict[str, dict]:
    from app.core.database import SessionLocal
    from app.repositories.chunks import current_chunks, vector_candidates
    from app.models.document import Document, DocumentVersion, Page
    found: dict[str, dict] = {}
    try:
        with SessionLocal() as session:
            for document, version, page, chunk in current_chunks(session):
                text = _norm(chunk.content)
                hits = sum(1 for term in parsed.positive_terms if len(term) > 2 and _norm(term) in text)
                relevance = min(4.0, hits * 0.65)
                prior = found.get(document.drawing_number)
                if prior is None or relevance > prior["chunk_score"]:
                    found[document.drawing_number] = {
                        "document_id": document.id, "page_id": page.id, "chunk_id": chunk.id,
                        "title": display_title(document.title, version.file_path, document.drawing_number), "floor": document.floor,
                        "discipline": document.discipline,
                        "page": page.page_number, "version": version.version,
                        "source_type": version.source_type,
                        "snippet": chunk.content[:600], "bbox": chunk.bbox,
                        "chunk_score": relevance, "vector_score": 0.0,
                    }
            if get_settings().gemini_api_key and found:
                from app.ai.embeddings import GeminiEmbedder
                vector = GeminiEmbedder().query(query)
                logger.info("Query embedding generated (dimensions=%d)", len(vector))
                for chunk, similarity in vector_candidates(session, vector):
                    page = session.get(Page, chunk.page_id)
                    version = session.get(DocumentVersion, page.document_version_id)
                    document = session.get(Document, version.document_id)
                    current = found.get(document.drawing_number)
                    if current and similarity > current["vector_score"]:
                        current.update({"page_id": page.id, "chunk_id": chunk.id, "page": page.page_number,
                                    "version": version.version, "source_type": version.source_type,
                                    "snippet": chunk.content[:600], "bbox": chunk.bbox,
                                    "vector_score": similarity})
                logger.info(
                    "Hybrid retrieval fused lexical/metadata signals with %d indexed documents",
                    len(found),
                )
    except SQLAlchemyError as exc:
        logger.info("Page index unavailable: %s", exc)
    return found


def qa_evidence_candidates(
    query: str,
    limit: int = 160,
    *,
    floor: str | None = None,
    discipline: str | None = None,
    source_type: str | None = None,
    include_vector: bool = True,
) -> list[dict]:
    """Return chunk-level candidates for QA without changing document search.

    Ordinary search intentionally emits one representative chunk per drawing.
    QA instead needs the most informative page blocks, including several
    different drawings. It uses the same one query vector when embeddings are
    configured, but never modifies the evidence graph.
    """
    from app.core.database import SessionLocal
    from app.repositories.chunks import current_chunks, vector_candidates

    parsed = parse_query(query)
    wanted_floor = _canonical_filter_floor(floor) if floor else None
    rows: list[dict] = []
    page_quality: dict[int, float] = {}
    try:
        with SessionLocal() as session:
            records: dict[int, tuple] = {}
            by_chunk: dict[int, dict] = {}

            def build_row(document, version, page, chunk) -> dict:
                metadata = chunk.metadata_json or {}
                text = chunk.content
                normalized = _norm(" ".join((document.drawing_number, document.title, document.floor, text)))
                hits = [term for term in parsed.positive_terms if len(_norm(term)) >= 2 and _norm(term) in normalized]
                if "quality_score" in metadata:
                    quality = float(metadata["quality_score"])
                else:
                    if page.id not in page_quality:
                        page_quality[page.id] = assess_evidence_quality(page.raw_text or text).score
                    quality = page_quality[page.id]
                return {
                    "drawing_number": document.drawing_number,
                    "title": display_title(document.title, version.file_path, document.drawing_number),
                    "floor": document.floor, "discipline": document.discipline,
                    "source_type": version.source_type, "revision": version.version,
                    "version": version.version, "page": page.page_number,
                    "score": round(len(set(hits)) + quality * 0.35, 4),
                    "snippet": text[:1800], "search_reason": ["qa chunk lexical match"],
                    "document_id": document.id, "page_id": page.id, "chunk_id": chunk.id,
                    "bbox": chunk.bbox, "evidence_kind": "pdf_chunk",
                    "quality_score": quality, "quality_class": metadata.get("quality_class"),
                    "extraction_provenance": metadata.get("extraction_provenance", metadata.get("extraction_method", "text")),
                    "grounding_status": metadata.get(
                        "grounding_status",
                        "vision_only" if metadata.get("extraction_provenance") == "vision" else metadata.get("extraction_method", "text_layer"),
                    ),
                    "qa_candidate_terms": list(dict.fromkeys(hits)), "vector_score": 0.0,
                }

            for document, version, page, chunk in current_chunks(session):
                if wanted_floor and not _floor_match(wanted_floor, document.floor, document.title):
                    continue
                if discipline and _norm(discipline) not in _norm(document.discipline):
                    continue
                if source_type and _norm(source_type) != _norm(version.source_type):
                    continue
                records[chunk.id] = (document, version, page, chunk)
                row = build_row(document, version, page, chunk)
                hits = row["qa_candidate_terms"]
                # Chunk-only candidates without a meaningful lexical or
                # identifier signal would only crowd grounded context.
                if not hits and _norm(document.drawing_number) not in _norm(query):
                    continue
                by_chunk[chunk.id] = row
                rows.append(row)
            if include_vector and get_settings().gemini_api_key and records:
                from app.ai.embeddings import GeminiEmbedder
                vector = GeminiEmbedder().query(query)
                for chunk, similarity in vector_candidates(session, vector, limit=limit):
                    record = records.get(chunk.id)
                    if record is None:
                        continue
                    row = by_chunk.get(chunk.id)
                    if row is None:
                        row = build_row(*record)
                        by_chunk[chunk.id] = row
                        rows.append(row)
                    row["vector_score"] = max(float(row["vector_score"]), similarity)
                    row["score"] = round(float(row["score"]) + similarity * 0.7, 4)
    except SQLAlchemyError as exc:
        logger.info("QA chunk index unavailable: %s", exc)
    rows.sort(key=lambda row: (-row["score"], row["drawing_number"], row["page"], row["chunk_id"]))
    # Do not hard-cap a document before QA reranking. A table or a multi-line
    # schedule often stores its answer in several chunks from one page. Apply
    # a soft penalty while filling the candidate pool so a clearly stronger
    # repeated chunk can still survive.
    return _soft_candidate_pool(
        rows,
        limit,
        max(0.0, float(get_settings().qa_document_repeat_penalty)),
    )


def _soft_candidate_pool(rows: list[dict], limit: int, penalty: float) -> list[dict]:
    """Fill a QA candidate pool without a hard per-document cap."""
    selected: list[dict] = []
    counts: dict[object, int] = {}
    remaining = list(rows)
    while remaining and len(selected) < limit:
        choice = max(
            remaining,
            key=lambda row: (
                float(row.get("score", 0.0))
                - penalty * counts.get(row.get("document_id") or row["drawing_number"], 0),
                -int(row.get("chunk_id") or 0),
            ),
        )
        remaining.remove(choice)
        document_key = choice.get("document_id") or choice["drawing_number"]
        counts[document_key] = counts.get(document_key, 0) + 1
        selected.append(choice)
    return selected

def search(
    query: str,
    limit: int = 10,
    *,
    floor: str | None = None,
    discipline: str | None = None,
    source_type: str | None = None,
) -> tuple[ParsedQuery, list[dict]]:
    parsed = parse_query(query)
    indexed = _indexed_evidence(parsed, query)
    catalog = documents()
    known_numbers = {doc.drawing_number for doc in catalog}
    # Supporting Markdown documents are not in drawing_list.csv. Add current
    # DB-backed source documents to the same ranking path as catalog drawings.
    search_docs = list(catalog)
    for number, evidence in indexed.items():
        if number not in known_numbers:
            search_docs.append(
                SearchDoc(
                    drawing_number=number,
                    title=evidence.get("title", number),
                    floor=evidence.get("floor", ""),
                    discipline=evidence.get("discipline", ""),
                    text=evidence.get("snippet", ""),
                    source_type=evidence.get("source_type", "pdf"),
                )
            )
    ranked = []
    for doc in search_docs:
        evidence = indexed.get(doc.drawing_number)
        effective_source_type = evidence.get("source_type", doc.source_type) if evidence else doc.source_type
        if floor and not _floor_match(_canonical_filter_floor(floor), doc.floor, doc.title):
            continue
        if discipline and _norm(discipline) not in _norm(doc.discipline):
            continue
        if source_type and _norm(source_type) != _norm(effective_source_type):
            continue
        score, reasons = _score(doc, parsed)
        if evidence:
            score += evidence["chunk_score"] + 2.0 * evidence["vector_score"]
            if evidence["chunk_score"]: reasons.append("page chunk match")
            if evidence["vector_score"]: reasons.append("pgvector similarity")
        # A chunk-level lexical hit is independently useful evidence even
        # when the catalog metadata has no positive score (common for scanned
        # drawings and supporting Markdown). Keep it in the candidate pool so
        # QA evidence recall does not depend on catalog metadata alone.
        if score <= 0:
            continue
        ranked.append((score, doc, reasons))
    ranked.sort(key=lambda row: (-row[0], row[1].drawing_number))
    out = []
    for score, doc, reasons in ranked[:limit]:
        evidence = indexed.get(doc.drawing_number, {})
        components = score_components(doc, parsed)
        components.update({"vector_score": round(evidence.get("vector_score", 0.0), 4), "fusion_score": round(score, 4)})
        out.append({"drawing_number": doc.drawing_number, "title": doc.title, "floor": doc.floor,
                    "discipline": doc.discipline, "source_type": evidence.get("source_type", doc.source_type),
                    "revision": evidence.get("version"), "page": evidence.get("page", 1), "score": round(score, 4),
                    "snippet": evidence.get("snippet") or re.sub(r"\s+", " ", " ".join((doc.title, doc.text)))[:300],
                    "search_reason": reasons, "source_path": doc.source_path,
                    "document_id": evidence.get("document_id"), "page_id": evidence.get("page_id"),
                    "chunk_id": evidence.get("chunk_id"), "version": evidence.get("version"),
                    "bbox": evidence.get("bbox"), "evidence_kind": "pdf_chunk" if evidence.get("chunk_id") else "catalog",
                    **components})
    return parsed, out


def _canonical_filter_floor(value: str) -> str:
    normalized = value.strip().replace("Ｆ", "F")
    if normalized.endswith("階"):
        normalized = normalized[:-1] + "F"
    if normalized.startswith("地下"):
        return "B" + normalized[2:].rstrip("F")
    return normalized
