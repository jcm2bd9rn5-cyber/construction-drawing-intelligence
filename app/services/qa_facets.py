"""Conservative, deterministic question requirements and typed evidence checks.

Numbered equipment labels are not an exhaustive inventory. In particular a
range in a drawing title never establishes the building's total equipment count.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.services.query_parser import decompose_query


@dataclass(frozen=True)
class Facet:
    name: str
    query: str
    subject: str = ""
    source_id: str | None = None


PATTERNS = {
    "用途": r"用途\s*[:：][^。\n]+|[一-龥ァ-ヶ]+用(?![一-龥])",
    "台数": r"\d+\s*(?:台|基)(?!盤)",
    "数量": r"\d+\s*(?:台|基|個|本|枚|箇所|か所|戸)",
    "人数": r"\d+\s*人",
    "定員": r"\d+\s*人",
    "積載量": r"(?:積載|荷重)[^。\n]{0,30}\d+\s*(?:kg|t|トン)",
    "重量": r"(?:重量|質量|自重)[^。\n]{0,30}\d+\s*(?:kg|t|トン)",
    "寸法": r"\d+(?:\.\d+)?\s*(?:mm|cm|m)(?![a-zA-Z0-9²³])|(?:幅|高さ|奥行|寸法)\s*[:：=]?\s*\d+",
    "階数": r"(?:地上|地下|階数)[^。\n]{0,15}\d+\s*階",
    "部屋数": r"\d+\s*室",
    "戸数": r"\d+\s*戸",
    "場所": r"(?:地下)?\d+階|[一-龥ァ-ヶ]+(?:室|階|内|置場|センター)|設置場所|所在地",
    "設置場所": r"(?:地下)?\d+階|[一-龥ァ-ヶ]+(?:室|階|内|置場|センター)|設置場所",
    "型式": r"型式|型番|品番|[A-Z]{2,}[-][A-Z0-9-]+",
    "型番": r"型式|型番|品番|[A-Z]{2,}[-][A-Z0-9-]+",
    "変更内容": r"変更前|変更後|変更箇所|変更概要",
    "影響範囲": r"影響|対象|範囲",
    "影響": r"影響|対象|範囲",
    "理由": r"理由|起因|ため",
    "対応": r"対応|対策|措置",
    "設備名": r"設備名|機器名|名称",
    "電圧": r"\d+(?:[.,]\d+)*\s*(?:kV|V)(?!A)",
    "容量": r"\d+(?:[.,]\d+)*\s*(?:kVA|kW|m3|m³|L|リットル)",
    "工法": r"工法|場所打ち|アースドリル|既製杭",
    "支持層": r"支持層|砂礫|GL\s*[-−]\s*\d+",
}
NUMERIC_NAMES = {"台数", "数量", "人数", "定員", "積載量", "重量", "寸法", "階数", "部屋数", "戸数", "電圧", "容量"}
CITATION_MARKER = re.compile(r"\[((?:E?\d+)(?:\s*[,、]\s*E?\d+)*)\]")

# Keep this deliberately stricter than the extraction patterns.  OCR text is
# still searchable evidence, but a malformed OCR fragment must never
# become a supported claim merely because its trailing ``00KVA`` looks valid.
_NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z0-9./])\d+(?:[.,]\d+)?\s*"
    r"(?:kVA|kV|kW|kg|mm|cm|m3|m³|m|t|V|W|台|基|人|階|室|戸)(?![A-Za-z0-9./])",
    re.I,
)

# A slash can express a legitimate transformer ratio (``6.6kV/210V``) or a
# multi-voltage supply (``100/200V``).  It is not safe, however, to split a
# visibly truncated OCR fragment such as ``220V/3V、`` into two asserted
# voltage facts.  Preserve the source verbatim and make only complete
# expressions available to typed-claim validation.
_VOLTAGE_EXPRESSION = re.compile(
    r"(?<![A-Za-z0-9.])(?:\d+(?:[.,]\d+)?\s*(?:kV|V)?\s*/\s*)?\d+(?:[.,]\d+)?\s*(?:kV|V)(?!A)",
    re.I,
)


def valid_voltage_expressions(text: str) -> list[str]:
    """Return complete, non-dangling voltage expressions for claim support."""
    normalized = unicodedata.normalize("NFKC", text)
    expressions = []
    for match in _VOLTAGE_EXPRESSION.finditer(normalized):
        expression = match.group().strip()
        # Do not accept a partial token from a longer slash expression.
        if (match.start() and normalized[match.start() - 1] == "/") or (
            match.end() < len(normalized) and normalized[match.end()] == "/"
        ):
            continue
        # A comma immediately after a slash expression is evidence that OCR
        # stopped mid-clause; it is not a complete pair of voltage facts.
        if "/" in expression and re.match(r"\s*[、,，]", normalized[match.end():]):
            continue
        expressions.append(expression)
    return list(dict.fromkeys(expressions))


def valid_numeric_tokens(text: str) -> list[str]:
    """Return only standalone, non-OCR-corrupted numeric claims.

    This is validation, not retrieval filtering: callers use it only when
    deciding whether a quote can support an answer.
    """
    return _NUMERIC_TOKEN.findall(unicodedata.normalize("NFKC", text))


def strip_citation_markers(text: str) -> str:
    return CITATION_MARKER.sub("", text)


def _clauses(text: str) -> list[str]:
    clauses, current, depth = [], [], 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        if character == "\n" or (character in "。、" and not depth):
            clauses.append("".join(current))
            current = []
        else:
            current.append(character)
    clauses.append("".join(current))
    return clauses


def extract_facets(question: str) -> list[Facet]:
    """Reuse QA decomposition; only recognize explicit requirement nouns.

    Longer names win (影響範囲 vs 影響). An arbitrary conjunction such as
    ガスメーターと給湯器 is not mistaken for two requested attributes.
    """
    facets = []
    for clause in decompose_query(question):
        matches = list(re.finditer("|".join(sorted(PATTERNS, key=len, reverse=True)), clause))
        for match in matches:
            name = match.group()
            if any(f.name == name for f in facets):
                continue
            prefix = clause[:match.start()].rstrip("の ")
            # A shared subject is explicit only before の. Avoid turning
            # modifiers such as コスト/対象/変圧器 into universal scope claims.
            subject = clause.rsplit("の", 1)[0] if "の" in clause[:match.start()] else ""
            if subject in {"この建物", "建物"} or "変更指示" in subject:
                subject = ""
            source = re.search(r"変更指示\s*(\d{4})", question)
            facets.append(Facet(name, prefix + " " + name, subject, source.group(1) if source else None))
    return facets


def evidence_body(text: str) -> str:
    if "抽出本文:" in text:
        text = text.split("抽出本文:", 1)[1]
    return re.sub(r"(?m)^\s*(?:図面名|図面番号|Revision|ページ|工事名称)\s*[:：].*$", "", text).strip()


def _subject_present(subject: str, text: str) -> bool:
    if not subject:
        return True
    aliases = {"EV": ("EV", "昇降機", "エレベーター", "エレベータ")}
    terms = aliases.get(subject, (subject,))
    for term in terms:
        if term == "EV":
            # EV充電 / EV需要 are cars, not elevator equipment labels.
            if re.search(r"\bEV(?:[-−－]\d+|\s|[（(:：はがのを計]|$)", text, re.I):
                return True
        elif term in text:
            return True
    return False


def supporting_spans(facet: Facet, quote: str) -> list[str]:
    text = unicodedata.normalize("NFKC", evidence_body(quote))
    scoped_subject = facet.subject if facet.name in {"用途", "台数", "数量", "人数", "定員"} else ""
    if not _subject_present(scoped_subject, text):
        return []
    # Short OCR line breaks frequently separate an equipment label and its
    # usage. Keep adjacent lines together, but never use title-only metadata.
    text = re.sub(r"([^。\n])\n(?=[(（])", r"\1", text)
    spans = []
    for line in _clauses(text):
        line = line.strip(" -*#|\t")
        if not line or len(line) > 600:
            continue
        # A section heading (コスト影響 / 支持層) names the requirement but
        # contains no value and must not validate a different document's fact.
        if re.fullmatch(r"[一-龥]+", line) and facet.name in line and len(line) <= len(facet.name) + 4:
            continue
        if not re.search(PATTERNS[facet.name], line, re.I):
            continue
        if facet.name in NUMERIC_NAMES and not valid_numeric_tokens(line):
            continue
        if re.search(r"不明|未確認|未定|確定でき|記載なし|根拠不足", line):
            continue
        if scoped_subject and not _subject_present(scoped_subject, line):
            continue
        spans.append(line)
    return list(dict.fromkeys(spans))[:2]


def evidence_spans(facet: Facet, item) -> list[str]:
    if facet.source_id and item.drawing_number != facet.source_id:
        return []
    return supporting_spans(facet, item.quote)


def numeric_facts(text: str) -> set[tuple[str, str, str]]:
    """Value + unit + semantic role, not bare substring matching of digits."""
    text = unicodedata.normalize("NFKC", text)
    facts = set()
    roles = {"台": "equipment_count", "基": "equipment_count", "人": "capacity",
             "kg": "mass", "t": "mass", "mm": "dimension", "cm": "dimension",
             "m": "dimension", "階": "floor", "室": "room_count", "戸": "dwelling_count"}
    for match in re.finditer(r"(?<![A-Za-z0-9./])(\d+(?:[.,]\d+)*)\s*(kVA|kV|kW|kg|mm|cm|m3|m³|m|t|V|W|台|基|人|階|室|戸)(?![A-Za-z0-9./])", text, re.I):
        value, unit = match.group(1).replace(",", ""), match.group(2).lower()
        role = roles.get(unit, unit)
        context = re.split(r"[。\n、]", text[:match.start()])[-1][-20:]
        if unit in {"kg", "t"}:
            role = "load" if re.search(r"積載|荷重", context) else "mass"
        facts.add((value, unit, role))
    return facts


def validate_coverage(facets: list[Facet], answer: str, evidence) -> list[dict]:
    coverage = []
    for facet in facets:
        answer_spans = supporting_spans(facet, answer)
        supported_ids = []
        for item in evidence:
            spans = evidence_spans(facet, item)
            for claim in answer_spans:
                facts = numeric_facts(claim)
                if facet.name in NUMERIC_NAMES:
                    supported = bool(facts) and any(facts <= numeric_facts(span) for span in spans)
                else:
                    # Require an actual value-bearing phrase in both, not
                    # merely the request label ("用途は不明" is not coverage).
                    values = [m.group() for span in spans for m in re.finditer(PATTERNS[facet.name], span, re.I)]
                    # Label overlap alone is not entailment. Numeric facts in
                    # a nonnumeric facet still require the matching values.
                    supported = any(value in claim and value != facet.name for value in values) or any(span in claim for span in spans)
                    if facts and not any(facts <= numeric_facts(span) for span in spans):
                        supported = False
                if supported:
                    supported_ids.append(item.evidence_id)
                    break
        coverage.append({"name": facet.name, "answered": bool(answer_spans),
                         "supported": bool(supported_ids), "evidence_ids": list(dict.fromkeys(supported_ids))})
    return coverage
