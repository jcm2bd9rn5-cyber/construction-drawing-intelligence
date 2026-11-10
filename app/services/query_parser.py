import re
from dataclasses import asdict, dataclass

STOPWORDS = {"の", "は", "が", "を", "に", "で", "へ", "と", "や", "について", "関する", "確認したい", "教えて", "ください", "どこ", "場所", "位置", "図面", "関係する", "使える"}
TYPE_TERMS = ("平面図", "系統図", "詳細図", "配置図", "立面図", "断面図", "伏図", "天井伏図", "製作図", "リスト", "計画図", "範囲図", "仕上表")

@dataclass(frozen=True)
class ParsedQuery:
    positive_terms: list[str]
    negative_terms: list[str]
    floors: list[str]
    drawing_types: list[str]
    tokens: list[str]
    raw_query: str = ""

def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for part in re.split(r"[のはとをがにでへや、。？?\s]+", text):
        if not part or part in STOPWORDS:
            continue
        if len(part) >= 2:
            terms.append(part)
        terms.extend(re.findall(r"地下[12]|[0-9]+階|基準階|屋上|塔屋|ピット", part))
        for type_term in TYPE_TERMS:
            if type_term in part:
                terms.append(type_term)
                if type_term.endswith("図"): terms.append(type_term[:-1])
        if len(part) >= 3 and all("\u3000" <= c <= "\u9fff" for c in part):
            terms.extend(part[i:i+2] for i in range(len(part) - 1))
    return list(dict.fromkeys(terms))

def _canonical_floor(raw: str) -> str:
    s = raw.replace("Ｆ", "F").replace("階", "F").replace(" ", "")
    if "地下" in s:
        found = re.search(r"[12]", s)
        return "B" + (found.group(0) if found else "")
    if "基準" in s: return "基準階"
    if "屋上" in s: return "R"
    if "塔屋" in s: return "PH"
    if "ピット" in s: return "PIT"
    return s

def parse_query(query: str) -> ParsedQuery:
    text = query.strip()
    negative: list[str] = []
    positive = text
    for marker in ("ではなく", "でなく", "以外", "除く", "除外", "なし"):
        if marker not in text: continue
        left, right = text.split(marker, 1)
        candidates = _terms(left)
        if candidates: negative.append(candidates[-1])
        positive = right if right else ""
    floors = [_canonical_floor(x) for x in re.findall(r"地下\s*[12]\s*(?:階|F|Ｆ)?|[0-9]+\s*(?:階|F|Ｆ)|基準階|屋上|塔屋|ピット", text)]
    floors = list(dict.fromkeys(floors))
    drawing_types = [x for x in TYPE_TERMS if x in text]
    if "平面" in text and "平面図" not in drawing_types: drawing_types.append("平面図")
    if "配置" in text and "配置図" not in drawing_types: drawing_types.append("配置図")
    tokens = _terms(positive)
    positive_terms = list(tokens)
    synonym_groups = {
        "空調": ["空調", "エアコン", "冷房", "室外機", "空調配管"],
        "換気": ["換気", "排気", "排煙", "全熱交換"],
        "給水": ["給水", "受水槽", "給水ポンプ"],
        "排水": ["排水", "汚水", "雑排水", "通気"],
        "建具": ["建具", "ドア", "サッシ", "ガラス", "AW", "SD", "SSD"],
        "照明": ["照明", "灯", "器具", "コーブ"],
        "防災": ["防災", "防火", "消火", "非常照明", "避難"],
        "機械式駐車場": ["機械式駐車場", "駐車", "パレット"],
        "配管": ["配管", "パイプ", "ルート"],
        "ブレーカー": ["ブレーカー", "分電盤", "動力盤", "盤"],
        "昇降機": ["エレベーター", "エレベータ", "昇降機", "EV"],
        "防犯": ["セキュリティ", "防犯", "入退室", "オートロック"],
        "防水": ["雨漏り", "防水", "防食", "笠木", "パラペット"],
        "廃棄物": ["ゴミ", "ごみ", "廃棄物", "ゴミ置場"],
        "会議室": ["会議室", "会議", "事務室"],
        "シャフト": ["シャフト", "EPS", "PS", "DS", "コア"],
        "入口": ["入口", "エントランス", "自動ドア", "ガラススクリーン"],
        # Expand to related concepts only. Concrete values are answers, not
        # query terms: injecting them would make retrieval dataset-specific.
        "受変電": ["受変電", "受電", "変圧器", "高圧", "単線結線", "配電"],
        "間取り": ["間取り", "住戸タイプ", "3LDK", "2LDK", "1LDK"],
        "防火区画": ["防火区画", "面積区画", "竪穴区画", "異種用途"],
        "外装仕上": ["外装", "仕上材", "ECP", "押出成形セメント", "タイル"],
        "発電負荷": ["発電機", "非常用発電", "300kVA", "消防", "保安"],
        "昇降機用途": ["エレベーター", "エレベータ", "昇降機", "EV", "乗用", "非常用", "住戸"],
        "ガス給湯": ["ガスメーター", "メーター", "給湯器", "給湯", "ガス"],
        "消火種類": ["消火", "屋内消火栓", "スプリンクラー", "SP", "連結送水管"],
        "自火報": ["自動火災報知", "受信機", "防災センター"],
        "搬入": ["搬入", "荷捌", "店舗", "経路"],
        "駐輪": ["駐輪場", "駐輪", "収容台数", "台"],
    }
    for variants in synonym_groups.values():
        if any(v in positive for v in variants): positive_terms.extend(variants)
    return ParsedQuery(
        list(dict.fromkeys(positive_terms)),
        negative,
        floors,
        drawing_types,
        tokens,
        raw_query=text,
    )


def decompose_query(query: str) -> list[str]:
    """Return conservative sub-queries for conjunctive fact questions.

    This is intentionally a small deterministic rule set. It preserves the
    surrounding subject (``杭の工法と支持層`` -> ``杭の工法`` and
    ``杭の支持層``) and leaves questions without an unambiguous conjunction
    unchanged. The function is used only to widen QA evidence retrieval; the
    ordinary document ranking path still receives the original query.
    """
    text = query.strip()
    for marker in ("および", "ならびに", "及び"):
        if marker in text:
            left, right = text.split(marker, 1)
            return _join_conjuncts(left, right)
    # A Japanese ``と`` is useful for lists, but avoid splitting common
    # suffixes and return the original query when either side is too short.
    match = re.search(r"(.{2,})と(.{2,})", text)
    if match:
        return _join_conjuncts(match.group(1), match.group(2))
    return [text]


def _join_conjuncts(left: str, right: str) -> list[str]:
    left = left.strip(" 、。？?\t")
    right = right.strip(" 、。？?\t")
    if len(left) < 2 or len(right) < 2:
        return [left + right]
    # The right clause usually carries the question ending. Use the noun
    # context before the last ``の`` as a subject for the right-hand fact.
    context = left.rsplit("の", 1)[0] + "の" if "の" in left else ""
    right_query = context + right
    return list(dict.fromkeys([left, right_query]))

def parsed_dict(parsed: ParsedQuery) -> dict:
    return asdict(parsed)
