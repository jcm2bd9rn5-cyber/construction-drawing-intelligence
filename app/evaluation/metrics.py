import math

def recall_at_k(relevant: dict[str, int], predicted: list[str], k: int = 5) -> float:
    gold = {x for x, grade in relevant.items() if grade > 0}
    return len(gold & set(predicted[:k])) / len(gold) if gold else 0.0

def mrr(relevant: dict[str, int], predicted: list[str]) -> float:
    for i, item in enumerate(predicted, 1):
        if relevant.get(item, 0) > 0: return 1 / i
    return 0.0

def ndcg_at_k(relevant: dict[str, int], predicted: list[str], k: int = 10) -> float:
    dcg = sum(relevant.get(item, 0) / math.log2(i + 2) for i, item in enumerate(predicted[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(value / math.log2(i + 2) for i, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def citation_scores(gold: list[str], predicted: list[str]) -> dict[str, float]:
    expected, actual = set(gold), set(predicted)
    matched = len(expected & actual)
    precision = matched / len(actual) if actual else 0.0
    recall = matched / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"CitationPrecision": precision, "CitationRecall": recall, "CitationF1": f1}

def must_contain_accuracy(groups: list[list[str]], answer: str) -> float:
    return float(all(any(term in answer for term in group) for group in groups))
