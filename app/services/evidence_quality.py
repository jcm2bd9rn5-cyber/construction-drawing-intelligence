"""Cheap, deterministic quality checks for extracted page evidence.

The score deliberately measures readability rather than topical relevance.  It
is calculated during ingestion and retained with the chunk metadata so a search
request never has to re-run the heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


QUALITY_GOOD = "GOOD"
QUALITY_WEAK = "WEAK"
QUALITY_POOR = "POOR"


@dataclass(frozen=True)
class EvidenceQuality:
    score: float
    quality_class: str
    text_length: int
    readable_ratio: float
    symbol_ratio: float
    garbage_ratio: float
    diversity: float
    repeated_ratio: float
    heading_only: bool
    numeric_only: bool


def _quality_class(score: float, *, good_threshold: float, vision_threshold: float) -> str:
    if score >= good_threshold:
        return QUALITY_GOOD
    if score >= vision_threshold:
        return QUALITY_WEAK
    return QUALITY_POOR


def assess_evidence_quality(
    text: str,
    *,
    good_threshold: float = 0.62,
    vision_threshold: float = 0.42,
) -> EvidenceQuality:
    """Score extractability without an NLP model or evaluation labels.

    Japanese glyphs, Latin/digit labels, and normal drawing punctuation are
    useful.  Control/replacement characters, long repeated runs, symbol soup,
    headings, and number-only labels lower confidence.  A concise but readable
    schedule is therefore WEAK rather than automatically POOR.
    """
    compact = re.sub(r"\s+", "", text or "")
    length = len(compact)
    if not compact:
        return EvidenceQuality(0.0, QUALITY_POOR, 0, 0.0, 0.0, 1.0, 0.0, 0.0, False, False)

    readable = sum(char.isalnum() or "\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in compact)
    symbols = sum(not (char.isalnum() or "\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff") for char in compact)
    garbage = sum(char == "�" or ord(char) < 32 for char in compact)
    repeated = sum(len(match.group(0)) for match in re.finditer(r"(.)\1{3,}", compact))
    meaningful_chars = [char for char in compact if char.isalnum() or "\u3040" <= char <= "\u9fff"]
    diversity = len(set(meaningful_chars)) / max(1, len(meaningful_chars))
    readable_ratio = readable / length
    symbol_ratio = symbols / length
    garbage_ratio = garbage / length
    repeated_ratio = repeated / length
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    numeric_only = bool(re.fullmatch(r"[0-9０-９.．,，:/：\-+㎡mMkKkVvAaWwHz％%]+", compact))
    heading_only = len(lines) <= 1 and length < 55 and not numeric_only

    length_component = min(1.0, length / 240)
    diversity_component = min(1.0, diversity * 3.0)
    score = (
        0.45 * length_component
        + 0.28 * readable_ratio
        + 0.17 * diversity_component
        + 0.10 * min(1.0, len(lines) / 3)
        - 0.35 * symbol_ratio
        - 0.60 * garbage_ratio
        - 0.45 * repeated_ratio
        - (0.16 if heading_only else 0.0)
        - (0.20 if numeric_only else 0.0)
    )
    score = max(0.0, min(1.0, score))
    return EvidenceQuality(
        round(score, 4),
        _quality_class(score, good_threshold=good_threshold, vision_threshold=vision_threshold),
        length,
        round(readable_ratio, 4),
        round(symbol_ratio, 4),
        round(garbage_ratio, 4),
        round(diversity, 4),
        round(repeated_ratio, 4),
        heading_only,
        numeric_only,
    )


def quality_metadata(quality: EvidenceQuality) -> dict[str, float | int | str | bool]:
    return {
        "quality_score": quality.score,
        "quality_class": quality.quality_class,
        "quality_text_length": quality.text_length,
        "quality_readable_ratio": quality.readable_ratio,
        "quality_symbol_ratio": quality.symbol_ratio,
        "quality_garbage_ratio": quality.garbage_ratio,
        "quality_diversity": quality.diversity,
        "quality_repeated_ratio": quality.repeated_ratio,
        "quality_heading_only": quality.heading_only,
        "quality_numeric_only": quality.numeric_only,
    }
