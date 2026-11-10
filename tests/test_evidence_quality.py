from app.services.evidence_quality import (
    QUALITY_GOOD,
    QUALITY_POOR,
    QUALITY_WEAK,
    assess_evidence_quality,
)
from app.services.ingestion import TextChunk, needs_vision


def test_quality_score_separates_readable_notes_from_ocr_garbage():
    good = assess_evidence_quality(
        "1階機械室 給水ポンプ P-1 は 18m3/h、揚程 35m。点検時はバルブを閉止する。" * 4
    )
    weak = assess_evidence_quality("1階 機械室\n給水ポンプ\n点検対象")
    poor = assess_evidence_quality("@@@@ #### �\x00 �\x00")

    assert good.quality_class == QUALITY_GOOD
    assert weak.quality_class == QUALITY_WEAK
    assert poor.quality_class == QUALITY_POOR
    assert good.score > weak.score > poor.score


def test_low_quality_detection_uses_quality_not_only_character_count():
    assert needs_vision([TextChunk("あ" * 120, None)])
    assert not needs_vision([TextChunk("給水ポンプP-1は機械室に設置。流量18m3/h。" * 6, None)])
