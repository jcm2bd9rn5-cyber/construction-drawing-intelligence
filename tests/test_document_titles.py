from app.services.document_titles import display_title


def test_markdown_heading_is_used_only_as_display_title(tmp_path):
    source = tmp_path / "manual.md"
    source.write_text("# 非常用ディーゼル発電機 300kVA 保守マニュアル\n\n本文", encoding="utf-8")
    assert display_title("", source, "man-gen") == "非常用ディーゼル発電機 300kVA 保守マニュアル"
    assert display_title("明示タイトル", source, "man-gen") == "明示タイトル"
    assert display_title("", None, "man-gen") == "man-gen"
