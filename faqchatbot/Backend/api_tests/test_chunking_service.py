from app.services.chunking_service import _split_into_segments


def test_split_into_segments_preserves_headings_and_paragraphs():
    text = "# Overview\nThis is the intro.\n\n## Details\nMore detail here.\n\n### Example\nUse this example."

    segments = _split_into_segments(text)

    assert segments[0].startswith("# Overview")
    assert segments[1].startswith("## Details")
    assert segments[2].startswith("### Example")
