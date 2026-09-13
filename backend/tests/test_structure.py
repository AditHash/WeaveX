"""Phase 8 tests — document structure detection.

Headings test uses the exact worked example from PLAN.md #20 (real
project documentation, not a fabricated sample). The paragraph-only test
uses the real FastAPICorpus text, cross-checked against how the same
text is chunked into paragraphs in fixtures.py. List/quote/code/ATX cases
use small targeted literals, since the real corpus and docs don't happen
to contain those block types — same narrow-case allowance used in
Phase 7's normalizer tests.
"""

from app.chunking import BlockType, detect_structure, normalize_text

from tests.fixtures import FASTAPI_CORPUS_TEXT


# ------------------------------------------------------- setext headings


def test_detects_setext_headings_from_plan_md_example():
    # Exact structure from PLAN.md #20's worked example.
    text = (
        "FASTAPI ARCHITECTURE\n"
        "====================\n"
        "\n"
        "FastAPI is a modern Python web framework.\n"
        "\n"
        "Components\n"
        "----------\n"
        "\n"
        "FastAPI uses Pydantic and Starlette."
    )
    blocks = detect_structure(normalize_text(text))

    assert [(b.block_type, b.level) for b in blocks] == [
        (BlockType.HEADING, 1),
        (BlockType.PARAGRAPH, None),
        (BlockType.HEADING, 2),
        (BlockType.PARAGRAPH, None),
    ]
    assert blocks[0].text == "FASTAPI ARCHITECTURE"
    assert blocks[2].text == "Components"
    assert blocks[3].text == "FastAPI uses Pydantic and Starlette."


def test_atx_heading_levels():
    text = "# Title\n\n## Subtitle\n\n###### Deep heading\n\nBody text."
    blocks = detect_structure(normalize_text(text))
    headings = [b for b in blocks if b.block_type == BlockType.HEADING]
    assert [(h.text, h.level) for h in headings] == [
        ("Title", 1),
        ("Subtitle", 2),
        ("Deep heading", 6),
    ]


# ------------------------------------------------------ real corpus text


def test_real_corpus_text_is_four_plain_paragraphs():
    # No headings/lists in the actual FastAPI intro corpus — every block
    # should classify as a plain paragraph, one per blank-line-separated
    # paragraph (matching fixtures.py's own paragraph split).
    blocks = detect_structure(normalize_text(FASTAPI_CORPUS_TEXT))

    assert len(blocks) == 4
    assert all(b.block_type == BlockType.PARAGRAPH for b in blocks)
    assert blocks[0].text == "FastAPI is a modern Python web framework."
    assert blocks[2].text == "Sebastián Ramírez created FastAPI."


# ------------------------------------------------------------------ list


def test_detects_list_block():
    text = "Intro paragraph.\n\n- Entities\n- Relationships\n- Evidence"
    blocks = detect_structure(normalize_text(text))
    assert blocks[0].block_type == BlockType.PARAGRAPH
    assert blocks[1].block_type == BlockType.LIST
    assert "Entities" in blocks[1].text


def test_numbered_list_detected():
    text = "1. First step\n2. Second step\n3. Third step"
    blocks = detect_structure(normalize_text(text))
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.LIST


# ----------------------------------------------------------------- quote


def test_detects_quote_block():
    text = "Paragraph before.\n\n> A quoted line.\n> Another quoted line.\n\nAfter."
    blocks = detect_structure(normalize_text(text))
    assert [b.block_type for b in blocks] == [
        BlockType.PARAGRAPH, BlockType.QUOTE, BlockType.PARAGRAPH,
    ]


# ------------------------------------------------------------ code block


def test_code_block_preserves_internal_blank_lines():
    # The whole reason code fences are extracted before blank-line
    # splitting: a blank line INSIDE the fence must not become a block
    # boundary.
    text = (
        "Before code.\n\n"
        "```python\n"
        "def foo():\n"
        "\n"
        "    return 1\n"
        "```\n\n"
        "After code."
    )
    blocks = detect_structure(normalize_text(text))
    assert [b.block_type for b in blocks] == [
        BlockType.PARAGRAPH, BlockType.CODE_BLOCK, BlockType.PARAGRAPH,
    ]
    assert "def foo():" in blocks[1].text
    assert "return 1" in blocks[1].text


def test_code_block_content_is_not_classified_as_something_else():
    # A code block full of "- " style lines must not be misread as a LIST
    # just because it happens to look like one syntactically.
    text = "```text\n- not\n- a\n- real\n- list\n```"
    blocks = detect_structure(normalize_text(text))
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.CODE_BLOCK
