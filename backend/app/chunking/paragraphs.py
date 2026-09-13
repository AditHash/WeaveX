"""Paragraph splitting with offset tracking.

Shared by both chunking versions (V1's size-based packing and V2's
similarity-based packing) — pulled out here once a second consumer
needed the exact same offset-tracking split, instead of duplicating it.
"""


def split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """Return (start_offset, end_offset, text) per blank-line-separated
    paragraph, offsets relative to the original `text`.

    Assumes text is already normalized (Phase 7) — paragraphs are
    separated by exactly one blank line, which normalize_text guarantees.
    """
    raw_paragraphs = [p for p in text.split("\n\n") if p.strip()]
    results: list[tuple[int, int, str]] = []
    search_from = 0
    for para in raw_paragraphs:
        stripped = para.strip("\n")
        start = text.index(stripped, search_from)
        end = start + len(stripped)
        results.append((start, end, stripped))
        search_from = end
    return results
