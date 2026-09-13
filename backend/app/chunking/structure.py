"""Document structure detection — heuristic, no LLM.

Classifies normalized text into a flat, ordered sequence of typed blocks
(heading/paragraph/list/code_block/quote). Semantic chunking (Phase 9+)
needs this to know what it's looking at: a heading is a boundary signal,
a list shouldn't be split mid-item, a code block's internal blank lines
aren't paragraph breaks.

KNOWN LIMITATION (flagged, not fixed here): Phase 7's normalize_text runs
before this, and collapses repeated internal whitespace — which would
also flatten meaningful leading indentation inside a code block, since
normalization has no idea a code block is coming. Not fixed now: doing
so means making the normalizer structure-aware (a bigger redesign than a
heuristic), and nothing in the current corpus has a code block to prove
it against yet.

KNOWN LIMITATION 2: list/quote detection requires every line in a
blank-line-isolated segment to be uniform (all list items / all quote
lines). An intro line immediately followed by a list with no blank line
between them (common in real markdown-ish text) falls through to
PARAGRAPH in this V1.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

_CODE_FENCE = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)
_SETEXT_H1 = re.compile(r"^=+$")
_SETEXT_H2 = re.compile(r"^-+$")
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_QUOTE_LINE = re.compile(r"^\s*>")


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"


@dataclass
class StructureBlock:
    block_type: BlockType
    text: str
    level: int | None = None  # heading level (1-6); None for non-headings


def _classify_segment(segment: str) -> StructureBlock:
    lines = segment.split("\n")

    # Setext heading: a line immediately followed by a line of repeated
    # "=" (level 1) or "-" (level 2) — the style used in CLAUDE.md/PLAN.md.
    if len(lines) == 2:
        if _SETEXT_H1.match(lines[1]) and len(lines[1]) >= 3:
            return StructureBlock(BlockType.HEADING, lines[0], level=1)
        if _SETEXT_H2.match(lines[1]) and len(lines[1]) >= 3:
            return StructureBlock(BlockType.HEADING, lines[0], level=2)

    # ATX heading: "# Title" .. "###### Title", one line only.
    if len(lines) == 1:
        atx = _ATX_HEADING.match(lines[0])
        if atx:
            return StructureBlock(
                BlockType.HEADING, atx.group(2), level=len(atx.group(1))
            )

    if all(_LIST_ITEM.match(line) for line in lines):
        return StructureBlock(BlockType.LIST, segment)

    if all(_QUOTE_LINE.match(line) for line in lines):
        return StructureBlock(BlockType.QUOTE, segment)

    return StructureBlock(BlockType.PARAGRAPH, segment)


def detect_structure(text: str) -> list[StructureBlock]:
    """Split normalized text into an ordered list of typed blocks.

    What problem it solves: chunking needs to treat headings, lists, and
    code blocks differently from ordinary paragraphs; this is the pass
    that tells it which is which.

    Complexity: O(n) in text length — one regex pass to extract code
    fences, then one blank-line split and one O(1) classification per
    resulting segment.
    """
    blocks: list[StructureBlock] = []
    pos = 0

    for match in _CODE_FENCE.finditer(text):
        if match.start() > pos:
            blocks.extend(_classify_text_region(text[pos : match.start()]))
        blocks.append(StructureBlock(BlockType.CODE_BLOCK, match.group(1)))
        pos = match.end()

    if pos < len(text):
        blocks.extend(_classify_text_region(text[pos:]))

    return blocks


def _classify_text_region(region: str) -> list[StructureBlock]:
    segments = [s for s in region.split("\n\n") if s.strip()]
    return [_classify_segment(segment.strip("\n")) for segment in segments]
