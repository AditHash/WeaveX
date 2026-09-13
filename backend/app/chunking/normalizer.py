"""Text normalization — first step of the chunking pipeline, no LLM.

Cleans incidental noise (Windows line endings, repeated whitespace,
non-canonical unicode, unbounded blank-line runs) without touching
structural information Phase 8's structure detector needs (heading
underlines like "====" or "----") — normalization must not destroy
meaningful formatting, only incidental mess.
"""

import re
import unicodedata

_REPEATED_SPACES = re.compile(r"[ \t]{2,}")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Canonicalize incidental text noise before chunking.

    What problem does it solve: raw ingested text is inconsistent (CRLF
    vs LF, double spaces, decomposed vs composed unicode accents,
    unbounded blank-line runs) and that inconsistency would otherwise
    leak into chunk boundaries, offsets, and entity-name comparisons.

    Complexity: O(n) in text length — each step is one linear regex pass
    or a linear unicodedata pass; there's no nesting between them.

    Limitations: purely syntactic cleanup. It does not understand
    document structure (that's Phase 8) and never removes/reorders
    content, only incidental whitespace/encoding noise.

    Steps, in order (order matters — see inline comments):
    """
    # 1. Unicode NFC: canonicalize composed vs decomposed accented chars
    #    (e.g. "í" as one codepoint vs "i" + combining acute) so the same
    #    name always compares equal regardless of source encoding.
    normalized = unicodedata.normalize("NFC", text)

    # 2. Line endings: CRLF/CR -> LF, so every later rule only has to
    #    reason about "\n".
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Collapse runs of 2+ spaces/tabs to one. Never touches "\n" —
    #    blank-line handling is a separate rule (step 5).
    normalized = _REPEATED_SPACES.sub(" ", normalized)

    # 4. Strip trailing whitespace per line. Must run before step 5: a
    #    "blank-ish" line (just spaces) needs to become truly empty
    #    first, or the blank-line collapse below won't recognize it.
    normalized = _TRAILING_WHITESPACE.sub("", normalized)

    # 5. Cap consecutive blank lines at one, so paragraph breaks stay
    #    meaningful (one blank line = one paragraph boundary) without
    #    unbounded gaps.
    normalized = _EXCESS_BLANK_LINES.sub("\n\n", normalized)

    # 6. Trim the whole document.
    return normalized.strip()
