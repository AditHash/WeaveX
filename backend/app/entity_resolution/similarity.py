"""String similarity metrics — built from scratch, not imported from a
library. Entity resolution is core to what this project demonstrates,
not infrastructure, so this is implemented rather than reached for.

Two metrics, each catching a different kind of near-miss:
  - Levenshtein similarity: catches typos / minor spelling variation
    ("Pydantic" vs "Pydnatic").
  - Jaccard token similarity: catches partial containment / reordering
    ("GPT-5" vs "OpenAI GPT-5" — one is a strict superset of words of
    the other, which edit distance scores poorly since it's dominated by
    the extra word's length).
"""


def levenshtein_distance(a: str, b: str) -> int:
    """Minimum single-character edits (insert/delete/substitute) to turn
    `a` into `b` — the standard way to quantify "these are almost the
    same string, just typo'd".

    Algorithm: classic dynamic programming. dp[i][j] = edit distance
    between a[:i] and b[:j], built up from empty-string base cases.

    Complexity: O(n*m) time and space, n=len(a), m=len(b). Fine for
    entity names (short strings) — a long-string use case would need a
    banded/Ukkonen variant, which this project doesn't have.

    Limitations: purely character-level — doesn't recognize "GPT-5" and
    "OpenAI GPT-5" as related by containment, only that they're far
    apart character-by-character. jaccard_token_similarity below is what
    catches that case instead.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[n][m]


def levenshtein_similarity(a: str, b: str) -> float:
    """Normalize levenshtein_distance to [0, 1]: 1.0 = identical, 0.0 =
    completely different (distance equals the longer string's length).
    Normalized by the longer string so scores are comparable across
    names of different lengths."""
    if not a and not b:
        return 1.0
    distance = levenshtein_distance(a, b)
    return 1.0 - distance / max(len(a), len(b))


def jaccard_token_similarity(a: str, b: str) -> float:
    """Token-set overlap: |intersection| / |union| of whitespace tokens.

    What problem it solves: catches containment/reordering edit distance
    misses — "GPT-5" and "OpenAI GPT-5" share the token "gpt-5" even
    though the strings differ by a whole extra word.

    Complexity: O(n+m) in token count.

    Limitations: token-level, not character-level — "GPT5" and "GPT-5"
    are different single tokens with zero overlap unless normalized
    first. Meant to run on already-normalized or pre-tokenized input,
    not necessarily raw names — see normalizer.py.
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
