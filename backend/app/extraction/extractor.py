"""LLM semantic extraction.

Turns one chunk of text into a validated ExtractionResult. The LLM call
itself is injected (llm_fn: prompt in -> raw response text out) rather
than hardcoded to a specific provider SDK — same dependency-injection
pattern as chunking V2/V3's embed_fn. This keeps the retry/parse/
validate logic fully testable without a real API key or network call,
and decouples this module from whichever provider settings.llm_provider
ends up naming.
"""

import json
import logging
from collections.abc import Callable

from app.extraction.prompts import build_extraction_prompt
from app.extraction.schemas import ExtractionResult
from app.extraction.validator import validate_extraction_result
from app.models import Chunk

logger = logging.getLogger(__name__)

LlmFn = Callable[[str], str]


def extract_from_chunk(chunk: Chunk, llm_fn: LlmFn) -> ExtractionResult | None:
    """Extract entities/relationships from one chunk via the LLM.

    What problem it solves: turns unstructured chunk text into the
    structured, schema-constrained shape (entities + relationships) the
    graph builder (Phase 17) needs.

    Algorithm: build prompt -> call llm_fn -> parse JSON -> Pydantic
    validation -> cross-field validation. On failure, retry ONCE with a
    stricter reminder appended to the prompt; if that also fails, log
    and return None rather than raise — CLAUDE.md is explicit that one
    bad chunk must never crash the whole ingestion pipeline.

    Complexity: 1-2 llm_fn calls (the expensive, network-bound part in
    practice); everything else is O(n) in response size.

    Limitations: does not talk to any specific LLM provider's SDK itself
    — that's the caller's job when constructing llm_fn. Does not do
    entity resolution/deduplication across chunks (Phase 16) — this is
    single-chunk extraction only.
    """
    result = _try_extract(chunk.text, llm_fn, strict=False)
    if result is not None:
        return result

    logger.warning("extraction failed validation for chunk %s, retrying", chunk.id)
    result = _try_extract(chunk.text, llm_fn, strict=True)
    if result is not None:
        return result

    logger.error("extraction failed for chunk %s after retry, skipping", chunk.id)
    return None


def _try_extract(chunk_text: str, llm_fn: LlmFn, strict: bool) -> ExtractionResult | None:
    prompt = build_extraction_prompt(chunk_text)
    if strict:
        prompt += (
            "\n\nIMPORTANT: your previous response was invalid. "
            "Return ONLY the JSON object, no markdown fences, no explanation."
        )

    raw_response = llm_fn(prompt)

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        return None

    try:
        result = ExtractionResult.model_validate(data)
    except Exception:
        return None

    if validate_extraction_result(result):
        return None

    return result
