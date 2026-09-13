"""Phase 12 tests — extract_from_chunk (prompt -> parse -> validate -> retry).

Uses a real chunk from FastAPICorpus and a canned JSON response shaped
exactly like that chunk's actual content, rather than an arbitrary
extraction example.
"""

import json

from app.extraction import extract_from_chunk

from tests.fixtures import FastAPICorpus


def _pydantic_starlette_chunk():
    corpus = FastAPICorpus()
    # chunks[1] == "FastAPI uses Pydantic for data validation and
    # Starlette for web functionality."
    return corpus.chunks[1]


_VALID_RESPONSE = json.dumps(
    {
        "entities": [
            {"name": "FastAPI", "entity_type": "Technology"},
            {"name": "Pydantic", "entity_type": "Technology"},
            {"name": "Starlette", "entity_type": "Technology"},
        ],
        "relationships": [
            {
                "source": "FastAPI", "target": "Pydantic", "relationship_type": "USES",
                "evidence": "FastAPI uses Pydantic for data validation.",
                "confidence": 0.94,
            },
            {
                "source": "FastAPI", "target": "Starlette", "relationship_type": "USES",
                "evidence": "Starlette for web functionality.",
                "confidence": 0.93,
            },
        ],
    }
)


def test_happy_path_extracts_real_chunk_content():
    chunk = _pydantic_starlette_chunk()
    result = extract_from_chunk(chunk, llm_fn=lambda prompt: _VALID_RESPONSE)

    assert result is not None
    assert {e.name for e in result.entities} == {"FastAPI", "Pydantic", "Starlette"}
    assert len(result.relationships) == 2
    assert all(r.relationship_type == "USES" for r in result.relationships)


def test_llm_fn_receives_prompt_containing_chunk_text():
    chunk = _pydantic_starlette_chunk()
    received_prompts = []

    def capturing_llm(prompt: str) -> str:
        received_prompts.append(prompt)
        return _VALID_RESPONSE

    extract_from_chunk(chunk, llm_fn=capturing_llm)
    assert chunk.text in received_prompts[0]


def test_retries_once_on_malformed_json_then_succeeds():
    chunk = _pydantic_starlette_chunk()
    calls = []

    def flaky_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "this is not json at all"
        return _VALID_RESPONSE

    result = extract_from_chunk(chunk, llm_fn=flaky_llm)

    assert result is not None
    assert len(calls) == 2
    assert "IMPORTANT" in calls[1]  # second attempt used the stricter prompt


def test_gives_up_after_retry_also_fails():
    chunk = _pydantic_starlette_chunk()
    calls = []

    def always_broken_llm(prompt: str) -> str:
        calls.append(prompt)
        return "still not json"

    result = extract_from_chunk(chunk, llm_fn=always_broken_llm)

    assert result is None
    assert len(calls) == 2  # tried once, retried once, then gave up — no infinite loop


def test_referentially_invalid_relationship_is_rejected():
    chunk = _pydantic_starlette_chunk()
    broken_response = json.dumps(
        {
            "entities": [{"name": "FastAPI", "entity_type": "Technology"}],
            "relationships": [
                {
                    # "Pydantic" was never listed as an entity above.
                    "source": "FastAPI", "target": "Pydantic",
                    "relationship_type": "USES",
                    "evidence": "FastAPI uses Pydantic.", "confidence": 0.9,
                }
            ],
        }
    )

    result = extract_from_chunk(chunk, llm_fn=lambda prompt: broken_response)
    assert result is None
