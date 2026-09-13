"""Phase 12 (client layer) tests — GroqLlmClient.

Uses httpx.MockTransport — no API key needed, no live request made.
Integration test at the bottom wires this client's as_llm_fn() straight
into extract_from_chunk on a real corpus chunk, proving the whole
extraction pipeline works end-to-end with a real (mocked) LLM call.
"""

import json

import httpx
import pytest

from app.extraction import extract_from_chunk
from app.llm import GroqLlmClient, LlmClientError

from tests.fixtures import FastAPICorpus


def _client_with_transport(handler) -> GroqLlmClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return GroqLlmClient(api_key="test-key", http_client=http_client)


def test_missing_api_key_raises_at_construction():
    with pytest.raises(LlmClientError):
        GroqLlmClient(api_key=None)


def test_successful_completion_returns_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entities": []}'}}]},
        )

    client = _client_with_transport(handler)
    assert client.complete("extract entities from: ...") == '{"entities": []}'


def test_request_shape_sends_prompt_and_model():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["authorization"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    client = _client_with_transport(handler)
    client.complete("my prompt")

    assert captured["body"]["messages"] == [{"role": "user", "content": "my prompt"}]
    assert captured["body"]["model"] == "openai/gpt-oss-120b"
    assert captured["auth"] == "Bearer test-key"


def test_non_200_response_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    client = _client_with_transport(handler)
    with pytest.raises(LlmClientError, match="401"):
        client.complete("prompt")


def test_malformed_response_shape_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _client_with_transport(handler)
    with pytest.raises(LlmClientError):
        client.complete("prompt")


def test_as_llm_fn_plugs_directly_into_extract_from_chunk():
    corpus = FastAPICorpus()
    chunk = corpus.chunks[1]  # "FastAPI uses Pydantic...Starlette..."

    canned_extraction = json.dumps(
        {
            "entities": [
                {"name": "FastAPI", "entity_type": "Technology"},
                {"name": "Pydantic", "entity_type": "Technology"},
            ],
            "relationships": [
                {
                    "source": "FastAPI", "target": "Pydantic",
                    "relationship_type": "USES",
                    "evidence": "FastAPI uses Pydantic for data validation.",
                    "confidence": 0.94,
                }
            ],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": canned_extraction}}]}
        )

    client = _client_with_transport(handler)
    result = extract_from_chunk(chunk, client.as_llm_fn())

    assert result is not None
    assert {e.name for e in result.entities} == {"FastAPI", "Pydantic"}
