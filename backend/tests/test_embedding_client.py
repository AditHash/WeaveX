"""Phase 20 tests — HuggingFaceEmbeddingClient.

Uses httpx.MockTransport (part of httpx, already a dependency) instead
of a real network call — no API key needed to run these tests, and no
live request is ever made.
"""

import httpx
import pytest

from app.chunking import chunk_document_semantic
from app.embedding import EmbeddingClientError, HuggingFaceEmbeddingClient
from app.models import Document

from tests.fixtures import FASTAPI_CORPUS_TEXT


def _client_with_transport(handler) -> HuggingFaceEmbeddingClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return HuggingFaceEmbeddingClient(api_key="test-key", http_client=http_client)


def test_missing_api_key_raises_at_construction():
    with pytest.raises(EmbeddingClientError):
        HuggingFaceEmbeddingClient(api_key=None)


def test_pooled_sentence_response_returns_vector_directly():
    # sentence-similarity-tagged models (the default here) return an
    # already-pooled flat vector.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[0.1, 0.2, 0.3])

    client = _client_with_transport(handler)
    assert client.embed("FastAPI uses Pydantic.") == [0.1, 0.2, 0.3]


def test_per_token_response_is_mean_pooled():
    # Some feature-extraction endpoints return raw per-token vectors
    # instead — client must mean-pool them client-side rather than error.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[1.0, 2.0], [3.0, 4.0]])

    client = _client_with_transport(handler)
    assert client.embed("some text") == [2.0, 3.0]  # mean of [1,3] and [2,4]


def test_request_shape_sends_text_and_waits_for_model():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["authorization"]
        captured["url"] = str(request.url)
        return httpx.Response(200, json=[1.0])

    client = _client_with_transport(handler)
    client.embed("some text")

    assert captured["body"]["inputs"] == "some text"
    assert captured["body"]["options"]["wait_for_model"] is True
    assert captured["auth"] == "Bearer test-key"
    assert "sentence-transformers/all-MiniLM-L6-v2" in captured["url"]


def test_non_200_response_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    client = _client_with_transport(handler)
    with pytest.raises(EmbeddingClientError, match="503"):
        client.embed("some text")


def test_malformed_response_shape_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "unexpected"})

    client = _client_with_transport(handler)
    with pytest.raises(EmbeddingClientError):
        client.embed("some text")


def test_as_embed_fn_plugs_directly_into_semantic_chunker():
    # Integration proof-point: the real client, wired straight into
    # Phase 10's chunker, produces a working chunk list end-to-end.
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        text = json.loads(request.content)["inputs"]
        # deterministic per-text vector so the chunker's similarity
        # comparisons are meaningful rather than constant
        return httpx.Response(200, json=[float(len(text)), 0.0])

    client = _client_with_transport(handler)
    document = Document(title="test", content=FASTAPI_CORPUS_TEXT)

    chunks = chunk_document_semantic(
        document, client.as_embed_fn(), max_chunk_size=1000, similarity_threshold=0.99
    )
    assert len(chunks) >= 1
    assert chunks[0].document_id == document.id
