"""Real embedding generation via the HuggingFace Inference API.

An external hosted API call (not a locally-run model) — fits CLAUDE.md's
"an embedding model/API may be used" allowance without touching the
"do not train or self-host a model" line. Uses a sentence-transformers
model (see app.config.settings.embedding_model), which HF's Inference
API pools into a single sentence vector server-side for models tagged
"sentence-similarity" — this client also handles the (rarer) case where
a model instead returns raw per-token vectors, by mean-pooling client-side.

Produces an EmbedFn (Callable[[str], list[float]]) matching the exact
interface chunking V2/V3 (Phases 10-11) already depend on via dependency
injection — nothing about those modules changes now that a real
implementation exists; they just get called with this instead of a
test fake.
"""

from collections.abc import Callable

import httpx

from app.config import settings as default_settings

_HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/{model}"

EmbedFn = Callable[[str], list[float]]


class EmbeddingClientError(RuntimeError):
    """Raised when the embedding API call fails, or isn't configured."""


class HuggingFaceEmbeddingClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or default_settings.embedding_api_key
        self._model = model or default_settings.embedding_model
        # http_client is injectable so tests can pass an httpx.MockTransport
        # -- same dependency-injection pattern used for embed_fn/llm_fn
        # elsewhere in this codebase, applied one layer deeper here.
        self._http_client = http_client or httpx.Client()

        if not self._api_key:
            raise EmbeddingClientError(
                "embedding_api_key is not configured "
                "(set EMBEDDING_API_KEY or pass api_key explicitly)"
            )

    def embed(self, text: str) -> list[float]:
        """Get the embedding vector for one piece of text.

        What problem it solves: produces the real vector chunking V2/V3
        and future retrieval phases need, in place of a test fake.

        Complexity: one network round-trip per call — network-latency
        bound in practice. Callers embedding one paragraph at a time (as
        chunk_document_semantic does) should expect this to dominate
        wall-clock time; batching multiple texts per request is a
        documented future optimization (Phase 38 "Performance"), not
        built here.

        Limitations: no retry/backoff on transient network failure —
        raises EmbeddingClientError and lets the caller decide, same
        philosophy as the Voyage client this replaced. HF-hosted models
        can also return a 503 while "warming up" (cold start) — that
        surfaces as EmbeddingClientError like any other non-200, not
        retried automatically; a caller that wants to wait for warm-up
        can catch the error and retry itself.
        """
        response = self._http_client.post(
            _HF_INFERENCE_URL.format(model=self._model),
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30.0,
        )
        if response.status_code != 200:
            raise EmbeddingClientError(
                f"embedding request failed: {response.status_code} {response.text}"
            )

        data = response.json()
        return self._to_vector(data)

    def _to_vector(self, data: object) -> list[float]:
        """Normalize HF's response shape to a flat list[float].

        Sentence-similarity-tagged models (like the default here) return
        an already-pooled vector: list[float]. Some feature-extraction
        endpoints instead return per-token vectors: list[list[float]] —
        mean-pool those client-side rather than error out, so this client
        works across both response shapes without the caller needing to
        know which one a given model happens to use.
        """
        if not isinstance(data, list) or not data:
            raise EmbeddingClientError(f"unexpected embedding response shape: {data}")

        if isinstance(data[0], (int, float)):
            return [float(x) for x in data]

        if isinstance(data[0], list):
            dimension = len(data[0])
            sums = [0.0] * dimension
            for token_vector in data:
                for i, value in enumerate(token_vector):
                    sums[i] += value
            return [s / len(data) for s in sums]

        raise EmbeddingClientError(f"unexpected embedding response shape: {data}")

    def as_embed_fn(self) -> EmbedFn:
        """Return this client's embed() as a plain callable, matching the
        EmbedFn interface chunking V2/V3 expect — pluggable directly into
        chunk_document_semantic/chunk_document_entity_aware."""
        return self.embed
