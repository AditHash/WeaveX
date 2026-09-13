"""Real LLM calls via the Groq API (OpenAI-compatible chat completions).

External hosted API call, not a self-hosted/trained model — fits
CLAUDE.md's "an LLM API may be used" allowance. Serves
app.config.settings.llm_model (default: the open-weight gpt-oss-120b).

Produces an LlmFn (Callable[[str], str]) matching the exact interface
extract_from_chunk (Phase 12) already depends on via dependency
injection — nothing about extractor.py changes now that a real client
exists; it just gets called with this instead of a test fake.
"""

from collections.abc import Callable

import httpx

from app.config import settings as default_settings

_GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

LlmFn = Callable[[str], str]


class LlmClientError(RuntimeError):
    """Raised when the LLM API call fails, or isn't configured."""


class GroqLlmClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or default_settings.llm_api_key
        self._model = model or default_settings.llm_model
        # http_client is injectable so tests can pass an httpx.MockTransport
        # -- same dependency-injection pattern used elsewhere in this
        # codebase (VectorStore's embed_fn, HuggingFaceEmbeddingClient).
        self._http_client = http_client or httpx.Client()

        if not self._api_key:
            raise LlmClientError(
                "llm_api_key is not configured "
                "(set LLM_API_KEY or pass api_key explicitly)"
            )

    def complete(self, prompt: str) -> str:
        """Send one prompt, get back the model's raw text response.

        What problem it solves: extract_from_chunk (Phase 12) needs
        something to actually call the LLM — this is that something, in
        place of a test fake.

        Complexity: one network round-trip per call — network-latency
        bound in practice, and the dominant cost in extract_from_chunk's
        1-2 calls per chunk (happy path + one retry).

        Limitations: no retry/backoff on transient network failure at
        this layer — raises LlmClientError and lets the caller decide.
        extract_from_chunk already has its own retry-on-invalid-output
        logic; a network-level retry here would be a second, different
        retry policy stacked on top of that, which isn't built — a
        network failure during extraction just fails that attempt today.
        """
        response = self._http_client.post(
            _GROQ_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        if response.status_code != 200:
            raise LlmClientError(
                f"LLM request failed: {response.status_code} {response.text}"
            )

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LlmClientError(f"unexpected LLM response shape: {data}") from exc

    def as_llm_fn(self) -> LlmFn:
        """Return this client's complete() as a plain callable, matching
        the LlmFn interface extract_from_chunk expects."""
        return self.complete
