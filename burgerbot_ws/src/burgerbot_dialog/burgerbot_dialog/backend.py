"""Talking to a local model server over HTTP. Thin on purpose.

Sits behind a Protocol so the node never imports a vendor SDK and so tests can
substitute a fake without a network. Adding a cloud backend later is a new file
implementing the same three methods, not a refactor of anything here.

Uses `requests` rather than an SDK because `python3-requests` has a rosdep key,
so the robot's `rosdep install` resolves it like any other dependency and
nothing here needs a pip step. Both Ollama and vLLM expose an
OpenAI-compatible `/chat/completions`, so one client covers the pair.

The rule the rest of the package depends on: **complete() never raises and
never overruns its deadline.** Connection refused, a 500, a timeout, a body
that is not JSON -- all of them come back as a populated `error` field. Not
because the distinctions do not matter, but because to the caller they are the
same thing: no answer this turn. One error path means the worker thread has no
route to dying quietly and leaving the robot permanently mute, which is the
failure this whole file is arranged to prevent.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class LLMResult:
    """What came back, or why nothing did."""

    text: str = ""
    latency: float = 0.0
    model: str = ""
    #: Empty on success. One of: "unreachable", "timeout", "http", "invalid".
    #: The kind rather than the detail, because the robot says something
    #: different for each and `detail` carries the rest for the log.
    error: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text)


class Backend(Protocol):
    name: str

    def complete(self, messages: List[Dict[str, str]], deadline: float) -> LLMResult:
        ...

    def health(self) -> "LLMResult":
        ...


@dataclass
class OpenAICompatBackend:
    """Any server speaking OpenAI's chat-completions API.

    Verified shapes this targets: Ollama at http://localhost:11434/v1 and vLLM
    at http://localhost:8000/v1. Both accept the same request body; they differ
    in whether they honour `response_format`, which is what `use_schema` is
    for.
    """

    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3:14b"
    #: Low, not zero. Zero makes a companion repeat itself word for word across
    #: turns, which is unsettling in a way that a slightly different phrasing
    #: is not; high makes it ignore the output contract.
    temperature: float = 0.4
    #: A two-sentence reply plus four short fields needs very little. Capping
    #: it is the cheapest latency control available and it also stops a model
    #: that has decided to write an essay.
    max_tokens: int = 200
    #: Send a JSON-schema `response_format`. vLLM honours it; Ollama's support
    #: depends on version. Turning it off falls back to relying on the prompt
    #: and on schema.py's validator, which is why the validator assumes nothing.
    use_schema: bool = True
    schema: Optional[Dict[str, Any]] = None
    api_key: str = ""
    name: str = "local"

    _session: Any = field(default=None, repr=False)

    def _client(self):
        # Imported lazily and kept in a session so connections are reused. A
        # fresh TCP handshake per turn is measurable next to a one-second
        # round trip, and the session is also where a future retry policy
        # would hang.
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _body(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.use_schema and self.schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": self.schema, "strict": True},
            }
        return body

    def complete(self, messages: List[Dict[str, str]], deadline: float) -> LLMResult:
        """One turn. Returns within `deadline` seconds, whatever happens."""
        started = time.monotonic()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            import requests
        except ImportError:
            return LLMResult(
                error="unreachable",
                detail="python3-requests is not installed",
            )

        try:
            response = self._client().post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=self._body(messages),
                headers=headers,
                timeout=deadline,
            )
        except requests.exceptions.Timeout:
            return LLMResult(error="timeout", latency=time.monotonic() - started,
                             detail=f"no response within {deadline:.1f}s")
        except requests.exceptions.RequestException as exc:
            # Connection refused is by far the most common case here, and it
            # means exactly one thing: the model server is not running. Worth
            # saying plainly rather than surfacing a stack trace.
            return LLMResult(error="unreachable", latency=time.monotonic() - started,
                             detail=str(exc))
        except Exception as exc:  # pragma: no cover - belt and braces
            return LLMResult(error="unreachable", latency=time.monotonic() - started,
                             detail=repr(exc))

        latency = time.monotonic() - started

        if response.status_code != 200:
            # The body usually says something useful (an unknown model name,
            # a context length overflow) and is worth keeping, truncated.
            return LLMResult(
                error="http", latency=latency,
                detail=f"HTTP {response.status_code}: {response.text[:300]}",
            )

        try:
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return LLMResult(error="invalid", latency=latency,
                             detail=f"unexpected response shape: {exc}")

        if not isinstance(text, str):
            return LLMResult(error="invalid", latency=latency,
                             detail=f"content was {type(text).__name__}")

        return LLMResult(text=text, latency=latency,
                         model=payload.get("model", self.model))

    def health(self) -> LLMResult:
        """A one-line probe, for saying something useful at startup.

        Checking at startup rather than discovering it on the first thing
        somebody says to the robot: a wrong base_url or a server that is not
        running should be a line in the log at boot, not a confused face
        several minutes later.
        """
        return self.complete(
            [{"role": "user", "content": 'Reply with exactly: {"say":"ok"}'}],
            deadline=10.0,
        )
