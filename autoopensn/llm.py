"""The language-model layer.

One wrapper over the sister repository's ``LLMClient``, which already handles
the TAMU AI Chat gateway, the Groq fallback, and the fact that the TAMU client
silently discards ``temperature`` and ``max_tokens``. Nothing here re-implements
any of that; it adds the four things a workflow tool needs and a chat client
does not:

**A timeout.** The TAMU client exposes none. A call is run on a worker thread
and abandoned, exactly as the sister repository's reviewer does. The request
keeps running until the provider closes it, which we cannot prevent; what we can
prevent is a study held hostage by one slow completion.

**JSON that survives a chatty model.** Models fence their JSON, apologise before
it, and explain after it. ``complete_json`` finds the object and parses it, and
returns a typed failure rather than raising, because the caller's next move is
usually to retry with the error rather than to crash.

**Model choice.** Two models, picked per task, following the sister repository's
own measured split: the fast model for structured output chosen from a bounded
menu, the thorough model for open-ended explanation. The ids are read from the
sister repository's config rather than duplicated here.

**An offline mode.** ``ScriptedLLM`` returns canned text in order. Every test in
this repository uses it, so the suite never touches the network, and a live call
is opt-in.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

DEFAULT_TIMEOUT = 120.0

# Enough for a spec (small) or an explanation (a few paragraphs). Ignored by the
# TAMU gateway, which accepts neither temperature nor max_tokens, and honoured
# by the OpenAI-compatible providers.
DEFAULT_MAX_TOKENS = 2000


class LLMError(Exception):
    """A completion could not be obtained."""


class LLMUnavailable(LLMError):
    """No provider is configured, or the sister repository is absent."""


@dataclass
class Completion:
    """One model response, with what it cost and where it came from."""

    text: str
    model: Optional[str] = None
    provider: Optional[str] = None
    elapsed: float = 0.0

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


# --- JSON extraction --------------------------------------------------------

_FENCE = re.compile(r"```(?:json|yaml|ya?ml)?\s*(.*?)```", re.DOTALL)


def extract_block(raw: str) -> str:
    """The fenced block from a response, or the response unchanged.

    Models fence structured output more often than not, and the fence is not
    part of the document. Stripping it here means every caller does not.
    """
    if not raw:
        return ""
    match = _FENCE.search(raw)
    return match.group(1).strip() if match else raw.strip()


def extract_json(raw: str) -> Optional[dict]:
    """Parse a JSON object out of a response, tolerating fences and prose."""
    text = extract_block(raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# --- the interface ----------------------------------------------------------


class LLM(ABC):
    """Something that can complete a chat exchange."""

    name: str = "llm"

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: Optional[str] = None,
    ) -> Completion:
        """One completion. Raises ``LLMError`` on failure."""

    def complete_json(self, system: str, user: str, **kwargs) -> tuple[Optional[dict], Completion]:
        """A completion parsed as JSON. Returns ``(None, completion)`` on garbage."""
        completion = self.complete(system, user, **kwargs)
        return extract_json(completion.text), completion


class ProviderLLM(LLM):
    """The real thing: the sister repository's client, with a timeout."""

    name = "provider"

    def __init__(
        self,
        client: Any,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        fast_model: Optional[str] = None,
        thorough_model: Optional[str] = None,
    ):
        self.client = client
        self.timeout = timeout
        self.fast_model = fast_model
        self.thorough_model = thorough_model

    @classmethod
    def from_env(cls, *, timeout: float = DEFAULT_TIMEOUT, quiet: bool = True) -> "ProviderLLM":
        """Build from the sister repository's configuration and ``.env``."""
        from autoopensn import kp_bridge  # noqa: PLC0415

        if not kp_bridge.available():
            raise LLMUnavailable(
                "the sister repository is not present, so no provider is configured. "
                "Set AUTOOPENSN_KP_REPO to your Code_assistant_TAU checkout."
            )
        try:
            client = kp_bridge.llm_client(quiet=quiet)
        except Exception as exc:
            raise LLMUnavailable(f"could not construct an LLM client: {exc}") from exc

        # The sister client tries TAMU, then Groq, then OpenAI, and when it has
        # no key for any of them it settles on a local Ollama without a word.
        # That is how this project once spent a day reporting results from a
        # local llama3.2 while everyone believed the TAMU key was in use. A
        # missing key is a configuration error with a known fix, and it is
        # reported as one. Using Ollama on purpose is one variable away.
        if getattr(client, "provider", None) == "ollama" and not os.getenv("AUTOOPENSN_ALLOW_OLLAMA"):
            raise LLMUnavailable(
                "no API key was found, so the only model available is a local Ollama, "
                "and that is not used without being asked for. Put TAMU_API_KEY in "
                f"{kp_bridge.ENV_FILE} (copy .env.example), or set "
                "AUTOOPENSN_ALLOW_OLLAMA=1 to use Ollama deliberately."
            )

        models = kp_bridge.llm_models()
        return cls(
            client,
            timeout=timeout,
            fast_model=models.get("fast"),
            thorough_model=models.get("thorough"),
        )

    # --- model choice ---

    def model_for(self, task: str) -> Optional[str]:
        """Which model to use for a task.

        Follows the sister repository's measured split. Structured output picked
        from a menu we supply is a lookup, and the two models are within a few
        percent there while the small one is several times faster. Open-ended
        explanation is where the larger model's extra development shows.

        Returns None when the operator has set ``TAMU_MODEL`` or when a
        non-TAMU provider is active, which leaves the client's own choice alone.
        """
        if os.getenv("TAMU_MODEL"):
            return None
        if getattr(self.client, "provider", None) != "tamu":
            return None
        return self.thorough_model if task == "thorough" else self.fast_model

    # --- the call ---

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: Optional[str] = None,
    ) -> Completion:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        started = time.monotonic()
        try:
            text = self._call(messages, temperature, max_tokens, model)
        except FutureTimeout as exc:
            raise LLMError(
                f"the model did not answer within {self.timeout:.0f}s"
            ) from exc
        except Exception as exc:
            raise LLMError(str(exc)) from exc

        return Completion(
            text=text or "",
            model=model or getattr(self.client, "model", None),
            provider=getattr(self.client, "provider", None),
            elapsed=time.monotonic() - started,
        )

    def _call(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        model: Optional[str],
    ) -> str:
        """Blocking provider call, abandoned after ``timeout`` seconds.

        The same device the sister repository's reviewer uses, and for the same
        reason: its client has no timeout of its own, so the call goes on a
        worker thread and we stop waiting. We cannot cancel it. The cost of
        giving up is one wasted completion; the cost of not giving up is a
        study that never finishes.
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self.client._call_with_fallback, messages, temperature, max_tokens, model
            )
            return future.result(timeout=self.timeout)


# --- offline -----------------------------------------------------------------


class ScriptedLLM(LLM):
    """Returns canned responses in order. For tests and demonstrations.

    Every test in this repository uses this. A test suite that reaches a paid
    API is a test suite that is run rarely and trusted less, and a stage whose
    only exercise is a live call is a stage nobody refactors.
    """

    name = "scripted"

    def __init__(self, responses: Sequence[str], *, model: str = "scripted"):
        self.responses = list(responses)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: Optional[str] = None,
    ) -> Completion:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": model,
            }
        )
        if not self.responses:
            raise LLMError("ScriptedLLM ran out of responses")
        return Completion(text=self.responses.pop(0), model=self.model, provider="scripted")


# --- construction ------------------------------------------------------------


def default_llm(*, timeout: float = DEFAULT_TIMEOUT, quiet: bool = True) -> LLM:
    """The provider-backed client, or ``LLMUnavailable`` with the reason."""
    return ProviderLLM.from_env(timeout=timeout, quiet=quiet)


def available() -> bool:
    """Whether a provider looks configured, without constructing a client.

    Checked by the interface so it can disable a button and say why, rather
    than offering an action that fails on click.
    """
    from autoopensn import kp_bridge  # noqa: PLC0415

    if not kp_bridge.available():
        return False
    keys = ("TAMU_assintant_key", "TAMU_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(kp_bridge.repo_path() / ".env")
    except Exception:
        pass
    return any(
        os.getenv(key) and not str(os.getenv(key)).startswith("your-") for key in keys
    )
