"""The one place the pipeline talks to a language model.

Every lookup, colour and mapping question is the same shape — some text in,
one JSON document out — so a model is wrapped as exactly that: `structured()`
takes a prompt and a JSON schema and returns a dict that satisfies it. The
callers never see content blocks, finish reasons or tool results, and never
know which provider answered.

One request per question, the response constrained to the schema, so
parsing cannot fail. There is no web-search variant: the free tier has no
search-grounding quota, so retrieval is done by `retrieve.py` and the
documents arrive inside the prompt. That also keeps the model's job to
reading, which every model does the same way.

`Gemini` is the default and the only provider installed by default: the
project runs on the Gemini API free tier. The free tier is metered per
minute and per day, so the client retries 429s and 503s with backoff, falls
back to a lighter model when the preferred one stays unavailable, and the
callers keep to one request per question (the whole pipeline is four
requests per clip). `claude.py` holds the alternative, behind the `claude`
extra.

Nothing here runs without an API key; `configured()` says whether one is
present so the pipeline can decide up front instead of failing mid-run.
"""

import json
import os
import sys
import time
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()

# H2S_LLM_PROVIDER picks the backend: "gemini" (default) or "claude".
PROVIDER_VAR = "H2S_LLM_PROVIDER"
DEFAULT_PROVIDER = "gemini"
# H2S_LLM_MODEL overrides the provider's default model.
MODEL_VAR = "H2S_LLM_MODEL"

# A mid-tier Flash that the free tier actually serves for prompts of the
# size this pipeline sends (measured 2026-09-22: 3.8-flash answered 429 on
# every long prompt, 3.7-flash 503, 3.6-flash ~30 s per call, 3.5-flash-lite
# ~3 s). Set H2S_LLM_MODEL=gemini-3.5-flash-lite to iterate fast at some
# cost to the mapping step's judgement.
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
# Where a request goes when the preferred model is over quota or under load
# even after the retries: the lightest current Flash, which in practice is
# always available. Set to "" to disable.
GEMINI_FALLBACK_MODEL = os.environ.get("H2S_LLM_FALLBACK_MODEL", "gemini-3.5-flash-lite")
# The SDK also reads GOOGLE_API_KEY; this is the one the docs lead with.
GEMINI_API_KEY_VAR = "GEMINI_API_KEY"

# Room for a long cast list without hitting the ceiling mid-document.
MAX_TOKENS = 16000

# Server trouble (503 under load, 5xx) clears in seconds, so the SDK retries
# it briefly. A 429 is deliberately *not* on this list: on the free tier it
# means a quota the preferred model has run out of — per minute, per day or
# tokens per minute — and waiting a minute on it just stalls the run (measured
# at 2½ minutes per call before this was learnt). A 429 goes straight to the
# fallback model instead; see `Gemini._call`.
RETRY_ATTEMPTS = 4
RETRY_INITIAL_DELAY = 2.0
RETRY_MAX_DELAY = 20.0
RETRY_STATUS_CODES = [500, 502, 503, 504]
REQUEST_TIMEOUT_MS = 180_000
# When the fallback model is over quota too, one wait this long and one more
# try — enough for a per-minute limit to reset, not so long the user wonders
# whether the run has died.
QUOTA_WAIT_SECONDS = 30.0


class KnowledgeError(RuntimeError):
    pass


class LLM(Protocol):
    """What the pipeline needs from a model: one schema-shaped answer per question."""

    def structured(self, system: str, prompt: str, schema: dict) -> dict: ...


def provider() -> str:
    return (os.environ.get(PROVIDER_VAR) or DEFAULT_PROVIDER).strip().lower()


def api_key_var(name: str | None = None) -> str:
    """The environment variable the chosen provider's key lives in."""
    name = name or provider()
    if name == "gemini":
        return GEMINI_API_KEY_VAR
    if name == "claude":
        from hearing_to_seeing.knowledge.claude import API_KEY_VAR

        return API_KEY_VAR
    raise KnowledgeError(f"{PROVIDER_VAR} 값이 잘못되었습니다 (gemini | claude): {name!r}")


def configured(name: str | None = None) -> bool:
    name = name or provider()
    if name == "gemini":
        return bool(os.environ.get(GEMINI_API_KEY_VAR) or os.environ.get("GOOGLE_API_KEY"))
    return bool(os.environ.get(api_key_var(name)))


def make_llm(name: str | None = None, model: str | None = None) -> LLM:
    """The configured provider, ready to ask."""
    name = name or provider()
    model = model or os.environ.get(MODEL_VAR) or None
    if name == "gemini":
        return Gemini(model=model)
    if name == "claude":
        from hearing_to_seeing.knowledge.claude import Claude

        return Claude(model=model)
    raise KnowledgeError(f"{PROVIDER_VAR} 값이 잘못되었습니다 (gemini | claude): {name!r}")


# --- Gemini ------------------------------------------------------------------

_REFUSAL_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


class Gemini:
    """A thin, schema-first client. Pass `client` to substitute a fake in tests."""

    def __init__(self, model: str | None = None, client: Any = None):
        self.model = model or GEMINI_DEFAULT_MODEL
        self._client = client

    @staticmethod
    def configured() -> bool:
        return configured("gemini")

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.configured():
                raise KnowledgeError(
                    f"{GEMINI_API_KEY_VAR} 가 설정되지 않았습니다. .env 에 API 키를 넣거나 "
                    "--lookup 없이 실행하세요. (https://aistudio.google.com/apikey)"
                )
            # Imported lazily so the tests and the no-lookup path never pay for it.
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                http_options=types.HttpOptions(
                    timeout=REQUEST_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(
                        attempts=RETRY_ATTEMPTS,
                        initial_delay=RETRY_INITIAL_DELAY,
                        max_delay=RETRY_MAX_DELAY,
                        http_status_codes=RETRY_STATUS_CODES,
                    ),
                ),
            )
        return self._client

    # --- public --------------------------------------------------------------

    def structured(self, system: str, prompt: str, schema: dict) -> dict:
        """Asks `prompt` and returns an answer that satisfies `schema`."""
        from google.genai import types

        response = self._call(
            prompt,
            types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=MAX_TOKENS,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        text = _text_of(response)
        if not text.strip():
            raise KnowledgeError("모델 응답에 텍스트가 없습니다")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # the schema constrains this; belt and braces
            raise KnowledgeError(f"모델 응답이 JSON 이 아닙니다: {exc}") from exc

    # --- calls ---------------------------------------------------------------

    def _call(self, prompt: str, config: Any) -> Any:
        """One request: the preferred model, then the fallback, then one wait.

        The preferred model is given up on at the first 429 — a quota does
        not clear in the time a run can afford to wait — and the fallback is
        tried at once. Only when the fallback is over quota too is there a
        pause, once, for a per-minute limit to reset.
        """
        try:
            return self._generate(self.model, prompt, config)
        except _Unavailable as exc:
            fallback = GEMINI_FALLBACK_MODEL
            if not fallback or fallback == self.model:
                raise KnowledgeError(str(exc)) from exc
            print(f"note: {self.model} {exc.reason}; {fallback} 로 대신 요청합니다", file=sys.stderr)
            try:
                return self._generate(fallback, prompt, config)
            except _Unavailable as again:
                if again.code != 429:
                    raise KnowledgeError(str(again)) from again
                print(
                    f"note: {fallback} 도 한도에 걸려 {QUOTA_WAIT_SECONDS:.0f}초 후 한 번 더 시도합니다",
                    file=sys.stderr,
                )
                time.sleep(QUOTA_WAIT_SECONDS)
                try:
                    return self._generate(fallback, prompt, config)
                except _Unavailable as last:
                    raise KnowledgeError(str(last)) from last

    def _generate(self, model: str, prompt: str, config: Any) -> Any:
        from google.genai import errors

        try:
            response = self.client.models.generate_content(
                model=model, contents=prompt, config=config,
            )
        except errors.ClientError as exc:
            code = getattr(exc, "code", None)
            if code in (401, 403):
                raise KnowledgeError(f"{GEMINI_API_KEY_VAR} 가 올바르지 않거나 권한이 없습니다") from exc
            if code == 404:
                raise KnowledgeError(
                    f"모델을 찾을 수 없습니다: {model} ({MODEL_VAR} 로 다른 모델을 지정하세요)"
                ) from exc
            if code == 429:
                raise _Unavailable(
                    429,
                    "은(는) 무료 티어 요청 한도에 걸렸습니다",
                    f"Gemini 무료 티어 요청 한도에 걸렸습니다 ({model}). "
                    "잠시 후 다시 시도하거나 https://ai.dev/rate-limit 에서 한도를 확인하세요",
                ) from exc
            raise KnowledgeError(f"Gemini API 오류 ({code}): {exc.message}") from exc
        except errors.ServerError as exc:
            code = getattr(exc, "code", None)
            if code == 503:
                raise _Unavailable(
                    503,
                    "이(가) 수요 폭주로 응답하지 않습니다",
                    f"Gemini 서버가 수요 폭주로 응답하지 않습니다 ({model}, 재시도 후에도 실패)",
                ) from exc
            raise KnowledgeError(f"Gemini 서버 오류 ({code}): {exc.message}") from exc
        except errors.APIError as exc:
            raise KnowledgeError(f"Gemini API 오류: {exc.message}") from exc

        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise KnowledgeError(f"모델이 요청을 거부했습니다: {feedback.block_reason}")

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise KnowledgeError("모델 응답에 후보가 없습니다")
        reason = _name(getattr(candidates[0], "finish_reason", None))
        if reason == "MAX_TOKENS":
            raise KnowledgeError("모델 응답이 길이 제한에서 잘렸습니다")
        if reason in _REFUSAL_REASONS:
            raise KnowledgeError(f"모델이 응답을 거부했습니다: {reason}")
        return response


class _Unavailable(RuntimeError):
    """The model could not take the request just now; another one might."""

    def __init__(self, code: int, reason: str, message: str):
        super().__init__(message)
        self.code = code
        self.reason = reason


def _name(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def _text_of(response: Any) -> str:
    # `.text` joins the text parts; it is None when the candidate has none.
    try:
        return response.text or ""
    except (AttributeError, ValueError):
        return ""
