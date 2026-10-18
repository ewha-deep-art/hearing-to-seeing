"""The one place the pipeline talks to a language model (Gemini).

Every question is the same shape — a prompt (and maybe a clip) in, one JSON
document out — so the whole interface is `ask()`: the response is constrained
to a JSON schema, so parsing cannot fail. Nothing runs without `GEMINI_API_KEY`; `configured()`
lets the caller skip the lookup up front instead of failing mid-run.
"""

import json
import math
import os
import sys
import threading
import time
from functools import lru_cache

API_KEY_VAR = "GEMINI_API_KEY"
# ~30 s per call on the free tier. H2S_LLM_MODEL overrides it.
MODEL = os.environ.get("H2S_LLM_MODEL", "gemini-3.6-flash")
# Where a request goes when the preferred model is over quota (429) or
# overloaded (503): the lightest Flash, ~3 s per call.
FALLBACK_MODEL = "gemini-3.5-flash-lite"

MAX_TOKENS = 16000
# Same question, same answer as far as Gemini allows (seed is best-effort).
TEMPERATURE = 0.0
SEED = 0
REQUEST_TIMEOUT_MS = 180_000
# Server trouble (5xx) usually clears in seconds, so the SDK retries it
# briefly. 429 (quota) and 503 (overloaded) go straight to the fallback: a
# retried 503 kept the cast lookup waiting minutes before it got there.
RETRY_ATTEMPTS = 4
RETRY_STATUS_CODES = [500, 502, 504]
# How often to check whether an uploaded video is ready to be asked about.
UPLOAD_POLL_SECONDS = 2.0


def configured() -> bool:
    return bool(os.environ.get(API_KEY_VAR))


# The cast lookup and the review upload ask for the client at once; built
# twice, the copy the cache drops closes itself (`Client.__del__`) mid-request.
_CLIENT_LOCK = threading.Lock()


def _client():
    with _CLIENT_LOCK:
        return _new_client()


@lru_cache(maxsize=1)
def _new_client():
    # Imported lazily so a run without a key never pays for the SDK.
    from google import genai
    from google.genai import types

    return genai.Client(
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(
                attempts=RETRY_ATTEMPTS,
                initial_delay=2.0,
                max_delay=20.0,
                http_status_codes=RETRY_STATUS_CODES,
            ),
        ),
    )


def upload(path: str):
    """Puts a media file where the model can read it; video takes a while to
    process (~10 s for five minutes), so a caller can start it early and hand
    the result to `ask` as `media`."""
    media = _client().files.upload(file=path)
    deadline = time.monotonic() + REQUEST_TIMEOUT_MS / 1000
    while media.state and media.state.name == "PROCESSING":
        if time.monotonic() > deadline:
            raise TimeoutError(f"{path} 처리가 끝나지 않음")
        time.sleep(UPLOAD_POLL_SECONDS)
        media = _client().files.get(name=media.name)
    if media.state and media.state.name == "FAILED":
        raise RuntimeError(f"{path} 업로드 처리 실패")
    return media


def discard(media) -> None:
    """Deletes an `upload` no question was asked about."""
    from google.genai import errors

    try:
        _client().files.delete(name=media.name)
    except errors.APIError:
        pass  # uploads expire on their own after two days


def ask(
    system: str,
    prompt: str,
    schema: dict,
    *,
    model: str | None = None,
    media_path: str | None = None,
    media=None,
    fallback: tuple[str, ...] = (FALLBACK_MODEL,),
    temperature: float = TEMPERATURE,
    seed: int = SEED,
    max_tokens: int = MAX_TOKENS,
    thinking_level: str | None = None,
    timeout_ms: int | None = None,
) -> dict:
    """Asks `prompt` and returns an answer that satisfies `schema`.

    Tries `model` (default `MODEL`), then each `fallback` model on 429/503.
    `media_path` is uploaded, sent at low resolution and deleted afterwards;
    `media` is a file `upload` already put up. With `timeout_ms`, each model
    gets one try of at most that long. Other failures raise.
    """
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_json_schema=schema,
        temperature=temperature,
        seed=seed,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW if media or media_path else None,
        thinking_config=(
            types.ThinkingConfig(thinking_level=types.ThinkingLevel(thinking_level.upper()))
            if thinking_level else None
        ),
        # Sent per request: the SDK keeps the first request's X-Server-Timeout
        # in the client's shared headers (evaluation/README.md 5.1).
        http_options=(
            types.HttpOptions(
                timeout=timeout_ms,
                headers={"X-Server-Timeout": str(math.ceil(timeout_ms / 1000))},
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            if timeout_ms else None
        ),
    )
    media = media or (upload(media_path) if media_path else None)
    contents = [media, prompt] if media else prompt
    models = list(dict.fromkeys([model or MODEL, *fallback]))
    try:
        for i, name in enumerate(models):
            try:
                response = _client().models.generate_content(
                    model=name, contents=contents, config=config,
                )
            except errors.APIError as exc:
                if exc.code not in (429, 503) or i + 1 == len(models):
                    raise
                print(f"… {name} 응답 없음 ({exc.code}), {models[i + 1]} 로 재요청", file=sys.stderr)
                continue
            finish = response.candidates[0].finish_reason if response.candidates else None
            if finish == types.FinishReason.MAX_TOKENS:
                # Thinking counts against the limit, so this can strike early.
                raise RuntimeError(f"{name} 응답이 {max_tokens} 토큰에서 잘림")
            return json.loads(response.text or "")
    finally:
        if media:
            discard(media)
    raise AssertionError("unreachable")
