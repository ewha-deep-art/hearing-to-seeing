"""The one place the pipeline talks to a language model (Gemini).

Every question is the same shape — a prompt in, one JSON document out — so the
whole interface is `ask()`: the response is constrained to a JSON schema, so
parsing cannot fail. Nothing runs without `GEMINI_API_KEY`; `configured()`
lets the caller skip the lookup up front instead of failing mid-run.
"""

import json
import os
import sys
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

API_KEY_VAR = "GEMINI_API_KEY"
# ~30 s per call on the free tier. H2S_LLM_MODEL overrides it.
MODEL = os.environ.get("H2S_LLM_MODEL", "gemini-3.6-flash")
# Where a request goes when the preferred model is over quota (429) or
# overloaded (503): the lightest Flash, ~3 s per call.
FALLBACK_MODEL = "gemini-3.5-flash-lite"

MAX_TOKENS = 16000
REQUEST_TIMEOUT_MS = 180_000
# Server trouble (5xx) usually clears in seconds, so the SDK retries it
# briefly. A 429 is a quota, not worth waiting on: it goes to the fallback.
RETRY_ATTEMPTS = 4
RETRY_STATUS_CODES = [500, 502, 503, 504]


def configured() -> bool:
    return bool(os.environ.get(API_KEY_VAR))


@lru_cache(maxsize=1)
def _client():
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


def ask(system: str, prompt: str, schema: dict) -> dict:
    """Asks `prompt` and returns an answer that satisfies `schema`.

    Tries `MODEL`, then `FALLBACK_MODEL` when the first is over quota or
    overloaded. Any other failure (bad key, blocked prompt, …) raises.
    """
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=MAX_TOKENS,
        response_mime_type="application/json",
        response_json_schema=schema,
    )
    models = list(dict.fromkeys([MODEL, FALLBACK_MODEL]))
    for i, model in enumerate(models):
        try:
            response = _client().models.generate_content(
                model=model, contents=prompt, config=config,
            )
        except errors.APIError as exc:
            if exc.code not in (429, 503) or i + 1 == len(models):
                raise
            print(f"… {model} 응답 없음 ({exc.code}), {models[i + 1]} 로 재요청", file=sys.stderr)
            continue
        return json.loads(response.text or "")
    raise AssertionError("unreachable")
