"""The Claude backend — the alternative to `llm.Gemini`, behind the `claude` extra.

Same contract as `Gemini` (see `llm.LLM`): a prompt and a schema in, one
conforming dict out. Retrieval happens outside the model (`retrieve.py`), so
no server tools are used. Select it with `H2S_LLM_PROVIDER=claude` and
install with `uv sync --extra claude`.
"""

import json
from typing import Any

from hearing_to_seeing.knowledge.llm import MAX_TOKENS, KnowledgeError, configured

DEFAULT_MODEL = "claude-opus-5"
API_KEY_VAR = "ANTHROPIC_API_KEY"


class Claude:
    """A thin, schema-first client. Pass `client` to substitute a fake in tests."""

    def __init__(self, model: str | None = None, client: Any = None):
        self.model = model or DEFAULT_MODEL
        self._client = client

    @staticmethod
    def configured() -> bool:
        return configured("claude")

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.configured():
                raise KnowledgeError(
                    f"{API_KEY_VAR} 가 설정되지 않았습니다. .env 에 API 키를 넣거나 "
                    "--lookup 없이 실행하세요."
                )
            try:
                import anthropic
            except ImportError as exc:
                raise KnowledgeError(
                    "anthropic 패키지가 없습니다: uv sync --extra claude"
                ) from exc

            self._client = anthropic.Anthropic()
        return self._client

    # --- public --------------------------------------------------------------

    def structured(self, system: str, prompt: str, schema: dict) -> dict:
        response = self._call(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise KnowledgeError("모델 응답에 텍스트가 없습니다")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise KnowledgeError(f"모델 응답이 JSON 이 아닙니다: {exc}") from exc

    def _call(self, **kwargs: Any) -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise KnowledgeError("anthropic 패키지가 없습니다: uv sync --extra claude") from exc

        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=MAX_TOKENS, **kwargs,
            )
        except anthropic.AuthenticationError as exc:
            raise KnowledgeError(f"{API_KEY_VAR} 가 올바르지 않습니다") from exc
        except anthropic.RateLimitError as exc:
            raise KnowledgeError("API 요청 한도에 걸렸습니다. 잠시 후 다시 시도하세요") from exc
        except anthropic.APIStatusError as exc:
            raise KnowledgeError(f"API 오류 ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise KnowledgeError("API 서버에 연결할 수 없습니다") from exc

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            reason = getattr(detail, "explanation", None) or "no explanation"
            raise KnowledgeError(f"모델이 응답을 거부했습니다: {reason}")
        if response.stop_reason == "max_tokens":
            raise KnowledgeError("모델 응답이 길이 제한에서 잘렸습니다")
        return response
