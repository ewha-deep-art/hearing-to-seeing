"""The optional Claude backend. Skipped when the `claude` extra is not installed."""

from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from hearing_to_seeing.knowledge.claude import Claude  # noqa: E402
from hearing_to_seeing.knowledge.llm import KnowledgeError  # noqa: E402


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _response(*blocks, stop_reason="end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, stop_details=None)


class FakeAnthropic:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # Copy the history: the caller keeps appending to its own list.
        self.requests.append({**kwargs, "messages": list(kwargs.get("messages", []))})
        return self.responses.pop(0)


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def test_structured_constrains_the_answer_to_the_schema():
    client = FakeAnthropic(_response(_text('{"a": 1}')))
    assert Claude(client=client).structured("sys", "q", SCHEMA) == {"a": 1}

    request = client.requests[0]
    assert request["output_config"]["format"]["schema"] is SCHEMA
    assert request["system"] == "sys"
    assert "tools" not in request


def test_structured_reports_a_refusal_instead_of_returning_garbage():
    client = FakeAnthropic(_response(stop_reason="refusal"))
    with pytest.raises(KnowledgeError):
        Claude(client=client).structured("sys", "q", SCHEMA)


def test_structured_reports_a_truncated_answer():
    client = FakeAnthropic(_response(_text('{"a":'), stop_reason="max_tokens"))
    with pytest.raises(KnowledgeError):
        Claude(client=client).structured("sys", "q", SCHEMA)


def test_client_without_a_key_fails_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert Claude.configured() is False
    with pytest.raises(KnowledgeError):
        Claude().structured("sys", "q", SCHEMA)


