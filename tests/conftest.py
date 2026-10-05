import pytest

from hearing_to_seeing.knowledge import llm


@pytest.fixture(autouse=True)
def no_model_calls(monkeypatch):
    """A `.env` with a real key must not make the tests call the model;
    tests that need answers install their own `ask`."""
    def ask(*_, **__):
        raise AssertionError("a test reached the real model")

    monkeypatch.setattr(llm, "configured", lambda: False)
    monkeypatch.setattr(llm, "ask", ask)
