import itertools
from types import SimpleNamespace

import pytest
from google.genai import errors

from hearing_to_seeing.knowledge import llm

# conftest swaps `llm.ask` for a guard; these tests drive the real one.
ASK = llm.ask


@pytest.fixture
def keys(monkeypatch):
    """Three keys whose clients log each call; `keys.fail[(key, model)]` is the
    status that pair answers with."""
    monkeypatch.setenv(llm.API_KEY_VAR, "a, b,c")
    monkeypatch.setattr(llm, "_turns", {})
    monkeypatch.setattr(llm, "_owner", {})
    calls, fail = [], {}

    def client(key):
        def generate_content(model, contents, config):
            calls.append((key, model))
            if (key, model) in fail:
                raise errors.APIError(fail[(key, model)], {})
            return SimpleNamespace(candidates=[], text='{"ok": true}')

        files = SimpleNamespace(
            upload=lambda file: SimpleNamespace(name=f"up{key}", state=None),
            get=None,
            delete=lambda name: calls.append((key, f"delete {name}")),
        )
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content), files=files)

    monkeypatch.setattr(llm, "_client", client)
    return SimpleNamespace(calls=calls, fail=fail)


def test_requests_take_the_keys_in_turn(keys):
    for _ in range(4):
        ASK("s", "p", {}, fallback=())
    assert [key for key, _ in keys.calls] == [0, 1, 2, 0]


def test_each_model_takes_the_keys_in_turn(keys):
    for _ in range(3):
        ASK("s", "p", {}, model="a", fallback=())
        ASK("s", "p", {}, model="b", fallback=())
    assert keys.calls == [(0, "a"), (0, "b"), (1, "a"), (1, "b"), (2, "a"), (2, "b")]


def test_a_key_over_quota_hands_the_request_to_the_next(keys):
    keys.fail[(0, llm.MODEL)] = 429
    assert ASK("s", "p", {}) == {"ok": True}
    assert keys.calls == [(0, llm.MODEL), (1, llm.MODEL)]


def test_fallback_model_only_after_every_key_is_over_quota(keys):
    for key in range(3):
        keys.fail[(key, llm.MODEL)] = 429
    ASK("s", "p", {})
    assert keys.calls[-1] == (0, llm.FALLBACK_MODEL)
    assert len(keys.calls) == 4


def test_overload_goes_straight_to_the_fallback_model(keys):
    keys.fail[(0, llm.MODEL)] = 503
    ASK("s", "p", {})
    assert keys.calls == [(0, llm.MODEL), (0, llm.FALLBACK_MODEL)]


def test_every_key_and_model_over_quota_raises(keys):
    for key in range(3):
        for model in (llm.MODEL, llm.FALLBACK_MODEL):
            keys.fail[(key, model)] = 429
    with pytest.raises(errors.APIError):
        ASK("s", "p", {})


def test_an_upload_is_asked_about_and_deleted_through_its_own_key(keys):
    ASK("s", "p", {})  # MODEL's turn, key 0
    llm.discard(llm.upload("clip.mp4"))  # the uploads' turn, key 0
    media = llm.upload("clip.mp4")  # key 1
    keys.fail[(1, llm.MODEL)] = 429
    ASK("s", "p", {}, media=media)
    assert keys.calls[2:] == [(1, llm.MODEL), (1, llm.FALLBACK_MODEL), (1, "delete up1")]


def test_configured_ignores_empty_entries(monkeypatch):
    monkeypatch.setenv(llm.API_KEY_VAR, " , ")
    assert llm._keys() == []
    monkeypatch.setenv(llm.API_KEY_VAR, "a,,b")
    assert llm._keys() == ["a", "b"]
