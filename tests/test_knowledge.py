import json
from types import SimpleNamespace

import pytest

from hearing_to_seeing.knowledge import lookup, transcript_excerpt
from hearing_to_seeing.knowledge.characters import retrieve_characters
from hearing_to_seeing.knowledge.llm import Gemini, KnowledgeError, make_llm, configured
from hearing_to_seeing.knowledge.retrieve import Document
from hearing_to_seeing.knowledge.title import WorkQuery, clean_title, parse_title
from hearing_to_seeing.schema import Character, MediaInfo, Transcript, WordEntry, WorkInfo


class FakeLLM:
    """Answers `structured()` from a queue and records what it was asked."""

    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.calls: list[dict] = []

    def structured(self, system, prompt, schema):
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.answers.pop(0)


def _transcript(title="[4K] 기생충 명장면 | 아들아 너는 계획이 다 있구나 #shorts") -> Transcript:
    return Transcript(
        words=[
            WordEntry(text="아들아", start=0.0, end=0.5, speaker="SPEAKER_00"),
            WordEntry(text="계획이", start=0.5, end=1.0, speaker="SPEAKER_00"),
            WordEntry(text="네", start=1.2, end=1.4, speaker="SPEAKER_01"),
        ],
        media=MediaInfo(title=title, source_url="https://youtu.be/abc"),
    )


# --- clean_title -------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("[4K] 기생충 명장면 모음 | 아들아 너는 계획이 다 있구나 #shorts", "기생충 아들아 너는 계획이 다 있구나"),
    ("[왕과 사는 남자] 공식 예고편", "왕과 사는 남자"),   # the work is the bracketed part
    ("[4K] 기생충 (ENG SUB)", "기생충"),
    ("기생충 (2019) Official Trailer HD", "기생충"),
    ("   기생충   ", "기생충"),
])
def test_clean_title_strips_furniture(raw, expected):
    assert clean_title(raw) == expected


def test_clean_title_never_returns_empty():
    # Everything stripped: hand the raw title back rather than nothing.
    assert clean_title("[4K] #shorts") == "[4K] #shorts"


# --- parse_title -------------------------------------------------------------

def test_parse_title_without_a_title_is_none():
    assert parse_title(MediaInfo(source_url="https://youtu.be/abc")) is None


def test_parse_title_without_a_model_falls_back_to_rules():
    query = parse_title(MediaInfo(title="[4K] 기생충 명장면 #shorts"))
    assert query == WorkQuery(title="기생충", source="rules")


def test_parse_title_splits_work_from_scene_hint():
    llm = FakeLLM({
        "title": "기생충", "year": 2019,
        "scene_hint": "아들아 너는 계획이 다 있구나", "confidence": 0.95,
    })
    query = parse_title(_transcript().media, llm)

    assert query.title == "기생충"
    assert query.year == 2019
    assert query.scene_hint == "아들아 너는 계획이 다 있구나"
    assert query.source == "llm"
    # The URL is offered as context, even though it is never fetched.
    assert "youtu.be/abc" in llm.calls[0]["prompt"]


def test_parse_title_distrusts_a_low_confidence_answer():
    llm = FakeLLM({"title": "엉뚱한 영화", "year": None, "scene_hint": None, "confidence": 0.2})
    query = parse_title(MediaInfo(title="[4K] 기생충 #shorts"), llm)
    assert query.title == "기생충"
    assert query.source == "rules"


def test_parse_title_falls_back_when_the_model_cannot_name_a_work():
    llm = FakeLLM({"title": None, "year": None, "scene_hint": None, "confidence": 0.9})
    assert parse_title(MediaInfo(title="브이로그 #3"), llm).source == "rules"


# --- retrieve_characters -----------------------------------------------------

def _character_answer(**overrides) -> dict:
    base = {
        "name": "기택", "aliases": ["아버지", "기사님"], "actor": "송강호",
        "importance": "main", "role": "가장", "personality": "무계획",
        "relationships": "충숙의 남편", "appearance": "회색 셔츠", "in_scene": True,
    }
    return {**base, **overrides}


NAMU = Document(source="나무위키", title="기생충(영화)", url="https://namu.wiki/w/기생충(영화)", text="## 등장인물\n기택 (송강호) …")
WIKI = Document(source="위키백과", title="기생충 (영화)", url="https://ko.wikipedia.org/wiki/기생충_(영화)", text="## 줄거리\n반지하 …")


def test_retrieve_characters_reads_the_documents_and_records_them_as_sources():
    llm = FakeLLM({
        "title": "기생충", "year": 2019, "summary": "반지하 가족", "scene": "기우가 과외를 나가는 장면",
        "confidence": 0.9, "characters": [_character_answer()],
    })
    query = WorkQuery(title="기생충", scene_hint="계획", source="llm")
    work = retrieve_characters(query, llm, [NAMU, WIKI], excerpt="SPEAKER_00: 아들아")

    prompt = llm.calls[0]["prompt"]
    assert "참고 문서 (2건)" in prompt
    assert '<document source="나무위키"' in prompt and "기택 (송강호)" in prompt
    assert "아들아" in prompt
    assert work.title == "기생충"
    assert work.scene_hint == "계획"
    assert work.confidence == 0.9
    # What the model was shown, in the order it was shown — not what it claims.
    assert work.sources == [NAMU.url, WIKI.url]
    character = work.characters[0]
    assert character.name == "기택"
    assert character.aliases == ["아버지", "기사님"]
    assert character.appearance == "회색 셔츠"
    # The colour step has not run yet.
    assert character.hue is None and character.hue_basis is None


def test_retrieve_characters_without_documents_asks_from_memory_and_says_so():
    llm = FakeLLM({"title": None, "year": None, "summary": None, "scene": None, "confidence": 0.3, "characters": []})
    work = retrieve_characters(WorkQuery(title="기생충"), llm)
    assert "검색 실패" in llm.calls[0]["prompt"]
    assert work.sources == []
    assert work.confidence == 0.3


def test_retrieve_characters_puts_scene_characters_first_then_leads():
    llm = FakeLLM({
        "title": "기생충", "year": None, "summary": None, "scene": None,
        "confidence": 0.8,
        "characters": [
            _character_answer(name="박사장", importance="main", in_scene=False),
            _character_answer(name="문광", importance="supporting", in_scene=True),
            _character_answer(name="근세", importance="supporting", in_scene=False),
            _character_answer(name="기택", importance="main", in_scene=True),
        ],
    })
    work = retrieve_characters(WorkQuery(title="기생충"), llm)
    assert [c.name for c in work.characters] == ["기택", "문광", "박사장", "근세"]


def test_retrieve_characters_drops_nameless_entries_and_falls_back_to_the_query():
    llm = FakeLLM({
        "title": None, "year": None, "summary": None, "scene": None,
        "confidence": 0.5,
        "characters": [_character_answer(), _character_answer(name="  ")],
    })
    work = retrieve_characters(WorkQuery(title="기생충", year=2019), llm)
    assert [c.name for c in work.characters] == ["기택"]
    # The query stands in for anything the answer left blank.
    assert work.title == "기생충"
    assert work.year == 2019


# --- lookup ------------------------------------------------------------------

def test_transcript_excerpt_groups_words_by_speaker_turn():
    assert transcript_excerpt(_transcript()) == "SPEAKER_00: 아들아 계획이\nSPEAKER_01: 네"


def test_lookup_runs_title_retrieval_then_characters_and_stores_the_work(monkeypatch):
    import hearing_to_seeing.knowledge as knowledge

    fetched: list[str] = []
    monkeypatch.setattr(knowledge, "retrieve", lambda title: fetched.append(title) or [NAMU])
    llm = FakeLLM(
        {"title": "기생충", "year": 2019, "scene_hint": "계획", "confidence": 0.9},
        {
            "title": "기생충", "year": 2019, "summary": None, "scene": None,
            "confidence": 0.9, "characters": [_character_answer()],
        },
    )
    transcript = _transcript()
    work = lookup(transcript, llm)

    assert fetched == ["기생충"]                # the parsed work title, not the clip title
    assert transcript.work is work
    assert work.characters[0].name == "기택"
    assert work.sources == [NAMU.url]
    assert len(llm.calls) == 2


def test_lookup_reuses_a_stored_work_without_asking_again():
    llm = FakeLLM()
    transcript = _transcript()
    transcript.work = WorkInfo(title="기생충", characters=[Character(name="기택")])

    assert lookup(transcript, llm) is transcript.work
    assert llm.calls == []


def test_lookup_without_a_title_returns_none_and_asks_nothing():
    llm = FakeLLM()
    assert lookup(_transcript(title=None), llm) is None
    assert llm.calls == []


def test_work_info_round_trips_through_json():
    work = WorkInfo(
        title="기생충", year=2019, scene_hint="계획", scene="과외", summary="요약",
        sources=["https://a"], confidence=0.9,
        characters=[Character(
            name="기택", aliases=["아버지"], actor="송강호", importance="main",
            appearance="회색 셔츠", hue=40.0, hue_basis="symbolic",
            hue_evidence="흙빛", hue_confidence=0.8,
        )],
    )
    restored = WorkInfo.from_dict(json.loads(json.dumps(work.to_dict())))
    assert restored == work


# --- Gemini wrapper ----------------------------------------------------------

def _enum(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _gemini_response(text, *, finish="STOP", urls=(), block_reason=None) -> SimpleNamespace:
    chunks = [SimpleNamespace(web=SimpleNamespace(uri=url)) for url in urls]
    candidate = SimpleNamespace(
        finish_reason=_enum(finish),
        grounding_metadata=SimpleNamespace(grounding_chunks=chunks) if chunks else None,
    )
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        prompt_feedback=SimpleNamespace(block_reason=block_reason) if block_reason else None,
    )


class FakeGenai:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def test_gemini_structured_constrains_the_answer_to_the_schema():
    client = FakeGenai(_gemini_response('{"a": 1}'))
    assert Gemini(client=client).structured("sys", "q", SCHEMA) == {"a": 1}

    request = client.requests[0]
    assert request["model"] == Gemini().model
    assert request["contents"] == "q"
    config = request["config"]
    assert config.system_instruction == "sys"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == SCHEMA
    assert config.tools is None


def test_gemini_falls_back_to_the_lighter_model_when_the_preferred_one_is_unavailable(monkeypatch, capsys):
    from google.genai import errors

    class Flaky:
        def __init__(self):
            self.models = SimpleNamespace(generate_content=self._generate)
            self.requests = []

        def _generate(self, **kwargs):
            self.requests.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.8-flash":
                raise errors.ServerError(503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}})
            return _gemini_response('{"a": 9}')

    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    client = Flaky()
    assert Gemini(model="gemini-3.8-flash", client=client).structured("s", "q", SCHEMA) == {"a": 9}
    assert client.requests == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    assert "gemini-3.5-flash-lite 로 대신" in capsys.readouterr().err


def test_gemini_gives_up_when_the_fallback_is_unavailable_too(monkeypatch):
    from google.genai import errors

    class Down:
        def __init__(self):
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, **kwargs):
            raise errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    slept: list[float] = []
    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.time.sleep", slept.append)
    with pytest.raises(KnowledgeError, match="한도"):
        Gemini(client=Down()).structured("s", "q", SCHEMA)
    assert slept == [30.0]                   # one wait for a per-minute limit, then give up


def test_gemini_waits_once_for_a_fallback_that_is_over_its_minute_quota(monkeypatch):
    from google.genai import errors

    class Bursty:
        def __init__(self):
            self.models = SimpleNamespace(generate_content=self._generate)
            self.requests = []

        def _generate(self, **kwargs):
            self.requests.append(kwargs["model"])
            if len(self.requests) <= 2:      # preferred 429, fallback 429
                raise errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
            return _gemini_response('{"a": 7}')

    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    slept: list[float] = []
    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.time.sleep", slept.append)
    client = Bursty()
    assert Gemini(model="gemini-3.8-flash", client=client).structured("s", "q", SCHEMA) == {"a": 7}
    assert client.requests == ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash-lite"]
    assert slept == [30.0]


def test_gemini_does_not_wait_on_a_fallback_server_error(monkeypatch):
    from google.genai import errors

    class Overloaded:
        def __init__(self):
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, **kwargs):
            raise errors.ServerError(503, {"error": {"message": "busy", "status": "UNAVAILABLE"}})

    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.time.sleep", lambda s: pytest.fail("slept"))
    with pytest.raises(KnowledgeError, match="수요 폭주"):
        Gemini(client=Overloaded()).structured("s", "q", SCHEMA)


def test_gemini_reports_a_blocked_prompt_or_refused_answer():
    with pytest.raises(KnowledgeError):
        Gemini(client=FakeGenai(_gemini_response("", block_reason="PROHIBITED_CONTENT"))).structured("s", "q", SCHEMA)
    with pytest.raises(KnowledgeError):
        Gemini(client=FakeGenai(_gemini_response("", finish="SAFETY"))).structured("s", "q", SCHEMA)


def test_gemini_reports_a_truncated_or_empty_answer():
    with pytest.raises(KnowledgeError):
        Gemini(client=FakeGenai(_gemini_response('{"a":', finish="MAX_TOKENS"))).structured("s", "q", SCHEMA)
    with pytest.raises(KnowledgeError):
        Gemini(client=FakeGenai(_gemini_response(None))).structured("s", "q", SCHEMA)


def test_gemini_translates_api_errors_into_one_kind(monkeypatch):
    from google.genai import errors

    class Failing:
        def __init__(self, code):
            self.models = SimpleNamespace(generate_content=self._fail)
            self.code = code

        def _fail(self, **kwargs):
            raise errors.ClientError(self.code, {"error": {"message": "nope", "status": "x"}})

    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.GEMINI_FALLBACK_MODEL", "")
    for code in (401, 404, 429, 400):
        with pytest.raises(KnowledgeError):
            Gemini(client=Failing(code)).structured("s", "q", SCHEMA)


def test_gemini_without_a_key_fails_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert Gemini.configured() is False
    with pytest.raises(KnowledgeError):
        Gemini().structured("sys", "q", SCHEMA)


def test_make_llm_defaults_to_gemini_and_honours_the_model_override(monkeypatch):
    monkeypatch.delenv("H2S_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("H2S_LLM_MODEL", "gemini-3.5-flash-lite")
    llm = make_llm()
    assert isinstance(llm, Gemini)
    assert llm.model == "gemini-3.5-flash-lite"


def test_make_llm_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("H2S_LLM_PROVIDER", "gpt")
    with pytest.raises(KnowledgeError):
        make_llm()
    with pytest.raises(KnowledgeError):
        configured()
