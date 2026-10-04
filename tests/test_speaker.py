import pytest

from hearing_to_seeing.design import speaker
from hearing_to_seeing.design.speaker import (
    MAIN_CHROMA_RATIO,
    MIN_HUE_GAP,
    _anchors,
    _rgb_to_oklch,
    _turns,
    assign_hues,
    assign_speaker_colors,
    hex_hue,
    hue_distance,
    hue_to_ass,
    lookup_characters,
    speaker_colors,
)
from hearing_to_seeing.schema import Transcript, WordEntry


def _transcript(*turns: tuple[str, str]) -> Transcript:
    words, t = [], 0.0
    for label, text in turns:
        for token in text.split():
            words.append(WordEntry(text=token, start=t, end=t + 0.3, speaker=label))
            t += 0.4
    return Transcript(words=words)


def _oklch(colour: str) -> tuple[float, float, float]:
    digits = colour.removeprefix("&H00")
    b, g, r = (int(digits[i:i + 2], 16) for i in (0, 2, 4))
    return _rgb_to_oklch(r, g, b)


def _character(name, hue, importance="main", aliases=()):
    return {"name": name, "aliases": list(aliases), "importance": importance, "role": None,
            "personality": None, "relationships": None, "color_basis": "symbolic",
            "color_reason": "근거", "color_confidence": 0.9, "hue": hue}


class FakeLLM:
    """Stands in for `llm.ask`, answering by which schema was asked for."""

    def __init__(self, **answers):
        self.answers = answers
        self.asked: list[str] = []

    def __call__(self, system, prompt, schema):
        key = next(iter(schema["properties"]))  # title | characters | speakers
        self.asked.append(key)
        answer = self.answers[key]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def fake_llm(monkeypatch):
    def install(**answers):
        fake = FakeLLM(**answers)
        monkeypatch.setattr(speaker.llm, "ask", fake)
        monkeypatch.setattr(speaker.llm, "configured", lambda: True)
        return fake
    return install


# --- colour model --------------------------------------------------------------

def test_hue_to_ass_keeps_the_hue_and_the_lightness():
    for hue in (0, 90, 200, 300):
        L, C, h = _oklch(hue_to_ass(hue))
        assert abs(L - speaker.LIGHTNESS) < 0.01
        assert hue_distance(h, hue) < 2
        assert C > 0.1


def test_hex_hue_ignores_greys():
    assert hex_hue("#808080") is None
    assert hex_hue("#1B2233") is None  # a navy suit
    assert hue_distance(hex_hue("#C0392B"), 29) < 5
    assert hex_hue("not a colour") is None


# --- hue assignment ------------------------------------------------------------

def test_assign_hues_spreads_evenly_without_preferences():
    hues = assign_hues(["SPEAKER_01", "SPEAKER_00", "SPEAKER_02"])
    assert hues == {"SPEAKER_00": 0.0, "SPEAKER_01": 120.0, "SPEAKER_02": 240.0}


def test_assign_hues_honours_preferences_in_order_and_keeps_the_gap():
    hues = assign_hues(["A", "B", "C"], {"A": 30.0, "B": 40.0})
    assert hues["A"] == 30.0                               # first in order gets its wish
    assert hue_distance(hues["B"], 30.0) == MIN_HUE_GAP    # pushed flush against A
    assert all(hue_distance(hues[x], hues[y]) >= MIN_HUE_GAP for x, y in [("A", "C"), ("B", "C")])


def test_assign_hues_places_everyone_when_the_circle_is_crowded():
    labels = [f"S{i}" for i in range(12)]
    hues = assign_hues(labels, {label: 0.0 for label in labels})
    assert len(set(round(h, 6) for h in hues.values())) == 12


# --- lookup --------------------------------------------------------------------

def test_lookup_characters_adds_a_hue_per_character(fake_llm, monkeypatch):
    monkeypatch.setattr(speaker, "retrieve", lambda title: f"<document>{title}</document>")
    fake = fake_llm(
        title={"title": "기생충"},
        characters={"characters": [
            {**_character("기택", None), "color": "#C0392B"},
            {**_character("충숙", None), "color": "#222222"},
        ]},
    )
    cast = lookup_characters("[4K] 기생충 명장면 #shorts")
    assert fake.asked == ["title", "characters"]
    assert hue_distance(cast[0]["hue"], 29) < 5
    assert cast[1]["hue"] is None  # grey: no preference


def test_lookup_characters_needs_a_title_and_a_key(monkeypatch):
    monkeypatch.setattr(speaker.llm, "configured", lambda: False)
    assert lookup_characters("기생충") is None
    monkeypatch.setattr(speaker.llm, "configured", lambda: True)
    assert lookup_characters(None) is None


def test_lookup_characters_stops_when_the_work_is_unknown(fake_llm):
    fake = fake_llm(title={"title": None})
    assert lookup_characters("브이로그") is None
    assert fake.asked == ["title"]


def test_lookup_characters_swallows_failures(fake_llm):
    fake_llm(title=RuntimeError("503"))
    assert lookup_characters("기생충") is None


# --- speaker → character -------------------------------------------------------

def test_anchors_count_who_named_whom_and_who_answered():
    turns = _turns(_transcript(("SPEAKER_00", "기택아 밥 먹었니"), ("SPEAKER_01", "네 먹었어요")))
    text = _anchors(turns, [_character("기택", 30.0)])
    assert "SPEAKER_00 이(가) '기택' 을(를) 1번 부름" in text
    assert "직후 SPEAKER_01 이(가) 1번 대답함" in text


def test_matched_speakers_take_their_characters_hue(fake_llm):
    transcript = _transcript(
        ("SPEAKER_00", "기택아 밥 먹었니 오늘"), ("SPEAKER_01", "네 어머니 먹었어요"),
        ("SPEAKER_02", "저도 먹었어요 방금"),
    )
    fake_llm(speakers={"speakers": [
        {"label": "SPEAKER_00", "character": "충숙", "confidence": 0.9, "reasoning": "호명"},
        {"label": "SPEAKER_01", "character": "기택", "confidence": 0.9, "reasoning": "응답"},
        {"label": "SPEAKER_02", "character": "기우", "confidence": 0.5, "reasoning": "약함"},
    ]})
    cast = [_character("기택", 30.0), _character("충숙", 250.0), _character("기우", 150.0)]
    profiles = assign_speaker_colors(transcript, cast)

    assert profiles["SPEAKER_00"]["name"] == "충숙"
    assert profiles["SPEAKER_01"]["name"] == "기택"
    assert hue_distance(_oklch(profiles["SPEAKER_01"]["color"])[2], 30.0) < 3
    assert hue_distance(_oklch(profiles["SPEAKER_00"]["color"])[2], 250.0) < 3
    # Below MIN_CONFIDENCE: no name, and muted so it reads as unidentified.
    assert profiles["SPEAKER_02"]["name"] is None
    _, chroma, hue = _oklch(profiles["SPEAKER_02"]["color"])
    assert chroma < speaker._max_chroma(speaker.LIGHTNESS, round(hue)) / 2


def test_failed_inference_falls_back_to_even_vivid_hues(fake_llm):
    transcript = _transcript(("SPEAKER_00", "하나 둘 셋"), ("SPEAKER_01", "넷 다섯 여섯"))
    fake_llm(speakers=RuntimeError("quota"))
    profiles = assign_speaker_colors(transcript, [_character("기택", 30.0)])
    assert profiles == assign_speaker_colors(transcript)
    assert all(p["name"] is None for p in profiles.values())


def test_speaker_colors_prefers_stored_profiles():
    transcript = _transcript(("SPEAKER_00", "a b c"))
    assert speaker_colors(transcript) == {"SPEAKER_00": hue_to_ass(0.0, MAIN_CHROMA_RATIO)}
    transcript.speaker_profiles = {"SPEAKER_00": {"color": "&H00123456"}}
    assert speaker_colors(transcript) == {"SPEAKER_00": "&H00123456"}
