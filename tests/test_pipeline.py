"""The tail of the pipeline, with the media decode and the model both faked.

What is checked is the wiring: which strategy wins for which speaker, what
ends up in `character_map`, and that the stored blocks are replayed rather
than recomputed.
"""

import json

import numpy as np
import pytest

from hearing_to_seeing import pipeline
from hearing_to_seeing.design.identity import manual_mapping
from hearing_to_seeing.schema import Character, MediaInfo, Transcript, WordEntry, WorkInfo


class FakeLLM:
    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.calls: list[dict] = []

    def structured(self, system, prompt, schema):
        self.calls.append({"prompt": prompt})
        return self.answers.pop(0)


@pytest.fixture(autouse=True)
def silent_audio(monkeypatch):
    monkeypatch.setattr(pipeline, "decode_audio", lambda path: (16000, np.zeros(16000 * 4, dtype=np.float32)))


@pytest.fixture(autouse=True)
def offline_retrieval(monkeypatch):
    import hearing_to_seeing.knowledge as knowledge
    from hearing_to_seeing.knowledge.retrieve import Document

    monkeypatch.setattr(knowledge, "retrieve", lambda title: [
        Document(source="나무위키", title=title, url=f"https://namu.wiki/w/{title}", text="## 등장인물\n기택, 기우"),
    ])


def _transcript() -> Transcript:
    words = []
    for i, (speaker, text) in enumerate([
        ("SPEAKER_00", "아버지 이거 좀 보세요"),
        ("SPEAKER_01", "뭔데 그래 인마 어서"),
    ]):
        for j, token in enumerate(text.split()):
            t = i * 2.5 + j * 0.5
            words.append(WordEntry(text=token, start=t, end=t + 0.4, speaker=speaker))
    return Transcript(words=words, media=MediaInfo(title="기생충 명장면"))


def _decided_work() -> WorkInfo:
    return WorkInfo(title="기생충", characters=[
        Character(name="기택", aliases=["아버지"], importance="main", hue=40.0, hue_basis="costume", hue_confidence=0.9),
        Character(name="기우", aliases=["아들"], importance="main", hue=220.0, hue_basis="symbolic", hue_confidence=0.8),
    ])


def _run(transcript, tmp_path, **kwargs) -> Transcript:
    src = tmp_path / "in.json"
    src.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False), encoding="utf-8")
    return pipeline.run_from_json(
        str(tmp_path / "clip.mp4"), str(src), str(tmp_path / "out.ass"),
        str(tmp_path / "out.json"), **kwargs,
    )


def test_lookup_hues_and_inference_run_in_order_and_fill_character_map(tmp_path):
    llm = FakeLLM(
        {"title": "기생충", "year": 2019, "scene_hint": None, "confidence": 0.9},          # title
        {"title": "기생충", "year": 2019, "summary": None, "scene": None,   # cast
         "confidence": 0.9, "characters": [
             {"name": "기택", "aliases": ["아버지"], "actor": None, "importance": "main", "role": None,
              "personality": None, "relationships": None, "appearance": "갈색 점퍼", "in_scene": True},
             {"name": "기우", "aliases": ["아들"], "actor": None, "importance": "main", "role": None,
              "personality": None, "relationships": None, "appearance": None, "in_scene": True},
         ]},
        {"characters": [                                                                    # hues
            {"name": "기택", "costume": {"hex": "#8B5A2B", "evidence": "갈색 점퍼"}, "symbolic": {"hex": None, "evidence": None}, "confidence": 0.9},
            {"name": "기우", "costume": {"hex": None, "evidence": None}, "symbolic": {"hex": "#1F4E9C", "evidence": "냉정"}, "confidence": 0.7},
        ]},
        {"speakers": [                                                                      # mapping
            {"label": "SPEAKER_00", "character": "기우", "confidence": 0.85, "reasoning": "아버지를 부름"},
            {"label": "SPEAKER_01", "character": "기택", "confidence": 0.9, "reasoning": "대답"},
        ]},
    )
    result = _run(_transcript(), tmp_path, llm=llm)

    assert len(llm.calls) == 4
    assert result.work.sources == ["https://namu.wiki/w/기생충"]
    assert result.character_map == {"SPEAKER_00": "기우", "SPEAKER_01": "기택"}
    assert result.speaker_profiles["SPEAKER_01"].source == "text"
    assert "색상 근거(costume): 갈색 점퍼" in result.speaker_profiles["SPEAKER_01"].note
    assert result.work.characters[0].hue_basis == "costume"

    saved = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert saved["character_map"] == {"SPEAKER_00": "기우", "SPEAKER_01": "기택"}
    assert saved["work"]["characters"][0]["hue"] == pytest.approx(result.work.characters[0].hue)


def test_manual_mapping_outranks_inference_and_inference_fills_the_rest(tmp_path):
    transcript = _transcript()
    transcript.work = _decided_work()          # lookup + hues already stored
    llm = FakeLLM({"speakers": [
        {"label": "SPEAKER_00", "character": "기택", "confidence": 0.9, "reasoning": "틀린 답"},
        {"label": "SPEAKER_01", "character": "기택", "confidence": 0.9, "reasoning": "맞는 답"},
    ]})
    result = _run(transcript, tmp_path, speaker_strategy=manual_mapping({"SPEAKER_00": "기우"}), llm=llm)

    assert len(llm.calls) == 1                 # only the mapping call: work was reused
    assert result.speaker_profiles["SPEAKER_00"].name == "기우"
    assert result.speaker_profiles["SPEAKER_00"].source == "manual"
    assert result.speaker_profiles["SPEAKER_01"].name == "기택"
    assert result.speaker_profiles["SPEAKER_01"].source == "text"
    assert result.character_map == {"SPEAKER_00": "기우", "SPEAKER_01": "기택"}


def test_a_stored_character_map_is_replayed_without_asking_the_model(tmp_path):
    transcript = _transcript()
    transcript.work = _decided_work()
    transcript.character_map = {"SPEAKER_00": "기우", "SPEAKER_01": "기택"}
    llm = FakeLLM()

    result = _run(transcript, tmp_path, llm=llm)

    assert llm.calls == []
    assert result.speaker_profiles["SPEAKER_00"].source == "stored"
    assert result.character_map == {"SPEAKER_00": "기우", "SPEAKER_01": "기택"}


def test_a_stored_character_map_works_without_a_model_at_all(tmp_path):
    transcript = _transcript()
    transcript.work = _decided_work()
    transcript.character_map = {"SPEAKER_01": "기택"}

    result = _run(transcript, tmp_path)         # no llm, no --lookup

    assert result.speaker_profiles["SPEAKER_01"].name == "기택"
    assert "색상 근거(costume)" in result.speaker_profiles["SPEAKER_01"].note   # hue still attached
    assert result.speaker_profiles["SPEAKER_00"].name is None
    assert result.character_map == {"SPEAKER_01": "기택"}


def test_fresh_discards_stored_work_and_mapping(tmp_path):
    transcript = _transcript()
    transcript.work = _decided_work()
    transcript.character_map = {"SPEAKER_00": "기우", "SPEAKER_01": "기택"}
    llm = FakeLLM(
        {"title": "기생충", "year": None, "scene_hint": None, "confidence": 0.9},
        {"title": "기생충", "year": None, "summary": None, "scene": None, "confidence": 0.9,
         "characters": [{"name": "기택", "aliases": [], "actor": None, "importance": "main", "role": None,
                         "personality": None, "relationships": None, "appearance": None, "in_scene": True}]},
        {"characters": [{"name": "기택", "costume": {"hex": None, "evidence": None},
                         "symbolic": {"hex": "#C0392B", "evidence": "x"}, "confidence": 0.8}]},
        {"speakers": [{"label": "SPEAKER_01", "character": "기택", "confidence": 0.9, "reasoning": "y"},
                      {"label": "SPEAKER_00", "character": None, "confidence": 0.1, "reasoning": "z"}]},
    )
    result = _run(transcript, tmp_path, llm=llm, fresh=True)

    assert len(llm.calls) == 4
    assert result.character_map == {"SPEAKER_01": "기택"}
    assert result.speaker_profiles["SPEAKER_01"].source == "text"


def test_a_rejected_inference_leaves_the_label_out_of_character_map(tmp_path):
    transcript = _transcript()
    transcript.work = _decided_work()
    llm = FakeLLM({"speakers": [
        {"label": "SPEAKER_00", "character": "기우", "confidence": 0.3, "reasoning": "애매"},
        {"label": "SPEAKER_01", "character": "기택", "confidence": 0.9, "reasoning": "확실"},
    ]})
    result = _run(transcript, tmp_path, llm=llm)
    assert result.character_map == {"SPEAKER_01": "기택"}
    assert "discarded 기우" in result.speaker_profiles["SPEAKER_00"].note


def test_from_json_reads_the_segment_export(tmp_path):
    src = tmp_path / "segments.json"
    src.write_text(json.dumps({
        "video_title": "[왕과 사는 남자] 공식 예고편",
        "video_url": "https://www.youtube.com/watch?v=abc",
        "segments": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "청룡포라고 들어보셨습니까"},
            {"speaker": "SPEAKER_02", "start": 2.0, "end": 3.0, "text": "누가 오든 말이다"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    result = pipeline.run_from_json(str(tmp_path / "clip.mp4"), str(src), str(tmp_path / "out.ass"))

    assert result.media.title == "[왕과 사는 남자] 공식 예고편"
    assert result.speakers() == ["SPEAKER_00", "SPEAKER_02"]
    assert (tmp_path / "out.ass").exists()


# --- run_lookup: identification with no media --------------------------------

def _segments_file(tmp_path):
    src = tmp_path / "segments.json"
    src.write_text(json.dumps({
        "video_title": "[기생충] 명장면 | 아들아 너는 계획이 다 있구나",
        "video_url": "https://www.youtube.com/watch?v=abc",
        "segments": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "아버지 이거 좀 보세요"},
            {"speaker": "SPEAKER_01", "start": 2.5, "end": 4.0, "text": "뭔데 그래 인마 어서"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    return src


def _full_lookup_answers() -> list[dict]:
    return [
        {"title": "기생충", "year": 2019, "scene_hint": "계획", "confidence": 0.9},
        {"title": "기생충", "year": 2019, "summary": None, "scene": "반지하", 
         "confidence": 0.9, "characters": [
             {"name": "기택", "aliases": ["아버지"], "actor": None, "importance": "main", "role": None,
              "personality": None, "relationships": None, "appearance": "갈색 점퍼", "in_scene": True},
             {"name": "기우", "aliases": ["아들"], "actor": None, "importance": "main", "role": None,
              "personality": None, "relationships": None, "appearance": None, "in_scene": True},
         ]},
        {"characters": [
            {"name": "기택", "costume": {"hex": "#8B5A2B", "evidence": "갈색 점퍼"}, "symbolic": {"hex": None, "evidence": None}, "confidence": 0.9},
            {"name": "기우", "costume": {"hex": None, "evidence": None}, "symbolic": {"hex": "#1F4E9C", "evidence": "냉정"}, "confidence": 0.7},
        ]},
        {"speakers": [
            {"label": "SPEAKER_00", "character": "기우", "confidence": 0.85, "reasoning": "아버지를 부름"},
            {"label": "SPEAKER_01", "character": "기택", "confidence": 0.9, "reasoning": "대답"},
        ]},
    ]


def test_run_lookup_reads_no_media_and_writes_the_annotated_json(tmp_path, monkeypatch):
    def no_audio(path):
        raise AssertionError("run_lookup must not touch media")
    monkeypatch.setattr(pipeline, "decode_audio", no_audio)

    llm = FakeLLM(*_full_lookup_answers())
    out = tmp_path / "out.json"
    result = pipeline.run_lookup(str(_segments_file(tmp_path)), str(out), llm=llm)

    assert len(llm.calls) == 4
    assert result.media.title == "[기생충] 명장면 | 아들아 너는 계획이 다 있구나"   # from the file
    assert result.work.title == "기생충"
    assert result.character_map == {"SPEAKER_00": "기우", "SPEAKER_01": "기택"}
    assert all(w.volume == 0.0 for w in result.words)

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["work"]["characters"][0]["hue_basis"] == "costume"
    assert saved["speaker_profiles"]["SPEAKER_01"]["name"] == "기택"
    assert saved["speaker_profiles"]["SPEAKER_01"]["color"].startswith("&H00")
    assert not (tmp_path / "out.ass").exists()


def test_run_lookup_can_also_write_an_ass_and_takes_a_title_override(tmp_path):
    llm = FakeLLM(*_full_lookup_answers())
    out_json, out_ass = tmp_path / "o.json", tmp_path / "o.ass"
    result = pipeline.run_lookup(
        str(_segments_file(tmp_path)), str(out_json), str(out_ass),
        media_info=MediaInfo(title="기생충"), llm=llm,
    )
    assert result.media.title == "기생충"                 # explicit title wins
    assert "기생충" in llm.calls[0]["prompt"]
    assert out_ass.exists() and out_json.exists()


def test_run_lookup_replays_its_own_output_without_asking_again(tmp_path):
    first_out = tmp_path / "first.json"
    pipeline.run_lookup(str(_segments_file(tmp_path)), str(first_out), llm=FakeLLM(*_full_lookup_answers()))

    llm = FakeLLM()
    result = pipeline.run_lookup(str(first_out), str(tmp_path / "second.json"), llm=llm)
    assert llm.calls == []
    assert result.speaker_profiles["SPEAKER_01"].source == "stored"
    assert result.character_map == {"SPEAKER_00": "기우", "SPEAKER_01": "기택"}
