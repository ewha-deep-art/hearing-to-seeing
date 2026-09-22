"""The `lookup` subcommand end to end, with the model faked at the CLI's seam."""

import json

import pytest

from hearing_to_seeing import cli


class FakeLLM:
    def __init__(self, *answers):
        self.answers = list(answers)

    def structured(self, system, prompt, schema):
        return self.answers.pop(0)


def _segments_file(tmp_path):
    src = tmp_path / "seg.json"
    src.write_text(json.dumps({
        "video_title": "[기생충] 명장면",
        "segments": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "아버지 이거 좀 보세요"},
            {"speaker": "SPEAKER_01", "start": 2.5, "end": 4.0, "text": "뭔데 그래 인마 어서"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    return src


def test_lookup_without_a_key_explains_and_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "configured", lambda: False)
    code = cli.main(["lookup", str(_segments_file(tmp_path))])
    assert code == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_lookup_with_a_missing_file_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "configured", lambda: True)
    assert cli.main(["lookup", str(tmp_path / "nope.json")]) == 2
    assert "not found" in capsys.readouterr().err


def test_lookup_writes_beside_the_input_and_reports(tmp_path, monkeypatch, capsys):
    llm = FakeLLM(
        {"title": "기생충", "year": 2019, "scene_hint": None, "confidence": 0.9},
        {"title": "기생충", "year": 2019, "summary": None, "scene": None, 
         "confidence": 0.9, "characters": [
             {"name": "기택", "aliases": ["아버지"], "actor": None, "importance": "main", "role": None,
              "personality": None, "relationships": None, "appearance": None, "in_scene": True}]},
        {"characters": [{"name": "기택", "costume": {"hex": None, "evidence": None},
                         "symbolic": {"hex": "#C0392B", "evidence": "분노"}, "confidence": 0.8}]},
        {"speakers": [{"label": "SPEAKER_01", "character": "기택", "confidence": 0.9, "reasoning": "대답"},
                      {"label": "SPEAKER_00", "character": None, "confidence": 0.2, "reasoning": "불명"}]},
    )
    monkeypatch.setattr(cli, "configured", lambda: True)
    monkeypatch.setattr(cli, "make_llm", lambda: llm)
    import hearing_to_seeing.knowledge as knowledge
    from hearing_to_seeing.knowledge.retrieve import Document
    monkeypatch.setattr(knowledge, "retrieve", lambda title: [Document("나무위키", title, "https://namu.wiki/w/x", "## 등장인물\n기택")])

    src = _segments_file(tmp_path)
    monkeypatch.chdir(tmp_path)                       # data/output/ is relative to the cwd
    assert cli.main(["lookup", str(src)]) == 0

    out = tmp_path / "data" / "output" / "seg.lookup.json"
    assert out.exists()                               # not beside the input
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["character_map"] == {"SPEAKER_01": "기택"}

    text = capsys.readouterr().out
    assert "SPEAKER_01 = 기택  [text, 0.90]" in text
    assert "SPEAKER_00 = ?" in text
    assert "기택: hue" in text and "(symbolic)" in text
    assert "namu.wiki" in text
