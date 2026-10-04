import threading

import numpy as np
import pytest

from hearing_to_seeing import pipeline
from hearing_to_seeing.schema import Transcript, WordEntry


def _transcript() -> Transcript:
    return Transcript(words=[
        WordEntry(text=t, start=i * 0.4, end=i * 0.4 + 0.3, speaker=f"SPEAKER_0{i // 3}")
        for i, t in enumerate("기택아 밥 먹었니 네 먹었어요 방금".split())
    ])


@pytest.fixture(autouse=True)
def no_audio(monkeypatch):
    monkeypatch.setattr(pipeline, "decode_audio", lambda _: (16000, np.ones(16000 * 5)))


def test_lookup_runs_while_stt_is_in_flight(monkeypatch, tmp_path):
    looked_up = threading.Event()
    seen = {}

    def transcribe(*_, **__):
        # Only returns if the lookup started without waiting for STT.
        assert looked_up.wait(timeout=5), "lookup did not run concurrently"
        return _transcript()

    def lookup_characters(title):
        looked_up.set()
        return [{"name": "기택"}]

    def assign_speaker_colors(transcript, characters):
        seen["characters"] = characters
        return {"SPEAKER_00": {"color": "&H00123456", "name": "기택"}}

    monkeypatch.setattr(pipeline, "transcribe", transcribe)
    monkeypatch.setattr(pipeline, "lookup_characters", lookup_characters)
    monkeypatch.setattr(pipeline, "assign_speaker_colors", assign_speaker_colors)

    transcript = pipeline.run("a.wav", str(tmp_path / "a.ass"), title="기생충 명장면")

    assert seen["characters"] == [{"name": "기택"}]
    assert transcript.speaker_profiles["SPEAKER_00"]["name"] == "기택"
    assert "&H00123456" in (tmp_path / "a.ass").read_text(encoding="utf-8")


def test_without_a_title_speakers_still_get_distinct_colours(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "transcribe", lambda *_, **__: _transcript())

    transcript = pipeline.run("a.wav", str(tmp_path / "a.ass"), output_json=str(tmp_path / "a.json"))

    colours = [p["color"] for p in transcript.speaker_profiles.values()]
    assert len(set(colours)) == 2
    assert '"speaker_profiles"' in (tmp_path / "a.json").read_text(encoding="utf-8")
