import pytest

from hearing_to_seeing import review
from hearing_to_seeing.schema import WordEntry


def _word(text, start, end, speaker="SPEAKER_00"):
    return {"word": text, "start": start, "end": end, "speaker": speaker}


def _result(*words):
    return {"language": "ko", "segments": [{"text": "", "speaker": words[0]["speaker"], "words": list(words)}]}


def _passes():
    """Two lines diarized as one speaker; 다리가 is heard by one pass of three."""
    def line(first):
        return _result(
            _word("이거", 1.0, 1.3), _word("뭐야?", 1.4, 1.8),
            _word(first, 3.0, 3.4, "SPEAKER_00"), _word("안", 3.5, 3.6), _word("떨어져요", 3.7, 4.2),
        )
    return [line("파리가"), line("다리가"), line("파리가")]


@pytest.fixture
def model(monkeypatch):
    """Stands in for `llm.ask`; `model.answer` is what it returns (or raises)."""
    calls = []

    def ask(system, prompt, schema, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        if isinstance(model.answer, Exception):
            raise model.answer
        return model.answer

    monkeypatch.setattr(review.llm, "configured", lambda: True)
    monkeypatch.setattr(review.llm, "ask", ask)
    model.calls = calls
    model.answer = {"utterances": [], "corrections": []}
    return model


def test_utterances_break_at_sentence_ends_pauses_and_label_changes():
    words = [
        {"text": "이거", "start": 0.0, "end": 0.2, "speaker": "A"},
        {"text": "뭐야?", "start": 0.3, "end": 0.5, "speaker": "A"},   # sentence end
        {"text": "몰라", "start": 0.6, "end": 0.8, "speaker": "A"},
        {"text": "정말", "start": 1.5, "end": 1.7, "speaker": "A"},    # pause
        {"text": "응", "start": 1.8, "end": 1.9, "speaker": "B"},      # label change
    ]
    assert [[w["text"] for w in u] for u in review._utterances(words)] == [["이거", "뭐야?"], ["몰라"], ["정말"], ["응"]]


def test_the_prompt_marks_each_disputed_spot_in_place(model):
    review.review_transcript(_passes(), "ko", [{"name": "스폰지밥"}], "clip.mp4")

    prompt = model.calls[0]["prompt"]
    assert "u1 [1.0–1.8s] SPEAKER_00: 이거 뭐야?" in prompt
    assert "u2 [3.0–4.2s] SPEAKER_00: ⟨d0 @3.0s: 1) 파리가 ×2 | 2) 다리가 ×1⟩ 안 떨어져요" in prompt
    assert "스폰지밥" in prompt
    assert "mouth moves on screen" in prompt  # judged by the picture, not the audio


def test_the_model_watches_the_clip_without_a_fallback(model):
    review.review_transcript(_passes(), "ko", [{"name": "스폰지밥"}], "clip.mp4")
    call = model.calls[0]
    assert (call["media_path"], call["fallback"], call["temperature"]) == ("clip.mp4", False, 0)
    assert (call["model"], call["thinking_level"]) == (review.REVIEW_MODEL, None)


def test_without_a_cast_only_speakers_are_asked_with_little_thinking(model):
    model.answer = {"utterances": [{"id": "u2", "speaker": "스폰지밥"}], "corrections": [{"id": "d0", "option": 2}]}

    result = review.review_transcript(_passes(), "ko", None, "clip.mp4")

    call = model.calls[0]
    assert call["thinking_level"] == "low"
    assert "⟨" not in call["prompt"] and "Task 2" not in call["prompt"]
    assert "파리가 안 떨어져요" in call["prompt"]
    assert [w.text for w in result.transcript.words][2] == "파리가"  # the stray correction is ignored


def test_an_audio_file_is_listened_to(model):
    review.review_transcript(_passes(), "ko", None, "clip.wav")
    assert "by the voice and the dialogue" in model.calls[0]["prompt"]


def test_the_answer_relabels_speakers_and_picks_readings(model):
    model.answer = {
        "utterances": [{"id": "u1", "speaker": "징징이"}, {"id": "u2", "speaker": "스폰지밥"}],
        "corrections": [{"id": "d0", "option": 2}],
    }

    result = review.review_transcript(_passes(), "ko", [{"name": "스폰지밥"}], "clip.mp4")

    words = result.transcript.words
    assert [w.text for w in words] == ["이거", "뭐야?", "다리가", "안", "떨어져요"]
    assert words[0].speaker != words[2].speaker
    assert result.names == {words[0].speaker: "징징이", words[2].speaker: "스폰지밥"}
    assert set(result.names) >= {"SPEAKER_00"}  # one of them keeps the diarization label


def test_answers_that_point_nowhere_are_ignored(model):
    model.answer = {
        "utterances": [{"id": "u9", "speaker": "징징이"}, {"id": "x", "speaker": "뚱이"}],
        "corrections": [{"id": "d0", "option": 7}, {"id": "d5", "option": 2}, {"id": "zz", "option": 2}],
    }

    result = review.review_transcript(_passes(), "ko", [{"name": "뚱이"}], "clip.mp4")

    assert [w.text for w in result.transcript.words] == ["이거", "뭐야?", "파리가", "안", "떨어져요"]
    assert {w.speaker for w in result.transcript.words} == {"SPEAKER_00"}
    assert result.names == {}


def test_a_failed_call_leaves_the_vote_to_the_caller(model):
    model.answer = RuntimeError("429 quota")
    assert review.review_transcript(_passes(), "ko", None, "clip.mp4") is None


def test_no_review_without_a_key(model, monkeypatch):
    monkeypatch.setattr(review.llm, "configured", lambda: False)
    assert review.review_transcript(_passes(), "ko", None, "clip.mp4") is None
    assert model.calls == []


def test_a_clip_too_long_is_not_sent(model, monkeypatch):
    monkeypatch.setattr(review, "MAX_SECONDS", 2.0)
    assert review.review_transcript(_passes(), "ko", None, "clip.mp4") is None
    assert model.calls == []


def test_relabel_gives_each_name_the_label_it_overlaps_most():
    words = [WordEntry("가", 0, 1, "SPEAKER_00"), WordEntry("나", 1, 2, "SPEAKER_00"),
             WordEntry("다", 2, 3, "SPEAKER_01"), WordEntry("라", 3, 4, "SPEAKER_00"),
             WordEntry("마", 4, 5, "SPEAKER_01")]
    names = ["집게사장", "집게사장", "집게사장", "손님1", None]

    labels = review._relabel(words, names)

    # 집게사장 holds most of SPEAKER_00; 손님1 needs a label of its own, since
    # SPEAKER_01 is still spoken by an unnamed word.
    assert [w.speaker for w in words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00", "SPEAKER_02", "SPEAKER_01"]
    assert labels == {"SPEAKER_00": "집게사장", "SPEAKER_02": "손님1"}


def test_an_early_upload_is_used_and_not_uploaded_again(model):
    review.review_transcript(_passes(), "ko", [{"name": "스폰지밥"}], "clip.mp4", "uploaded")

    assert model.calls[0]["media"] == "uploaded"


def test_an_early_upload_is_discarded_when_there_is_nothing_to_review(model, monkeypatch):
    discarded = []
    monkeypatch.setattr(review.llm, "discard", discarded.append)

    assert review.review_transcript([_result(_word("x", 700.0, 700.5))], "ko", None, "clip.mp4", "up") is None
    assert discarded == ["up"] and model.calls == []


def test_upload_skips_clips_too_long_to_review(monkeypatch):
    monkeypatch.setattr(review.llm, "configured", lambda: True)
    monkeypatch.setattr(review.llm, "upload", lambda path: f"up {path}")

    assert review.upload("clip.mp4", review.MAX_SECONDS + 1) is None
    assert review.upload("clip.mp4", 60.0) == "up clip.mp4"


def test_a_failed_early_upload_leaves_the_review_to_upload(monkeypatch):
    monkeypatch.setattr(review.llm, "configured", lambda: True)

    def fail(path):
        raise RuntimeError("network")

    monkeypatch.setattr(review.llm, "upload", fail)
    assert review.upload("clip.mp4", 60.0) is None
