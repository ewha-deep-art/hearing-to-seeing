"""Label merging and utterance reassignment, with voice prints faked by time."""

import numpy as np
import pytest

from hearing_to_seeing import voice
from hearing_to_seeing.schema import Transcript, WordEntry

A, B = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


@pytest.fixture
def voices(monkeypatch):
    """Whose voice each second of audio is: {second: print}, the rest A."""
    table: dict[int, np.ndarray] = {}
    monkeypatch.setattr(voice, "_embed", lambda samples, start, end: table.get(int(start), A))
    return table


def _transcript(*spans: tuple[float, float, str]) -> Transcript:
    return Transcript(words=[WordEntry(text="말", start=s, end=e, speaker=label) for s, e, label in spans])


AUDIO = np.zeros(16_000 * 60, dtype=np.int16)


def test_labels_with_the_same_voice_are_merged(voices):
    voices.update({10: B, 20: B})
    t = _transcript((0, 2, "S0"), (5, 7, "S1"), (10, 12, "S2"), (20, 22, "S3"))
    assert voice.merge_labels(t, 16_000, AUDIO) == {"S1": "S0", "S3": "S2"}
    assert [w.speaker for w in t.words] == ["S0", "S0", "S2", "S2"]


def test_different_voices_keep_their_labels(voices):
    voices.update({10: B})
    t = _transcript((0, 2, "S0"), (10, 12, "S1"))
    assert voice.merge_labels(t, 16_000, AUDIO) == {}


def test_an_utterance_in_someone_elses_voice_moves_to_them(voices):
    voices.update({k: B for k in (10, 20, 30)})
    t = _transcript((0, 2, "S0"), (3, 5, "S0"), (10, 12, "S1"), (20, 22, "S1"), (30, 32, "S0"))
    assert voice.reassign_utterances(t, 16_000, AUDIO) == 1
    assert [w.speaker for w in t.words] == ["S0", "S0", "S1", "S1", "S1"]


def test_short_utterances_and_unknown_voices_stay(voices):
    voices.update({k: B for k in (10, 20, 30)})
    t = _transcript((0, 2, "S0"), (3, 5, "S0"), (10, 12, "S1"), (20, 22, "S1"), (30, 30.5, "S0"))
    assert voice.reassign_utterances(t, 16_000, AUDIO) == 0


def test_without_a_model_the_labels_are_kept(monkeypatch):
    def broken(*_):
        raise OSError("offline")

    monkeypatch.setattr(voice, "_embed", broken)
    t = _transcript((0, 2, "S0"), (5, 7, "S1"))
    assert voice.merge_labels(t, 16_000, AUDIO) == {} and voice.reassign_utterances(t, 16_000, AUDIO) == 0
    assert [w.speaker for w in t.words] == ["S0", "S1"]


def test_audio_is_brought_to_mono_16k_on_the_int16_scale():
    stereo = np.full((48_000, 2), 0.5, dtype=np.float32)
    out = voice._mono_16k(48_000, stereo)
    assert out.shape == (16_000,) and abs(out[8000] - 16384) < 1


def _response(*words: tuple[str, float, float]) -> dict:
    return {"segments": [{"text": "", "speaker": "SPEAKER_09",
                          "words": [{"word": t, "start": s, "end": e, "speaker": "SPEAKER_09"} for t, s, e in words]}]}


def test_diarize_groups_utterances_by_voice(voices):
    voices.update({10: B, 11: B, 30: B})
    result = _response(("안녕.", 0, 1.5), ("응", 10, 11.5), ("뭐해?", 20, 21.5), ("그냥", 30, 31.5), ("음", 40, 40.2))

    voice.diarize(result, 16_000, AUDIO)

    labels = [w["speaker"] for w in result["segments"][0]["words"]]
    assert labels[0] == labels[2] == labels[4] != labels[1] == labels[3]  # the short one joins its neighbour
    assert result["segments"][0]["speaker"] is None


def test_diarize_cuts_where_the_voice_changes_mid_sentence(voices):
    # One run of speech with no pause: A until 12 s, then B.
    voices.update({12: B, 13: B, 14: B, 15: B})
    result = _response(*[("말", float(s), s + 0.9) for s in range(8, 16)], ("끝", 30, 32))

    voice.diarize(result, 16_000, AUDIO)

    labels = [w["speaker"] for w in result["segments"][0]["words"]]
    assert labels[:4] == [labels[0]] * 4 and labels[4:8] == [labels[4]] * 4 and labels[0] != labels[4]


def test_diarize_takes_a_known_speaker_count(voices):
    # Alike enough to group by similarity alone; a count of two splits them.
    voices.update({10: np.array([0.6, 0.8, 0.0]), 20: B})
    words = [("가.", 0, 1.5), ("나.", 10, 11.5), ("다.", 20, 21.5)]

    by_similarity, by_count = _response(*words), _response(*words)
    voice.diarize(by_similarity, 16_000, AUDIO)
    voice.diarize(by_count, 16_000, AUDIO, num_speakers=2)

    labels = lambda r: [w["speaker"] for w in r["segments"][0]["words"]]  # noqa: E731
    assert len(set(labels(by_similarity))) == 1
    a, b, c = labels(by_count)
    assert a != b == c
