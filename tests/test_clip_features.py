import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from clip_features import duration_bin, features, joined, speakers_bin  # noqa: E402


def word(text, start, end, speaker="A", volume=0.5):
    return {"text": text, "start": start, "end": end, "speaker": speaker, "volume": volume}


def test_counts_bit_parts_turns_and_overlap():
    words = [word(f"말{i}", i, i + 0.5) for i in range(12)]
    words += [word("응", 3.2, 3.8, speaker="B"), word("...", 20, 21, speaker="C")]
    f = features({"words": words}, {"genre": "애니메이션", "duration": 60})
    assert f["animation"] and f["duration"] == 60
    assert f["words"] == 13  # "..." has no text left once punctuation goes
    assert f["speakers"] == 2 and f["minor_speakers"] == 1
    assert f["minor_word_share"] == round(1 / 13, 4)
    assert f["turns_per_min"] == 2  # A → B → A in one minute
    assert f["overlap_share"] == round(2 / 13, 4)  # B's word and A's word at 3 s
    # speech time is the union of word spans: 12 × 0.5 s, plus B's 0.3 s past 3.5 s
    assert f["speech_ratio"] == round(6.3 / 60, 4)
    assert f["chars_per_sec"] == round(27 / 6.3, 2)  # 말0–말9: 20, 말10·말11: 6, 응: 1


def test_whispered_words_and_review():
    words = [word("작게", i, i + 0.5, volume=0.02 if i < 3 else 0.5) for i in range(10)]
    profiles = {"A": {"confidence": 0.8, "reason": "영상 검토에서 말하는 인물로 확인"}}
    f = features({"words": words, "speaker_profiles": profiles})
    assert f["whisper_share"] == 0.3
    assert f["reviewed"] and f["character_confidence"] == 0.8
    assert f["duration"] == 9.5  # no meta: where the last word ends


def test_empty_transcript_and_bins():
    f = features({"words": []})
    assert f["words"] == 0 and f["speech_ratio"] == 0.0 and not f["reviewed"]
    assert joined({"words": [word("안녕,", 0, 1), word("!", 1, 2)]}) == "안녕"
    assert [duration_bin(s) for s in (300, 301, 601)] == ["≤5분", "5–10분", ">10분"]
    assert [speakers_bin(n) for n in (3, 6, 7)] == ["≤3명", "4–6명", "≥7명"]
