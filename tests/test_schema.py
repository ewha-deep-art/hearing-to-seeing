import pytest
from hearing_to_seeing.schema import WordEntry, Transcript


def test_word_entry_duration():
    w = WordEntry(text="hello", start=1.0, end=1.5)
    assert w.duration == 0.5


def test_word_entry_defaults():
    w = WordEntry(text="hi", start=0.0, end=0.3)
    assert w.speaker == "SPEAKER_00"
    assert w.volume == 0.0


def test_transcript_speakers_deduplicated_and_sorted():
    t = Transcript(words=[
        WordEntry("a", 0.0, 0.1, speaker="SPEAKER_01"),
        WordEntry("b", 0.1, 0.2, speaker="SPEAKER_00"),
        WordEntry("c", 0.2, 0.3, speaker="SPEAKER_01"),
    ])
    assert t.speakers() == ["SPEAKER_00", "SPEAKER_01"]


def test_transcript_to_dict_roundtrip():
    original = Transcript(
        words=[
            WordEntry("hello", 0.0, 0.5, speaker="SPEAKER_00", volume=0.75),
            WordEntry("world", 0.6, 1.0, speaker="SPEAKER_01", volume=0.25),
        ],
        language="en",
    )
    data = original.to_dict()
    restored = Transcript.from_dict(data)

    assert restored.language == "en"
    assert len(restored.words) == 2
    assert restored.words[0].text == "hello"
    assert restored.words[0].volume == 0.75
    assert restored.words[1].speaker == "SPEAKER_01"


def test_transcript_from_dict_missing_optional_fields():
    data = {
        "words": [{"text": "test", "start": 0.0, "end": 0.5}]
    }
    t = Transcript.from_dict(data)
    assert t.language == "ko"
    assert t.words[0].speaker == "SPEAKER_00"
    assert t.words[0].volume == 0.0


def test_word_entry_duration_precision():
    w = WordEntry(text="x", start=0.1, end=0.4)
    assert abs(w.duration - 0.3) < 1e-9


def test_transcript_empty():
    t = Transcript()
    assert t.speakers() == []
    assert t.to_dict() == {"language": "ko", "words": []}
