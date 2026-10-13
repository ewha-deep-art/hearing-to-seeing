from hearing_to_seeing.schema import Transcript, WordEntry


def test_word_entry_duration_rounds_to_milliseconds():
    word = WordEntry(text="a", start=0.1, end=0.30005)
    assert word.duration == 0.2


def test_transcript_speakers_returns_sorted_unique():
    transcript = Transcript(words=[
        WordEntry(text="a", start=0, end=1, speaker="SPEAKER_01"),
        WordEntry(text="b", start=1, end=2, speaker="SPEAKER_00"),
        WordEntry(text="c", start=2, end=3, speaker="SPEAKER_01"),
    ])
    assert transcript.speakers() == ["SPEAKER_00", "SPEAKER_01"]


def test_to_dict_from_dict_round_trip():
    original = Transcript(
        words=[
            WordEntry(text="hello", start=0.0, end=0.5, speaker="SPEAKER_00", volume=0.7),
            WordEntry(text="world", start=0.6, end=1.1, speaker="SPEAKER_01", volume=0.3),
        ],
        language="en",
    )
    restored = Transcript.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_defaults_missing_speaker_and_volume():
    transcript = Transcript.from_dict({
        "words": [{"text": "hi", "start": 0.0, "end": 0.4}],
    })
    word = transcript.words[0]
    assert word.speaker == "SPEAKER_00"
    assert word.volume == 0.0
    assert transcript.language is None


def test_speaker_profiles_round_trip():
    profiles = {"SPEAKER_00": {"color": "&H00123456", "name": "기택", "confidence": 0.9, "reason": "호명"}}
    t = Transcript(words=[WordEntry(text="a", start=0.0, end=1.0)], speaker_profiles=profiles)
    assert Transcript.from_dict(t.to_dict()).speaker_profiles == profiles
    assert Transcript.from_dict({"words": []}).speaker_profiles == {}
