from hearing_to_seeing.schema import MediaInfo, SpeakerProfile, Transcript, WordEntry


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
    assert transcript.language == "ko"


def test_from_char_timestamps_splits_on_whitespace():
    data = {
        "characters": [
            {"char": "h", "start": 0.0, "end": 0.1},
            {"char": "i", "start": 0.1, "end": 0.2},
            {"char": " ", "start": 0.2, "end": 0.25},
            {"char": "u", "start": 0.25, "end": 0.35},
        ]
    }
    transcript = Transcript.from_char_timestamps(data)
    assert [w.text for w in transcript.words] == ["hi", "u"]
    assert transcript.words[0].start == 0.0
    assert transcript.words[0].end == 0.2
    # No speaker/volume info in this format — both fall back to defaults.
    assert transcript.words[0].speaker == "SPEAKER_00"
    assert transcript.words[0].volume == 0.0


def test_from_char_timestamps_splits_on_large_gap_without_whitespace():
    # Two "words" glued together because the export dropped the boundary space,
    # but the silence between them is long enough to still tell them apart.
    data = {
        "characters": [
            {"char": "a", "start": 0.0, "end": 0.1},
            {"char": "b", "start": 1.0, "end": 1.1},
        ]
    }
    transcript = Transcript.from_char_timestamps(data)
    assert [w.text for w in transcript.words] == ["a", "b"]


def test_from_char_timestamps_empty_input():
    assert Transcript.from_char_timestamps({"characters": []}).words == []


def test_to_dict_from_dict_round_trip_keeps_media_info():
    original = Transcript(
        words=[WordEntry(text="hi", start=0.0, end=0.4)],
        media=MediaInfo(title="기생충", source_url="https://youtu.be/abc123"),
    )
    restored = Transcript.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_defaults_missing_media_info():
    transcript = Transcript.from_dict({
        "words": [{"text": "hi", "start": 0.0, "end": 0.4}],
    })
    assert transcript.media == MediaInfo()
    assert transcript.media.title is None


def test_speaker_profile_round_trip_keyed_by_label():
    original = Transcript(
        words=[WordEntry(text="hi", start=0.0, end=0.4, speaker="SPEAKER_00")],
        speaker_profiles={
            "SPEAKER_00": SpeakerProfile(
                label="SPEAKER_00", color="&H00009FE6", name="기택",
                confidence=0.91, source="llm", note="갈색 점퍼",
            ),
        },
    )
    restored = Transcript.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_rebuilds_a_profile_label_from_its_key():
    transcript = Transcript.from_dict({
        "words": [],
        "speaker_profiles": {"SPEAKER_02": {"color": "&H00E9B456"}},
    })
    profile = transcript.speaker_profiles["SPEAKER_02"]
    assert profile.label == "SPEAKER_02"
    assert profile.source == "palette"
    assert profile.confidence == 0.0


def test_from_dict_defaults_missing_speaker_profiles():
    transcript = Transcript.from_dict({"words": []})
    assert transcript.speaker_profiles == {}
