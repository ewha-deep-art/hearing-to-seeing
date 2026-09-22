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


def test_to_dict_from_dict_round_trip_keeps_the_work_block():
    from hearing_to_seeing.schema import Character, WorkInfo

    original = Transcript(
        words=[WordEntry(text="hi", start=0.0, end=0.4)],
        work=WorkInfo(
            title="기생충", year=2019, confidence=0.9,
            characters=[Character(name="기택", aliases=["아버지"], hue=40.0, hue_basis="costume")],
        ),
    )
    restored = Transcript.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_without_a_work_block_keeps_it_unset():
    # Never looked up is not the same as looked up and found nothing.
    assert Transcript.from_dict({"words": []}).work is None
    assert Transcript.from_dict({"words": [], "work": None}).work is None


def test_work_info_find_matches_name_or_alias_ignoring_spaces():
    from hearing_to_seeing.schema import Character, WorkInfo

    work = WorkInfo(characters=[Character(name="박 사장", aliases=["사장님"])])
    assert work.find("박사장") is work.characters[0]
    assert work.find("사장님") is work.characters[0]
    assert work.find("기택") is None


def test_character_map_round_trips_and_drops_blank_names():
    original = Transcript(words=[], character_map={"SPEAKER_00": "기택"})
    assert Transcript.from_dict(original.to_dict()) == original
    loaded = Transcript.from_dict({"words": [], "character_map": {"SPEAKER_00": "기택", "SPEAKER_01": "  "}})
    assert loaded.character_map == {"SPEAKER_00": "기택"}


def test_turns_join_consecutive_words_of_one_speaker():
    transcript = Transcript(words=[
        WordEntry(text="아버지", start=0.0, end=0.4, speaker="SPEAKER_00"),
        WordEntry(text="보세요", start=0.5, end=0.9, speaker="SPEAKER_00"),
        WordEntry(text="뭔데", start=1.2, end=1.5, speaker="SPEAKER_01"),
        WordEntry(text="응", start=2.0, end=2.2, speaker="SPEAKER_00"),
    ])
    assert transcript.turns() == [
        ("SPEAKER_00", 0.0, 0.9, "아버지 보세요"),
        ("SPEAKER_01", 1.2, 1.5, "뭔데"),
        ("SPEAKER_00", 2.0, 2.2, "응"),
    ]


def test_from_segments_reads_the_project_input_format():
    transcript = Transcript.from_segments({
        "video_title": "[왕과 사는 남자] 공식 예고편",
        "video_url": "https://www.youtube.com/watch?v=abc",
        "segments": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "청룡포라고 들어보셨습니까"},
            {"speaker": "SPEAKER_02", "start": 2.0, "end": 3.0, "text": "누가 오든"},
            {"speaker": "SPEAKER_02", "start": 3.0, "end": 3.5, "text": "   "},
        ],
    })
    assert transcript.media.title == "[왕과 사는 남자] 공식 예고편"
    assert transcript.media.source_url == "https://www.youtube.com/watch?v=abc"
    assert [w.text for w in transcript.words] == ["청룡포라고", "들어보셨습니까", "누가", "오든"]
    # Words share the segment's span evenly, in order.
    assert (transcript.words[0].start, transcript.words[0].end) == (0.0, 1.0)
    assert (transcript.words[1].start, transcript.words[1].end) == (1.0, 2.0)
    assert transcript.words[2].speaker == "SPEAKER_02"
    assert transcript.speakers() == ["SPEAKER_00", "SPEAKER_02"]
