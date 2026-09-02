import pytest
from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.converter.ass import generate_ass, _fmt_time, _group_by_speaker


def _make_transcript():
    return Transcript(
        words=[
            WordEntry("Hello", 1.0, 1.3, speaker="SPEAKER_00", volume=0.5),
            WordEntry("world", 1.4, 1.7, speaker="SPEAKER_00", volume=0.8),
            WordEntry("안녕", 2.0, 2.4, speaker="SPEAKER_01", volume=0.2),
        ],
        language="en",
    )


def test_generate_ass_contains_script_info():
    output = generate_ass(_make_transcript())
    assert "[Script Info]" in output
    assert "ScriptType: v4.00+" in output


def test_generate_ass_contains_styles():
    output = generate_ass(_make_transcript())
    assert "[V4+ Styles]" in output
    assert "Style: Default" in output


def test_generate_ass_contains_events():
    output = generate_ass(_make_transcript())
    assert "[Events]" in output
    assert "Dialogue:" in output


def test_generate_ass_two_dialogue_lines():
    output = generate_ass(_make_transcript())
    lines = [l for l in output.splitlines() if l.startswith("Dialogue:")]
    assert len(lines) == 2


def test_generate_ass_karaoke_tags_present():
    output = generate_ass(_make_transcript())
    assert r"\k" in output


def test_generate_ass_font_size_tags_present():
    output = generate_ass(_make_transcript())
    assert r"\fs" in output


def test_generate_ass_speaker_names_in_dialogue():
    output = generate_ass(_make_transcript())
    assert "SPEAKER_00" in output
    assert "SPEAKER_01" in output


def test_fmt_time_zero():
    assert _fmt_time(0.0) == "0:00:00.00"


def test_fmt_time_one_minute():
    assert _fmt_time(61.5) == "0:01:01.50"


def test_fmt_time_one_hour():
    assert _fmt_time(3661.25) == "1:01:01.25"


def test_group_by_speaker_consecutive():
    words = [
        WordEntry("a", 0.0, 0.1, speaker="SPEAKER_00"),
        WordEntry("b", 0.1, 0.2, speaker="SPEAKER_00"),
        WordEntry("c", 0.2, 0.3, speaker="SPEAKER_01"),
    ]
    groups = _group_by_speaker(words)
    assert len(groups) == 2
    assert groups[0][0] == "SPEAKER_00"
    assert len(groups[0][1]) == 2
    assert groups[1][0] == "SPEAKER_01"


def test_group_by_speaker_alternating():
    words = [
        WordEntry("a", 0.0, 0.1, speaker="SPEAKER_00"),
        WordEntry("b", 0.1, 0.2, speaker="SPEAKER_01"),
        WordEntry("c", 0.2, 0.3, speaker="SPEAKER_00"),
    ]
    groups = _group_by_speaker(words)
    assert len(groups) == 3


def test_generate_ass_empty_transcript():
    t = Transcript(words=[])
    output = generate_ass(t)
    assert "[Script Info]" in output
    dialogue_lines = [l for l in output.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue_lines) == 0


def test_generate_ass_color_codes_present():
    output = generate_ass(_make_transcript())
    # Color codes in ASS format start with &H
    assert "&H" in output
