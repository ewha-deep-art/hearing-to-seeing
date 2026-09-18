import pytest

from hearing_to_seeing.design.identity import (
    SpeakerMapError,
    manual_mapping,
    parse_speaker_map,
)
from hearing_to_seeing.schema import Transcript, WordEntry


def test_parse_speaker_map_reads_pairs():
    assert parse_speaker_map("SPEAKER_00=기택,SPEAKER_01=충숙") == {
        "SPEAKER_00": "기택",
        "SPEAKER_01": "충숙",
    }


def test_parse_speaker_map_tolerates_whitespace_and_trailing_comma():
    assert parse_speaker_map("  SPEAKER_00 = 기택 ,  ") == {"SPEAKER_00": "기택"}


def test_parse_speaker_map_keeps_spaces_inside_a_name():
    assert parse_speaker_map("SPEAKER_00=박 사장") == {"SPEAKER_00": "박 사장"}


def test_parse_speaker_map_of_empty_string():
    assert parse_speaker_map("") == {}


@pytest.mark.parametrize("text", [
    "SPEAKER_00",          # no separator
    "=기택",               # no label
    "SPEAKER_00=",         # no name
])
def test_parse_speaker_map_rejects_malformed_entries(text):
    # Raised rather than skipped: subtitles still look fine with one speaker
    # silently unmapped, so a typo would go unnoticed.
    with pytest.raises(SpeakerMapError):
        parse_speaker_map(text)


def test_parse_speaker_map_rejects_a_duplicate_label():
    with pytest.raises(SpeakerMapError):
        parse_speaker_map("SPEAKER_00=기택,SPEAKER_00=충숙")


def test_manual_mapping_reports_certainty_and_no_colour_opinion():
    transcript = Transcript(
        words=[WordEntry(text="w", start=0.0, end=0.5, speaker="SPEAKER_00")],
    )
    candidates = manual_mapping({"SPEAKER_00": "기택"})(transcript, "clip.mp4")

    candidate = candidates["SPEAKER_00"]
    assert candidate.name == "기택"
    assert candidate.confidence == 1.0
    assert candidate.source == "manual"
    # Naming a speaker says nothing about which colour suits them.
    assert candidate.preference == {}
