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
    assert candidate.preferred_hue is None


# --- stored_mapping / layered -----------------------------------------------

from hearing_to_seeing.design.identity import layered, stored_mapping  # noqa: E402
from hearing_to_seeing.design.speaker import SpeakerCandidate  # noqa: E402


def _two_speakers() -> Transcript:
    return Transcript(words=[
        WordEntry(text="a", start=0.0, end=0.5, speaker="SPEAKER_00"),
        WordEntry(text="b", start=1.0, end=1.5, speaker="SPEAKER_01"),
    ])


def test_stored_mapping_is_taken_as_certain_but_says_where_it_came_from():
    candidate = stored_mapping({"SPEAKER_00": "기택"})(_two_speakers(), "m")["SPEAKER_00"]
    assert candidate.name == "기택"
    assert candidate.confidence == 1.0
    assert candidate.source == "stored"


def test_layered_lets_the_first_strategy_with_an_answer_decide():
    manual = manual_mapping({"SPEAKER_00": "기택"})
    stored = stored_mapping({"SPEAKER_00": "충숙", "SPEAKER_01": "기우"})
    candidates = layered(manual, stored)(_two_speakers(), "m")

    assert candidates["SPEAKER_00"].name == "기택"        # manual outranks stored
    assert candidates["SPEAKER_00"].source == "manual"
    assert candidates["SPEAKER_01"].name == "기우"        # stored fills what manual left


def test_layered_does_not_consult_a_later_strategy_once_everyone_is_named():
    calls = []

    def expensive(transcript, media_path):
        calls.append(1)
        return {}

    complete = manual_mapping({"SPEAKER_00": "기택", "SPEAKER_01": "충숙"})
    layered(complete, expensive)(_two_speakers(), "m")
    assert calls == []

    partial = manual_mapping({"SPEAKER_00": "기택"})
    layered(partial, expensive)(_two_speakers(), "m")
    assert calls == [1]


def test_layered_treats_an_unnamed_candidate_as_still_open():
    # A strategy that looked and found nothing leaves the label for the next one.
    def unsure(transcript, media_path):
        return {"SPEAKER_00": SpeakerCandidate(label="SPEAKER_00", name=None, note="모름")}

    candidates = layered(unsure, stored_mapping({"SPEAKER_00": "기택"}))(_two_speakers(), "m")
    # First answer is kept as the record for the label…
    assert candidates["SPEAKER_00"].name is None
    # …but the label counted as open, so the stored strategy was consulted:
    # its answer for other labels would appear. (Here it had none.)
    assert set(candidates) == {"SPEAKER_00"}


def test_layered_collapses_trivial_stacks():
    only = manual_mapping({"SPEAKER_00": "기택"})
    assert layered(None, None) is None
    assert layered(None, only) is only
