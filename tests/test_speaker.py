import pytest

from hearing_to_seeing.design.oklch import from_ass, rgb_to_oklch
from hearing_to_seeing.design.speaker import (
    BASE_COLOUR,
    MIN_HUE_GAP,
    SpeakerCandidate,
    assign_hues,
    hue_distance,
    resolve_speakers,
)
from hearing_to_seeing.schema import Transcript, WordEntry


def _transcript(*labels: str) -> Transcript:
    return Transcript(
        words=[
            WordEntry(text="w", start=float(i), end=float(i) + 0.5, speaker=label)
            for i, label in enumerate(labels)
        ]
    )


def _chroma(color: str) -> float:
    return rgb_to_oklch(*from_ass(color))[1]


def _hue(color: str) -> float:
    return rgb_to_oklch(*from_ass(color))[2]


# --- hue_distance ------------------------------------------------------------

def test_hue_distance_takes_the_shorter_way_round():
    assert hue_distance(10, 350) == pytest.approx(20)
    assert hue_distance(0, 180) == pytest.approx(180)


# --- assign_hues -------------------------------------------------------------

def test_assign_hues_without_preferences_spreads_evenly():
    hues = assign_hues(["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"])
    assert sorted(hues.values()) == pytest.approx([0, 90, 180, 270])


def test_assign_hues_honours_a_preference_exactly():
    hues = assign_hues(["SPEAKER_00"], preferences={"SPEAKER_00": 137.0})
    assert hues["SPEAKER_00"] == pytest.approx(137.0)


def test_assign_hues_moves_the_loser_of_a_collision_aside():
    # Both want the same hue; the first keeps it and the second is pushed to
    # the nearest spot that clears the minimum gap.
    hues = assign_hues(
        ["SPEAKER_00", "SPEAKER_01"],
        preferences={"SPEAKER_00": 100.0, "SPEAKER_01": 100.0},
    )
    assert hues["SPEAKER_00"] == pytest.approx(100.0)
    assert hue_distance(hues["SPEAKER_01"], 100.0) == pytest.approx(MIN_HUE_GAP)


def test_assign_hues_lets_the_heavier_weight_win():
    hues = assign_hues(
        ["SPEAKER_00", "SPEAKER_01"],
        preferences={"SPEAKER_00": 100.0, "SPEAKER_01": 100.0},
        weights={"SPEAKER_00": 0.2, "SPEAKER_01": 0.9},
    )
    assert hues["SPEAKER_01"] == pytest.approx(100.0)
    assert hue_distance(hues["SPEAKER_00"], 100.0) == pytest.approx(MIN_HUE_GAP)


def test_assign_hues_puts_an_unpreferred_speaker_in_the_widest_gap():
    hues = assign_hues(
        ["SPEAKER_00", "SPEAKER_01"], preferences={"SPEAKER_00": 0.0},
    )
    assert hues["SPEAKER_01"] == pytest.approx(180.0)


def test_assign_hues_keeps_everyone_apart():
    labels = [f"SPEAKER_{i:02d}" for i in range(6)]
    hues = assign_hues(labels, preferences={label: 200.0 for label in labels})

    values = list(hues.values())
    assert len(hues) == len(labels)
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            assert hue_distance(a, b) > 1.0


def test_assign_hues_tightens_the_gap_for_a_large_cast():
    # Twenty speakers cannot all sit MIN_HUE_GAP apart, so the gap shrinks
    # rather than leaving anyone unplaced.
    labels = [f"SPEAKER_{i:02d}" for i in range(20)]
    hues = assign_hues(labels, preferences={labels[0]: 0.0})
    assert len(hues) == 20
    assert len({round(hue, 6) for hue in hues.values()}) == 20


def test_assign_hues_of_no_labels():
    assert assign_hues([]) == {}


# --- resolve_speakers --------------------------------------------------------

def test_resolve_speakers_without_candidates_gives_distinct_colours():
    profiles = resolve_speakers(_transcript("SPEAKER_00", "SPEAKER_01"))

    assert len({p.color for p in profiles.values()}) == 2
    assert all(p.name is None for p in profiles.values())
    assert all(p.source == "unidentified" for p in profiles.values())


def test_resolve_speakers_keeps_full_chroma_when_nobody_was_identified():
    # Muted means "not one of the named characters" — with no named character
    # it would say nothing, and only make the whole film paler.
    alone = resolve_speakers(_transcript("SPEAKER_00", "SPEAKER_01"))
    mixed = resolve_speakers(
        _transcript("SPEAKER_00", "SPEAKER_01"),
        {"SPEAKER_00": SpeakerCandidate(label="SPEAKER_00", name="기택", confidence=0.9)},
    )
    assert _chroma(alone["SPEAKER_01"].color) > _chroma(mixed["SPEAKER_01"].color)


def test_resolve_speakers_makes_an_identified_speaker_more_vivid():
    profiles = resolve_speakers(
        _transcript("SPEAKER_00", "SPEAKER_01"),
        {
            "SPEAKER_00": SpeakerCandidate(
                label="SPEAKER_00", name="기택", confidence=0.9, source="llm",
            )
        },
    )
    # Vivid means "we know who this is"; the unidentified one stays muted.
    assert _chroma(profiles["SPEAKER_00"].color) > _chroma(profiles["SPEAKER_01"].color)
    assert profiles["SPEAKER_00"].name == "기택"
    assert profiles["SPEAKER_01"].name is None


def test_resolve_speakers_honours_a_preferred_hue():
    profiles = resolve_speakers(
        _transcript("SPEAKER_00"),
        {
            "SPEAKER_00": SpeakerCandidate(
                label="SPEAKER_00", name="기택", confidence=0.9, preferred_hue=40.0,
            )
        },
    )
    assert _hue(profiles["SPEAKER_00"].color) == pytest.approx(40.0, abs=1.0)


def test_resolve_speakers_keeps_a_named_candidate_with_no_hue_preference():
    # Identity and colour come from different steps: a manual mapping is
    # certain about the name and has no opinion on the colour.
    profiles = resolve_speakers(
        _transcript("SPEAKER_00", "SPEAKER_01"),
        {
            "SPEAKER_00": SpeakerCandidate(
                label="SPEAKER_00", name="기택", confidence=1.0, source="manual",
            )
        },
    )
    assert profiles["SPEAKER_00"].name == "기택"
    assert profiles["SPEAKER_00"].source == "manual"
    assert profiles["SPEAKER_00"].color


def test_resolve_speakers_drops_a_candidate_below_the_threshold():
    profiles = resolve_speakers(
        _transcript("SPEAKER_00"),
        {
            "SPEAKER_00": SpeakerCandidate(
                label="SPEAKER_00", name="기택", confidence=0.3, preferred_hue=40.0,
            )
        },
    )
    fallen_back = profiles["SPEAKER_00"]
    assert fallen_back.name is None          # an unsure name is a claim, not a label
    assert fallen_back.source == "unidentified"
    assert "기택" in fallen_back.note        # but the rejection is recorded


def test_resolve_speakers_honours_a_custom_threshold():
    candidates = {
        "SPEAKER_00": SpeakerCandidate(
            label="SPEAKER_00", name="기택", confidence=0.4,
        )
    }
    profiles = resolve_speakers(
        _transcript("SPEAKER_00"), candidates, min_confidence=0.3,
    )
    assert profiles["SPEAKER_00"].name == "기택"


def test_resolve_speakers_ignores_candidates_for_absent_speakers():
    profiles = resolve_speakers(
        _transcript("SPEAKER_00"),
        {
            "SPEAKER_99": SpeakerCandidate(
                label="SPEAKER_99", name="없는사람", confidence=1.0,
            )
        },
    )
    assert set(profiles) == {"SPEAKER_00"}


def test_resolve_speakers_covers_every_speaker_with_a_unique_colour():
    labels = [f"SPEAKER_{i:02d}" for i in range(9)]
    profiles = resolve_speakers(_transcript(*labels))

    assert set(profiles) == set(labels)
    # Nine speakers used to collide once the seven-colour palette wrapped.
    assert len({p.color for p in profiles.values()}) == len(labels)


def test_resolve_speakers_never_produces_the_base_colour():
    # White is what an unfilled word is drawn in; a speaker colour matching it
    # would make the karaoke fill invisible.
    labels = [f"SPEAKER_{i:02d}" for i in range(12)]
    profiles = resolve_speakers(_transcript(*labels))
    assert all(p.color != BASE_COLOUR for p in profiles.values())
