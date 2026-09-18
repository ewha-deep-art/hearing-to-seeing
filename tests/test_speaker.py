from hearing_to_seeing.design.speaker import (
    _PALETTE,
    SpeakerCandidate,
    assign_from_preferences,
    assign_speaker_colors,
    fill_from_palette,
    resolve_speakers,
)
from hearing_to_seeing.schema import Transcript, WordEntry


def test_assign_speaker_colors_is_stable_by_sorted_label():
    colors = assign_speaker_colors(["SPEAKER_01", "SPEAKER_00"])
    assert colors == {
        "SPEAKER_00": _PALETTE[0],
        "SPEAKER_01": _PALETTE[1],
    }


def test_assign_speaker_colors_wraps_around_palette():
    speakers = [f"SPEAKER_{i:02d}" for i in range(len(_PALETTE) + 1)]
    colors = assign_speaker_colors(speakers)
    # One more speaker than colours in the palette — the extra one wraps to the start.
    assert colors[speakers[0]] == colors[speakers[-1]] == _PALETTE[0]


def test_assign_speaker_colors_empty_input():
    assert assign_speaker_colors([]) == {}


def _transcript(*labels: str) -> Transcript:
    return Transcript(
        words=[
            WordEntry(text="w", start=float(i), end=float(i) + 0.5, speaker=label)
            for i, label in enumerate(labels)
        ]
    )


# --- assign_from_preferences -------------------------------------------------

def test_assign_from_preferences_gives_each_speaker_its_best_colour():
    assigned = assign_from_preferences({
        "SPEAKER_00": {_PALETTE[0]: 0.9, _PALETTE[4]: 0.2},
        "SPEAKER_01": {_PALETTE[4]: 0.8, _PALETTE[1]: 0.1},
    })
    assert assigned == {"SPEAKER_00": _PALETTE[0], "SPEAKER_01": _PALETTE[4]}


def test_assign_from_preferences_breaks_a_collision_by_score():
    # Both want orange; the stronger preference takes it and the other drops
    # to its own next best rather than sharing.
    assigned = assign_from_preferences({
        "SPEAKER_00": {_PALETTE[0]: 0.9, _PALETTE[4]: 0.2},
        "SPEAKER_01": {_PALETTE[0]: 0.8, _PALETTE[3]: 0.6},
    })
    assert assigned == {"SPEAKER_00": _PALETTE[0], "SPEAKER_01": _PALETTE[3]}


def test_assign_from_preferences_never_repeats_a_colour():
    preferences = {f"SPEAKER_{i:02d}": {_PALETTE[0]: 0.9} for i in range(3)}
    assigned = assign_from_preferences(preferences)
    # Only the winner is placed; the others expressed no other preference.
    assert assigned == {"SPEAKER_00": _PALETTE[0]}


def test_assign_from_preferences_ignores_non_positive_scores():
    assert assign_from_preferences({"SPEAKER_00": {_PALETTE[0]: 0.0}}) == {}


def test_assign_from_preferences_ignores_colours_outside_the_palette():
    assigned = assign_from_preferences({
        "SPEAKER_00": {"&H00123456": 0.9, _PALETTE[2]: 0.5},
    })
    assert assigned == {"SPEAKER_00": _PALETTE[2]}


def test_assign_from_preferences_is_deterministic_on_ties():
    preferences = {
        "SPEAKER_01": {_PALETTE[0]: 0.5},
        "SPEAKER_00": {_PALETTE[0]: 0.5},
    }
    # Equal scores: the label decides, so the result does not depend on dict order.
    assert assign_from_preferences(preferences) == {"SPEAKER_00": _PALETTE[0]}


# --- fill_from_palette -------------------------------------------------------

def test_fill_from_palette_skips_colours_already_taken():
    assigned = fill_from_palette(["SPEAKER_00"], taken=[_PALETTE[0]])
    assert assigned == {"SPEAKER_00": _PALETTE[1]}


def test_fill_from_palette_cycles_once_every_colour_is_used():
    labels = [f"SPEAKER_{i:02d}" for i in range(len(_PALETTE) + 2)]
    assigned = fill_from_palette(labels)
    assert assigned[labels[0]] == assigned[labels[len(_PALETTE)]] == _PALETTE[0]
    assert assigned[labels[1]] == assigned[labels[len(_PALETTE) + 1]] == _PALETTE[1]


# --- resolve_speakers --------------------------------------------------------

def test_resolve_speakers_without_candidates_matches_palette_order():
    profiles = resolve_speakers(_transcript("SPEAKER_01", "SPEAKER_00"))
    assert profiles["SPEAKER_00"].color == _PALETTE[0]
    assert profiles["SPEAKER_01"].color == _PALETTE[1]
    assert profiles["SPEAKER_00"].name is None
    assert profiles["SPEAKER_00"].source == "palette"


def test_resolve_speakers_keeps_a_confident_candidate():
    profiles = resolve_speakers(
        _transcript("SPEAKER_00", "SPEAKER_01"),
        {
            "SPEAKER_00": SpeakerCandidate(
                label="SPEAKER_00", name="기택", confidence=0.9,
                source="llm", note="갈색 점퍼",
                preference={_PALETTE[5]: 0.9},
            )
        },
    )
    kept = profiles["SPEAKER_00"]
    assert kept.color == _PALETTE[5]
    assert (kept.name, kept.source, kept.note) == ("기택", "llm", "갈색 점퍼")
    assert kept.confidence == 0.9
    # The speaker that fell back must not be handed the colour just taken.
    assert profiles["SPEAKER_01"].color != _PALETTE[5]


def test_resolve_speakers_drops_a_candidate_below_the_threshold():
    profiles = resolve_speakers(
        _transcript("SPEAKER_00"),
        {
            "SPEAKER_00": SpeakerCandidate(
                label="SPEAKER_00", name="기택", confidence=0.3,
                preference={_PALETTE[5]: 0.9},
            )
        },
    )
    fallen_back = profiles["SPEAKER_00"]
    assert fallen_back.color == _PALETTE[0]      # palette order, not the preference
    assert fallen_back.name is None              # an unsure name is a claim, not a label
    assert fallen_back.source == "palette"
    assert "기택" in fallen_back.note            # but the rejection is recorded


def test_resolve_speakers_honours_a_custom_threshold():
    candidates = {
        "SPEAKER_00": SpeakerCandidate(
            label="SPEAKER_00", name="기택", confidence=0.4,
            preference={_PALETTE[5]: 0.9},
        )
    }
    profiles = resolve_speakers(
        _transcript("SPEAKER_00"), candidates, min_confidence=0.3,
    )
    assert profiles["SPEAKER_00"].color == _PALETTE[5]
    assert profiles["SPEAKER_00"].name == "기택"


def test_resolve_speakers_ignores_candidates_for_absent_speakers():
    profiles = resolve_speakers(
        _transcript("SPEAKER_00"),
        {
            "SPEAKER_99": SpeakerCandidate(
                label="SPEAKER_99", name="없는사람", confidence=1.0,
                preference={_PALETTE[3]: 1.0},
            )
        },
    )
    assert set(profiles) == {"SPEAKER_00"}
    assert profiles["SPEAKER_00"].color == _PALETTE[0]


def test_resolve_speakers_covers_every_speaker():
    labels = [f"SPEAKER_{i:02d}" for i in range(4)]
    profiles = resolve_speakers(_transcript(*labels))
    assert set(profiles) == set(labels)
    assert all(p.color for p in profiles.values())


def test_resolve_speakers_keeps_a_named_candidate_with_no_colour_preference():
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
    named = profiles["SPEAKER_00"]
    assert named.name == "기택"
    assert named.source == "manual"
    assert named.color == _PALETTE[0]        # colour still comes from the palette
    assert profiles["SPEAKER_01"].name is None
