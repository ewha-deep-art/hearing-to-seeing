from hearing_to_seeing.design.sync import (
    MAX_TRAILING_HOLD,
    karaoke_durations,
    lift_tags,
    seconds_to_centiseconds,
)
from hearing_to_seeing.schema import WordEntry


def test_seconds_to_centiseconds_rounds_and_floors_at_one():
    assert seconds_to_centiseconds(1.5) == 150
    assert seconds_to_centiseconds(0.0) == 1  # never zero — \k0 would not advance the clock


def test_karaoke_durations_spans_to_next_word_start():
    words = [
        WordEntry(text="a", start=0.0, end=0.3),
        WordEntry(text="b", start=0.5, end=0.9),
        WordEntry(text="c", start=1.0, end=1.05),
    ]
    durations = karaoke_durations(words)
    assert durations[0] == seconds_to_centiseconds(0.5 - 0.0)  # holds through the gap after "a"
    assert durations[1] == seconds_to_centiseconds(1.0 - 0.5)  # holds through the gap after "b"
    assert durations[2] == seconds_to_centiseconds(0.05)       # last word: its own duration


def test_karaoke_durations_caps_trailing_word_hold():
    words = [WordEntry(text="solo", start=0.0, end=10.0)]
    assert karaoke_durations(words) == [seconds_to_centiseconds(MAX_TRAILING_HOLD)]


def test_lift_tags_chains_offsets_from_durations():
    tags = lift_tags([70, 50])
    assert tags[0].startswith("\\t(0,")
    assert "\\t(700," in tags[1]  # second word's lift starts where the first word's \kf (70cs = 700ms) ends


def test_lift_tags_scales_down_for_short_words():
    # A 10ms word is far shorter than LIFT_RISE + LIFT_FALL (120ms + 200ms), so the
    # rise/fall must shrink to fit instead of overrunning into the next word's lift.
    (tag,) = lift_tags([1])
    assert tag == "\\t(0,3,\\fscy110\\shad4)\\t(3,10,\\fscy100\\shad1)"
