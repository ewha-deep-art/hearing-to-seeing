from hearing_to_seeing.design.sync import (
    MAX_TRAILING_HOLD,
    WAVE_MIN_MOTION_MS,
    WAVE_RISE_PX,
    fall_move,
    hand_over_ms,
    hide_tag,
    karaoke_durations,
    rise_move,
    seconds_to_centiseconds,
    spoken_end,
    wave_chars,
    wave_hand_over_ms,
    wipe_tags,
)
from hearing_to_seeing.schema import WordEntry


def test_seconds_to_centiseconds_rounds_and_floors_at_one():
    assert seconds_to_centiseconds(1.5) == 150
    assert seconds_to_centiseconds(0.0) == 1


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


def test_spoken_end_caps_stretched_words():
    assert spoken_end(WordEntry(text="a", start=1.0, end=1.4)) == 1.4
    assert spoken_end(WordEntry(text="a", start=1.0, end=30.0)) == 1.0 + MAX_TRAILING_HOLD


def test_wipe_waits_for_its_start_then_sweeps():
    assert wipe_tags(1.0, 1.0, 1.5) == ("", "\\kf50")
    assert wipe_tags(1.0, 1.3, 1.5) == ("\\k30", "\\kf20")


def test_fill_hand_over_is_when_the_word_is_said():
    word = WordEntry(text="안녕", start=1.6, end=2.0)
    assert hand_over_ms(word, 1.0) == 600
    assert hide_tag(600) == "\\t(600,601,\\alpha&HFF&)"


def test_wave_characters_take_turns_across_the_word():
    word = WordEntry(text="ab", start=1.0, end=2.0)
    a, b = wave_chars(word)
    assert a.rise_start == 1.0 and b.rise_start == 1.5
    assert a.wipe_start >= a.rise_end              # wiped while falling, not rising
    assert b.wipe_end == 2.0                       # the last share ends with the word
    assert wave_hand_over_ms(word, 1.0) == [0, 499]  # each character leaves Fill as it rises


def test_wave_of_a_short_word_overlaps_into_a_ripple():
    word = WordEntry(text="abcd", start=0.0, end=0.2)
    chars = wave_chars(word)
    assert chars[0].rise_ms + chars[0].fall_ms == WAVE_MIN_MOTION_MS
    assert chars[1].rise_start < chars[0].rise_end + chars[0].fall_ms / 1000


def test_wave_of_a_stretched_word_ends_at_the_cap():
    word = WordEntry(text="a", start=0.0, end=40.0)
    assert wave_chars(word)[-1].wipe_end == MAX_TRAILING_HOLD


def test_wave_lifts_from_the_anchor_and_lands_back_on_it():
    assert rise_move((960, 1020), 100) == f"\\move(960,1020,960,{1020 - WAVE_RISE_PX},0,100)"
    assert fall_move((960, 1020), 200) == f"\\move(960,{1020 - WAVE_RISE_PX},960,1020,0,200)"
