from hearing_to_seeing.design.sync import (
    LOUD_SCALE,
    MAX_TRAILING_HOLD,
    WAVE_RISE_PX,
    WHISPER_SCALE,
    build_fill_text,
    build_loud_event_text,
    build_wave_events,
    build_whisper_event_text,
    karaoke_durations,
    seconds_to_centiseconds,
    spoken_end,
)
from hearing_to_seeing.schema import WordEntry

RED = "&H000000FF"


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


def test_fill_text_hides_each_word_while_it_is_said_then_colours_it():
    words = [WordEntry(text="안녕", start=1.0, end=1.5), WordEntry(text="하세요", start=1.6, end=2.0)]
    text = build_fill_text(words, RED)
    # Offsets are from the event start, i.e. the first word's start.
    assert "\\t(0,1,\\alpha&HFF&)\\t(500,501,\\alpha&H00&\\c&H000000FF)}안녕" in text
    assert "\\t(600,601,\\alpha&HFF&)\\t(1000,1001,\\alpha&H00&\\c&H000000FF)}하세요" in text
    assert "\\N" not in text


def test_fill_text_breaks_rows_at_break_index():
    words = [WordEntry(text=t, start=i, end=i + 0.5) for i, t in enumerate("abc")]
    text = build_fill_text(words, RED, break_at=2)
    assert "}a {" in text and "}b\\N{" in text


def test_loud_word_grows_then_settles():
    word = WordEntry(text="야", start=0.0, end=1.0)
    text = build_loud_event_text(word, RED, 960, 1000)
    assert text.startswith("{\\an5\\pos(960,1000)")
    assert f"\\t(0,400,\\fscx{LOUD_SCALE}\\fscy{LOUD_SCALE})\\t(400,1000,\\fscx100\\fscy100)" in text


def test_whisper_word_shrinks_then_settles():
    word = WordEntry(text="쉿", start=0.0, end=1.0)
    text = build_whisper_event_text(word, RED, 960, 1000)
    assert f"\\fscx{WHISPER_SCALE}\\fscy{WHISPER_SCALE}" in text


def test_wave_lifts_each_character_in_turn():
    word = WordEntry(text="ab", start=1.0, end=2.0)
    events = build_wave_events(word, RED, [("a", 100, 500), ("b", 120, 500)])
    assert len(events) == 4  # rise + fall per character
    (r0, _, rise_a), (_, f0_end, fall_a), (r1, _, _), _ = events
    assert r0 == 1.0 and r1 == 1.5              # characters take turns across the word
    assert f"\\move(100,500,100,{500 - WAVE_RISE_PX}," in rise_a
    assert f"\\move(100,{500 - WAVE_RISE_PX},100,500," in fall_a
    assert f0_end == 2.0                         # rests in place until the word is done


def test_wave_of_a_stretched_word_ends_at_the_cap():
    word = WordEntry(text="a", start=0.0, end=40.0)
    events = build_wave_events(word, RED, [("a", 0, 0)])
    assert events[-1][1] == MAX_TRAILING_HOLD
