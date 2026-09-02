from hearing_to_seeing.converter.ass import (
    MAX_DURATION,
    MAX_GAP,
    MAX_WIDTH,
    MIN_WIDTH_FOR_BREAK,
    ROW_WIDTH,
    _row_break_index,
    _split_into_lines,
)
from hearing_to_seeing.schema import WordEntry


def _word(text, start, end, speaker="SPEAKER_00"):
    return WordEntry(text=text, start=start, end=end, speaker=speaker)


def test_split_into_lines_keeps_short_line_together():
    words = [_word("hi", 0.0, 0.3), _word("there", 0.3, 0.6)]
    assert _split_into_lines(words) == [words]


def test_split_into_lines_breaks_on_speaker_change():
    words = [
        _word("a", 0.0, 0.3, speaker="SPEAKER_00"),
        _word("b", 0.3, 0.6, speaker="SPEAKER_01"),
    ]
    lines = _split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def test_split_into_lines_breaks_on_long_silence():
    words = [_word("a", 0.0, 0.3), _word("b", 0.3 + MAX_GAP + 0.1, 1.0)]
    lines = _split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def test_split_into_lines_breaks_when_width_exceeds_max():
    words = [_word("a" * 50, 0.0, 1.0), _word("b" * 50, 1.0, 2.0)]
    lines = _split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def test_split_into_lines_breaks_on_long_duration():
    words = [_word("a", 0.0, 1.0), _word("b", 1.0, MAX_DURATION + 1.0)]
    lines = _split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def test_split_into_lines_breaks_after_terminator_once_line_is_long_enough():
    long_sentence = "a" * MIN_WIDTH_FOR_BREAK + "."
    words = [_word(long_sentence, 0.0, 1.0), _word("next", 1.0, 1.3)]
    lines = _split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def test_split_into_lines_does_not_break_on_terminator_when_line_is_short():
    words = [_word("a.", 0.0, 0.3), _word("next", 0.3, 0.6)]
    assert _split_into_lines(words) == [words]


def test_row_break_index_none_when_line_fits_one_row():
    words = [_word("hi", 0.0, 0.3), _word("there", 0.3, 0.6)]
    assert _row_break_index(words) is None


def test_row_break_index_picks_most_balanced_split():
    words = [_word("a" * 15, i, i + 1) for i in range(4)]
    assert sum(len(w.text) for w in words) + len(words) - 1 > ROW_WIDTH
    assert _row_break_index(words) == 2
