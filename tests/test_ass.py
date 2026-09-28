from hearing_to_seeing.converter.ass import (
    MAX_DURATION,
    MAX_GAP,
    MAX_WIDTH,
    ROW_WIDTH,
    _merge_unterminated_lines,
    _row_break_index,
    generate_ass,
    generate_plain_ass,
    split_into_lines,
)
from hearing_to_seeing.schema import Transcript, WordEntry


def _word(text, start, end, speaker="SPEAKER_00"):
    return WordEntry(text=text, start=start, end=end, speaker=speaker)


def testsplit_into_lines_keeps_short_line_together():
    words = [_word("hi", 0.0, 0.3), _word("there", 0.3, 0.6)]
    assert split_into_lines(words) == [words]


def testsplit_into_lines_breaks_on_speaker_change():
    words = [
        _word("a", 0.0, 0.3, speaker="SPEAKER_00"),
        _word("b", 0.3, 0.6, speaker="SPEAKER_01"),
    ]
    lines = split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def testsplit_into_lines_breaks_on_long_silence():
    words = [_word("a", 0.0, 0.3), _word("b", 0.3 + MAX_GAP + 0.1, 1.0)]
    lines = split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def testsplit_into_lines_breaks_when_width_exceeds_max():
    words = [_word("a" * 50, 0.0, 1.0), _word("b" * 50, 1.0, 2.0)]
    lines = split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def testsplit_into_lines_breaks_on_long_duration():
    words = [_word("a", 0.0, 1.0), _word("b", 1.0, MAX_DURATION + 1.0)]
    lines = split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def testsplit_into_lines_breaks_after_terminator_regardless_of_line_length():
    words = [_word("a.", 0.0, 0.3), _word("next", 0.3, 0.6)]
    lines = split_into_lines(words)
    assert lines == [[words[0]], [words[1]]]


def test_merge_unterminated_lines_reattaches_fragment_that_stays_within_limits():
    # A line without a terminator only exists because some other rule (speaker
    # change, silence, width, duration) forced it apart from what follows. This
    # exercises the merge in isolation, since split_into_lines can only ever
    # produce such a fragment when one of those limits was actually crossed --
    # here the two lines are constructed directly to check the merge condition
    # on its own, independent of how the split happened.
    a = [_word("a", 0.0, 0.3)]
    b = [_word("b.", 0.3, 0.6)]
    assert _merge_unterminated_lines([a, b]) == [a + b]


def test_merge_unterminated_lines_keeps_lines_separate_when_prev_ends_a_sentence():
    a = [_word("a.", 0.0, 0.3)]
    b = [_word("b.", 0.3, 0.6)]
    assert _merge_unterminated_lines([a, b]) == [a, b]


def test_merge_unterminated_lines_keeps_lines_separate_when_next_has_no_terminator():
    a = [_word("a", 0.0, 0.3)]
    b = [_word("b", 0.3, 0.6)]
    assert _merge_unterminated_lines([a, b]) == [a, b]


def test_merge_unterminated_lines_respects_speaker_change():
    a = [_word("a", 0.0, 0.3, speaker="SPEAKER_00")]
    b = [_word("b.", 0.3, 0.6, speaker="SPEAKER_01")]
    assert _merge_unterminated_lines([a, b]) == [a, b]


def test_merge_unterminated_lines_respects_max_gap():
    a = [_word("a", 0.0, 0.3)]
    b = [_word("b.", 0.3 + MAX_GAP + 0.1, 1.0)]
    assert _merge_unterminated_lines([a, b]) == [a, b]


def test_merge_unterminated_lines_respects_max_width():
    a = [_word("a" * 50, 0.0, 0.3)]
    b = [_word("b" * 50 + ".", 0.3, 0.6)]
    assert _merge_unterminated_lines([a, b]) == [a, b]


def test_merge_unterminated_lines_respects_max_duration():
    a = [_word("a", 0.0, 0.3)]
    b = [_word("b.", 0.3, MAX_DURATION + 1.0)]
    assert _merge_unterminated_lines([a, b]) == [a, b]


def testsplit_into_lines_does_not_split_a_sentence_that_never_breaks():
    # No speaker change, no silence, fits the width/duration budget, and no
    # terminator until the very end -- one group, nothing for the merge pass
    # to do.
    words = [_word("a", 0.0, 0.3), _word("b.", 0.3, 0.6)]
    assert split_into_lines(words) == [words]


def test_row_break_index_none_when_line_fits_one_row():
    words = [_word("hi", 0.0, 0.3), _word("there", 0.3, 0.6)]
    assert _row_break_index(words) is None


def test_row_break_index_picks_most_balanced_split():
    words = [_word("a" * 15, i, i + 1) for i in range(4)]
    assert sum(len(w.text) for w in words) + len(words) - 1 > ROW_WIDTH
    assert _row_break_index(words) == 2


def _dialogues(ass: str) -> list[str]:
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


def test_plain_ass_matches_kinetic_timing_without_effects():
    transcript = Transcript(words=[
        _word("hello", 0.0, 0.4, speaker="SPEAKER_00"),
        _word("world", 0.4, 0.8, speaker="SPEAKER_00"),
        _word("bye", 1.0, 1.3, speaker="SPEAKER_01"),
    ])
    kinetic = _dialogues(generate_ass(transcript))
    plain = _dialogues(generate_plain_ass(transcript))

    assert len(plain) == len(kinetic) == 2
    for k, p in zip(kinetic, plain):
        assert k.split(",", 9)[:9] == p.split(",", 9)[:9]
    assert plain[0].endswith(",hello world")
    assert "{" not in "".join(plain)
