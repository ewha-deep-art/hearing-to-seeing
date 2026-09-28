from hearing_to_seeing.converter.ass import (
    MAX_DURATION,
    MAX_GAP,
    MAX_WIDTH,
    ROW_WIDTH,
    MARGIN_H,
    MARGIN_V,
    PLAY_RES,
    _compute_word_layout,
    _merge_unterminated_lines,
    _row_break_index,
    generate_ass,
    generate_plain_ass,
    split_into_lines,
)
from hearing_to_seeing.design.layout import text_width
from hearing_to_seeing.design.volume import BASE_FONT_SIZE
from hearing_to_seeing.schema import Transcript, WordEntry


def _word(text, start, end, speaker="SPEAKER_00", volume=0.5):
    return WordEntry(text=text, start=start, end=end, speaker=speaker, volume=volume)


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
    kinetic = [line for line in _dialogues(generate_ass(transcript)) if ",Box," in line]
    plain = _dialogues(generate_plain_ass(transcript))

    assert len(plain) == len(kinetic) == 2
    for k, p in zip(kinetic, plain):
        k_fields, p_fields = k.split(",", 9), p.split(",", 9)
        assert k_fields[:3] + k_fields[4:9] == p_fields[:3] + p_fields[4:9]  # all but the style
    assert plain[0].endswith(",hello world")
    assert "{" not in "".join(plain)


def test_layout_centres_a_single_row_above_the_bottom_margin():
    words = [_word("안녕", 0.0, 0.3), _word("하세요", 0.3, 0.6)]
    layout = _compute_word_layout(words, None)
    width = text_width("안녕 하세요", BASE_FONT_SIZE)
    centre = PLAY_RES[0] / 2
    assert abs(layout[0][0] - (centre - width / 2)) < 0.01
    assert layout[1][0] > layout[0][0] + text_width("안녕", BASE_FONT_SIZE)  # past the space
    assert layout[0][1] == layout[1][1] == PLAY_RES[1] - MARGIN_V - BASE_FONT_SIZE / 2


def test_layout_stacks_two_rows_each_centred_on_its_own_width():
    words = [_word("a" * 15, i, i + 1) for i in range(4)]
    layout = _compute_word_layout(words, 2)
    top, bottom = layout[0][1], layout[2][1]
    assert bottom - top == BASE_FONT_SIZE
    assert layout[0][0] == layout[2][0]  # identical rows start at the same x
    assert MARGIN_H < layout[0][0]


def test_generate_ass_draws_box_fill_and_pop_layers():
    transcript = Transcript(words=[
        _word("크게", 0.0, 0.4, volume=1.0),
        _word("작게", 0.4, 0.8, volume=0.0),
        _word("보통.", 0.8, 1.2, volume=0.5),
    ])
    lines = _dialogues(generate_ass(transcript))
    styles = [line.split(",")[3] for line in lines]
    assert styles[:2] == ["Box", "Fill"]
    pops = [line for line in lines if ",Pop," in line]
    assert pops[0].startswith("Dialogue: 2,") and "\\fscx150" in pops[0]  # loud
    assert "\\fscx65" in pops[1]                                            # whisper
    assert len(pops) == 2 + 2 * len("보통.")                                 # wave: 2 per character
