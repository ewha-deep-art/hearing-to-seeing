import numpy as np

from hearing_to_seeing.design.volume import (
    LOUD_SCALE_MAX,
    LOUD_SCALE_MIN,
    LOUD_THRESHOLD,
    QUIET_THRESHOLD,
    WHISPER_SCALE_MAX,
    WHISPER_SCALE_MIN,
    WHISPER_V_CENTER,
    calculate_rms,
    classify_volume,
    loud_scale,
    loud_tags,
    narrow_tag,
    normalize,
    whisper_lift,
    whisper_phases,
    whisper_return_tags,
    whisper_scale,
    whisper_tags,
    width_scale,
)


def test_calculate_rms_of_known_signal():
    # RMS of [3, 4] is sqrt((9 + 16) / 2) = sqrt(12.5).
    assert calculate_rms(np.array([3.0, 4.0])) == np.sqrt(12.5)


def test_calculate_rms_empty_or_none_is_zero():
    assert calculate_rms(np.array([])) == 0.0
    assert calculate_rms(None) == 0.0


def test_normalize_empty_list():
    assert normalize([]) == []


def test_normalize_scales_to_unit_range():
    assert normalize([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]


def test_normalize_uniform_values_avoids_division_by_zero():
    # min == max would otherwise divide by zero; every value maps to 0.0 instead.
    assert normalize([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]


def test_classify_volume_bands():
    assert classify_volume(1.0) == "loud"
    assert classify_volume(LOUD_THRESHOLD) == "loud"
    assert classify_volume(0.5) == "normal"
    assert classify_volume(QUIET_THRESHOLD) == "whisper"
    assert classify_volume(0.0) == "whisper"


def test_louder_words_grow_bigger_and_quieter_ones_shrink_more():
    assert LOUD_SCALE_MIN == loud_scale(LOUD_THRESHOLD) < loud_scale(1.0) == LOUD_SCALE_MAX
    assert WHISPER_SCALE_MIN == whisper_scale(0.0) < whisper_scale(QUIET_THRESHOLD) == WHISPER_SCALE_MAX


def test_no_word_gives_up_width_in_the_line():
    assert width_scale(0.0) == width_scale(0.5) == width_scale(1.0) == 1.0
    assert narrow_tag(0.0) == narrow_tag(0.5) == narrow_tag(1.0) == ""


def test_loud_word_grows_holds_and_returns_within_its_spoken_time():
    tags = loud_tags(1.0, 1.0)
    assert tags == (
        f"\\t(0,200,\\fscx{LOUD_SCALE_MAX}\\fscy{LOUD_SCALE_MAX})"
        "\\t(800,1000,\\fscx100\\fscy100)"
    )


def test_whisper_shrinks_holds_and_returns_within_its_spoken_time():
    assert whisper_phases(1.0) == (200, 800, 200)
    assert whisper_tags(0.0, 1.0) == f"\\t(0,200,\\fscx{WHISPER_SCALE_MIN}\\fscy{WHISPER_SCALE_MIN})"
    assert whisper_return_tags(0.0, 1.0) == (
        f"\\fscx{WHISPER_SCALE_MIN}\\fscy{WHISPER_SCALE_MIN}\\t(0,200,\\fscx100\\fscy100)"
    )


def test_whisper_is_lifted_to_mid_row():
    assert whisper_lift(0.0, 100) == round(100 * (1 - WHISPER_SCALE_MIN / 100) * WHISPER_V_CENTER)
