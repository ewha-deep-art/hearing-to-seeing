import numpy as np

from hearing_to_seeing.design.volume import (
    FONT_SIZE_MAX,
    FONT_SIZE_MIN,
    calculate_rms,
    compute_font_size,
    normalize,
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


def test_compute_font_size_bounds():
    assert compute_font_size(0.0) == FONT_SIZE_MIN
    assert compute_font_size(1.0) == FONT_SIZE_MAX
    assert compute_font_size(0.5) == round((FONT_SIZE_MIN + FONT_SIZE_MAX) / 2)
