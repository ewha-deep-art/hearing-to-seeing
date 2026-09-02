import numpy as np
import pytest
from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.design.volume import (
    calculate_rms,
    normalize,
    compute_font_size,
    annotate_volumes,
    FONT_SIZE_MIN,
    FONT_SIZE_MAX,
)


def test_calculate_rms_constant_signal():
    audio = np.array([2.0, 2.0, 2.0, 2.0])
    assert calculate_rms(audio) == pytest.approx(2.0)


def test_calculate_rms_zero_signal():
    audio = np.zeros(100)
    assert calculate_rms(audio) == pytest.approx(0.0)


def test_calculate_rms_empty():
    assert calculate_rms(np.array([])) == 0.0


def test_calculate_rms_none():
    assert calculate_rms(None) == 0.0


def test_calculate_rms_known_value():
    audio = np.array([1.0, -1.0, 1.0, -1.0])
    assert calculate_rms(audio) == pytest.approx(1.0)


def test_normalize_basic():
    values = [0.0, 1.0, 2.0, 4.0]
    result = normalize(values)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)
    assert result[2] == pytest.approx(0.5)


def test_normalize_all_same():
    result = normalize([5.0, 5.0, 5.0])
    assert result == [0.0, 0.0, 0.0]


def test_normalize_empty():
    assert normalize([]) == []


def test_compute_font_size_min():
    assert compute_font_size(0.0) == FONT_SIZE_MIN


def test_compute_font_size_max():
    assert compute_font_size(1.0) == FONT_SIZE_MAX


def test_compute_font_size_midpoint():
    mid = compute_font_size(0.5)
    assert FONT_SIZE_MIN < mid < FONT_SIZE_MAX


def test_compute_font_size_custom_range():
    assert compute_font_size(0.0, min_size=10, max_size=20) == 10
    assert compute_font_size(1.0, min_size=10, max_size=20) == 20
    assert compute_font_size(0.5, min_size=10, max_size=20) == 15


def test_annotate_volumes_sets_values():
    sample_rate = 100
    # 1 second of audio: first half loud, second half quiet
    audio_data = np.concatenate([np.full(50, 100.0), np.full(50, 10.0)])
    transcript = Transcript(words=[
        WordEntry("loud", start=0.0, end=0.5),
        WordEntry("quiet", start=0.5, end=1.0),
    ])
    annotate_volumes(transcript, sample_rate, audio_data)

    assert transcript.words[0].volume == pytest.approx(1.0)
    assert transcript.words[1].volume == pytest.approx(0.0)


def test_annotate_volumes_all_same_rms():
    sample_rate = 100
    audio_data = np.full(100, 50.0)
    transcript = Transcript(words=[
        WordEntry("a", 0.0, 0.5),
        WordEntry("b", 0.5, 1.0),
    ])
    annotate_volumes(transcript, sample_rate, audio_data)
    assert transcript.words[0].volume == 0.0
    assert transcript.words[1].volume == 0.0
