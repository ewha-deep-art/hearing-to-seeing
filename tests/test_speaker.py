import pytest
from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.design.speaker import assign_speaker_colors, annotate_speakers


def test_assign_single_speaker():
    colors = assign_speaker_colors(["SPEAKER_00"])
    assert len(colors) == 1
    assert colors["SPEAKER_00"].startswith("&H")


def test_assign_multiple_speakers_unique_colors():
    speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    colors = assign_speaker_colors(speakers)
    assert len(colors) == 3
    assert len(set(colors.values())) == 3


def test_assign_colors_is_deterministic():
    speakers = ["SPEAKER_01", "SPEAKER_00"]
    c1 = assign_speaker_colors(speakers)
    c2 = assign_speaker_colors(speakers)
    assert c1 == c2


def test_assign_colors_sorted_order():
    colors = assign_speaker_colors(["SPEAKER_01", "SPEAKER_00"])
    # SPEAKER_00 should get the first palette color (sorted alphabetically)
    colors_sorted = assign_speaker_colors(["SPEAKER_00", "SPEAKER_01"])
    assert colors == colors_sorted


def test_assign_duplicate_speakers():
    colors = assign_speaker_colors(["SPEAKER_00", "SPEAKER_00"])
    assert len(colors) == 1


def test_palette_wraps_beyond_seven():
    speakers = [f"SPEAKER_{i:02d}" for i in range(10)]
    colors = assign_speaker_colors(speakers)
    assert len(colors) == 10
    # palette wraps, so first and 8th have same color
    assert colors["SPEAKER_00"] == colors["SPEAKER_07"]


def test_annotate_speakers_returns_color_map():
    t = Transcript(words=[
        WordEntry("hi", 0.0, 0.3, speaker="SPEAKER_00"),
        WordEntry("hello", 0.4, 0.8, speaker="SPEAKER_01"),
    ])
    color_map = annotate_speakers(t)
    assert "SPEAKER_00" in color_map
    assert "SPEAKER_01" in color_map
