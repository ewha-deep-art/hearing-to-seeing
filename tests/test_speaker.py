from hearing_to_seeing.design.speaker import _PALETTE, assign_speaker_colors


def test_assign_speaker_colors_is_stable_by_sorted_label():
    colors = assign_speaker_colors(["SPEAKER_01", "SPEAKER_00"])
    assert colors == {
        "SPEAKER_00": _PALETTE[0],
        "SPEAKER_01": _PALETTE[1],
    }


def test_assign_speaker_colors_wraps_around_palette():
    speakers = [f"SPEAKER_{i:02d}" for i in range(len(_PALETTE) + 1)]
    colors = assign_speaker_colors(speakers)
    # One more speaker than colours in the palette — the extra one wraps to the start.
    assert colors[speakers[0]] == colors[speakers[-1]] == _PALETTE[0]


def test_assign_speaker_colors_empty_input():
    assert assign_speaker_colors([]) == {}
