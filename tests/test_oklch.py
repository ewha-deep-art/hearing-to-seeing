import pytest

from hearing_to_seeing.design.oklch import (
    from_ass,
    hue_to_ass,
    in_gamut,
    max_chroma,
    oklch_to_rgb,
    rgb_to_oklch,
    to_ass,
)


def test_to_ass_reverses_the_byte_order():
    # ASS stores &HAABBGGRR, so #E69F00 becomes &H00009FE6.
    assert to_ass(0xE6, 0x9F, 0x00) == "&H00009FE6"


def test_from_ass_reverses_to_ass():
    assert from_ass("&H00009FE6") == (0xE6, 0x9F, 0x00)


def test_from_ass_tolerates_the_trailing_ampersand():
    assert from_ass("&H00009FE6&") == (0xE6, 0x9F, 0x00)


@pytest.mark.parametrize("rgb", [(230, 159, 0), (86, 180, 233), (0, 114, 178)])
def test_oklch_round_trip(rgb):
    restored = oklch_to_rgb(*rgb_to_oklch(*rgb))
    # Allowed a unit of rounding per channel on the way back through 8 bits.
    assert all(abs(a - b) <= 1 for a, b in zip(restored, rgb))


def test_max_chroma_is_inside_the_gamut_and_its_excess_is_not():
    lightness, hue = 0.75, 120.0
    chroma = max_chroma(lightness, hue)
    assert in_gamut(lightness, chroma, hue)
    assert not in_gamut(lightness, chroma + 0.01, hue)


def test_max_chroma_varies_by_hue():
    # The sRGB gamut is not a cylinder: yellow reaches much further than blue,
    # which is why chroma is expressed as a ratio of this rather than fixed.
    assert max_chroma(0.75, 110) > max_chroma(0.75, 260)


def test_fixed_lightness_holds_across_hues():
    # The property the whole colour model rests on: every hue comes out about
    # equally bright, so legibility does not depend on which colour was picked.
    lightnesses = [
        rgb_to_oklch(*from_ass(hue_to_ass(hue, 0.75, 0.95)))[0]
        for hue in range(0, 360, 30)
    ]
    assert all(abs(value - 0.75) < 0.01 for value in lightnesses)


def test_hue_to_ass_keeps_the_hue_it_was_given():
    for hue in (0, 45, 200, 310):
        produced = rgb_to_oklch(*from_ass(hue_to_ass(hue, 0.75, 0.95)))[2]
        assert abs(produced - hue) < 1.0


def test_lower_chroma_ratio_produces_a_duller_colour():
    vivid = rgb_to_oklch(*from_ass(hue_to_ass(200, 0.75, 0.95)))[1]
    muted = rgb_to_oklch(*from_ass(hue_to_ass(200, 0.75, 0.32)))[1]
    assert muted < vivid


def test_hue_wraps_around_the_circle():
    assert hue_to_ass(370, 0.75, 0.95) == hue_to_ass(10, 0.75, 0.95)
