import os

from PIL import ImageFont

from hearing_to_seeing.design.layout import (
    FONT_BOLD, FONT_FILE, FONT_NAME, FONTS_DIR, SPACE_TIGHTNESS, space_scale,
)
from hearing_to_seeing.schema import WordEntry


def test_bundled_font_is_the_one_the_styles_ask_for():
    # ASS finds the face by the name inside the file, so a mismatch would
    # silently render in some other font.
    assert os.listdir(FONTS_DIR) == [FONT_FILE]
    family, style = ImageFont.truetype(os.path.join(FONTS_DIR, FONT_FILE), 10).getname()
    assert FONT_NAME.startswith(family)
    assert FONT_BOLD == ("Bold" in style)


def test_every_space_is_tightened_the_same_even_next_to_a_whisper():
    # A whisper keeps its full width in the line (volume.width_scale), so the
    # spaces around it are no narrower than any other.
    normal, whisper = WordEntry("a", 0.0, 0.3, volume=0.5), WordEntry("b", 0.3, 0.6, volume=0.0)
    expected = round(100 * SPACE_TIGHTNESS)
    assert space_scale(normal, normal) == space_scale(normal, whisper) == space_scale(whisper, normal) == expected
