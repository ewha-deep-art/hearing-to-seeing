import os

from PIL import ImageFont

from hearing_to_seeing.design.layout import FONT_BOLD, FONT_FILE, FONT_NAME, FONTS_DIR, space_scale
from hearing_to_seeing.design.volume import WHISPER_SCALE_MIN
from hearing_to_seeing.schema import WordEntry


def test_bundled_font_is_the_one_the_styles_ask_for():
    # ASS finds the face by the name inside the file, so a mismatch would
    # silently render in some other font.
    assert os.listdir(FONTS_DIR) == [FONT_FILE]
    family, style = ImageFont.truetype(os.path.join(FONTS_DIR, FONT_FILE), 10).getname()
    assert FONT_NAME.startswith(family)
    assert FONT_BOLD == ("Bold" in style)


def test_space_next_to_a_whisper_narrows_with_it():
    normal, whisper = WordEntry("a", 0.0, 0.3, volume=0.5), WordEntry("b", 0.3, 0.6, volume=0.0)
    assert space_scale(normal, normal) > space_scale(normal, whisper)
    assert space_scale(normal, whisper) == space_scale(whisper, normal) <= WHISPER_SCALE_MIN
