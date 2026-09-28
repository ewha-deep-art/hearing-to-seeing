"""Measuring text in the subtitle font, so words can be placed by coordinate.

The pop and wave effects (sync.py) move single words and characters, which ASS
can only do with a Dialogue event of their own positioned by `\\pos`/`\\move`.
Those coordinates have to land exactly where libass draws the same word in the
flowing line underneath, so the widths come from the real font file rather
than a character count.
"""

import os
from functools import lru_cache

from PIL import ImageFont

# Bundled so layout is identical on every machine. SemiBold is the face the web
# player renders every line in (watch.js SUBTITLE_FONT_NAME), and the family
# name the ASS styles ask for (converter/ass.py).
FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_PATH = os.path.join(FONTS_DIR, "Pretendard-SemiBold.otf")

# Measured at the font's own unit size, so `getmetrics()` returns the ascent
# and descent in font units.
_METRICS_SIZE = 2048


@lru_cache(maxsize=1)
def _em_per_ass_size() -> float:
    """How many em an ASS font size of 1 is.

    PIL sizes a font by its em square, but libass (like VSFilter) sizes it so
    that ascent + descent equal `\\fs` — for Pretendard that is ~0.84 em. Without
    this correction every measured width comes out ~19% too wide.
    """
    ascent, descent = ImageFont.truetype(FONT_PATH, _METRICS_SIZE).getmetrics()
    return _METRICS_SIZE / (ascent + descent)


@lru_cache(maxsize=32)
def _get_font(em_px: int) -> "ImageFont.FreeTypeFont":
    return ImageFont.truetype(FONT_PATH, em_px)


def text_width(text: str, size: float) -> float:
    """Advance width of `text` at ASS font size `size`, in script pixels.

    The font is loaded 4x oversize and scaled down, so rounding the pixel size
    to an integer costs a quarter pixel of accuracy rather than a whole one.
    """
    em_px = max(1, round(size * _em_per_ass_size() * 4))
    return _get_font(em_px).getlength(text) / 4
