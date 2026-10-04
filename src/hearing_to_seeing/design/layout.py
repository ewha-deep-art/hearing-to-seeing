"""Where the effects meet: the font, its size, and the spacing between words.

Each effect module owns one signal — sync.py (timing: fill wipe, wave motion),
volume.py (size), speaker.py (colour). Anything that depends on more than one
of them, but not on the ASS file format itself, lives here; the rest of the
assembly is converter/ass.py.
"""

import os

from hearing_to_seeing.design.volume import width_scale
from hearing_to_seeing.schema import WordEntry

# The one font every subtitle is drawn in, bundled so it looks the same on
# every machine; the preview renderer hands this folder to libass (render.py).
# ASS finds a face by a family name stored inside the file, not by the file
# name, and a wrong name silently falls back to some other font. This file
# lists itself as "KoPubDotum" and "KoPubDotum Bold" (style Bold), so the
# styles ask for the latter with the Bold flag on (`fc-scan` shows the names).
FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_FILE = "KoPub Dotum Bold.ttf"
FONT_NAME = "KoPubDotum Bold"
FONT_BOLD = True

# Every word is drawn at one size; loudness shows as a change of scale on top
# of it instead (volume.py). Keeping the size fixed stops the line from
# reflowing word by word.
# TODO: BASE_FONT_SIZE는 임시 값 — Netflix Timed Text Style Guide, BBC Subtitle
#       Guidelines, WCAG 등 접근성 표준 조사 후 기획서 §다음 논의 필요 사항에 따라 최종값 확정 필요.
BASE_FONT_SIZE = 64

# The space next to a whispered word is narrowed to that word's width scale
# (see space_scale). Multiply it further to tighten (<1.0) or loosen (>1.0)
# the gaps around whispers; 1.0 = same scale as the whisper itself.
WHISPER_SPACE_FACTOR = 1.0

# Every space in the Box / Fill / Pop lines is narrowed to this share of its
# normal width (1.0 = untouched). Applied on top of the whisper narrowing above.
SPACE_TIGHTNESS = 0.92


def space_scale(prev: WordEntry, nxt: WordEntry) -> int:
    """Width (percent) of the space between two words.

    A whispered word is narrowed (volume.width_scale), but a plain space stays
    at 100%, so the gaps around small words looked wider than the words
    themselves. The space therefore takes the smaller neighbour's scale (whisper
    next to normal -> the whisper's; two whispers -> the smaller one), times
    SPACE_TIGHTNESS.
    """
    pct = round(
        min(width_scale(prev.volume), width_scale(nxt.volume)) * 100
        * WHISPER_SPACE_FACTOR * SPACE_TIGHTNESS
    )
    return min(100, pct)
