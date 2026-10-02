"""Which font face each speaker's subtitles are drawn in.

Same shape as speaker.py's colours: `speaker_weights(transcript)` gives every
speaker a weight label, and `resolve_faces()` turns a label into the face libass
is asked for. The weights are four steps of one family (NanumGothic) so a
speaker change reads as "a different voice" without changing glyph shapes or
line widths much.

ASS finds a face by the family name stored inside the font file, and a bold face
is asked for with the Bold flag of the style. `FACES` holds those names for each
weight of NanumGothic; the font files in `layout.FONTS_DIR` are only checked by
file name (no font parsing) so a weight whose file is missing falls back to the
closest one that is there.
"""

import os
import sys
from collections import Counter

from hearing_to_seeing.design.layout import FONTS_DIR
from hearing_to_seeing.schema import Transcript

# Only fonts whose family name starts with this are considered, so any other
# family dropped into the fonts folder never ends up in the weight table.
FONT_FAMILY = "NanumGothic"

# Light -> thin strokes ... ExtraBold -> heavy strokes.
WEIGHTS = ("Light", "Regular", "Bold", "ExtraBold")

# One weight for every speaker (NanumGothicBold). Set to None to give each
# speaker a weight of their own again (speaker_weights below).
FIXED_WEIGHT: str | None = "Bold"

# The weight of a speaker who has none assigned.
DEFAULT_WEIGHT = FIXED_WEIGHT or "Regular"

# Order in which speakers receive a weight when none is given: the most talkative
# speaker gets the baseline, the next the most different one, and so on.
# TODO: 음높이·굵기 분석이 생기면 이 자동 배정 대신 목소리 특성에서 매핑한다.
AUTO_ORDER = ("Regular", "Bold", "Light", "ExtraBold")

# Fixed choices win over the automatic order, e.g. {"SPEAKER_00": "Bold"}.
SPEAKER_WEIGHT_OVERRIDES: dict[str, str] = {}

# weight -> (ASS font name, bold flag). NanumGothic's Bold is the bold face of the
# "NanumGothic" family; Light and ExtraBold are families of their own.
FACES = {
    "Light": ("NanumGothic Light", False),
    "Regular": ("NanumGothic", False),
    "Bold": ("NanumGothic", True),
    "ExtraBold": ("NanumGothic ExtraBold", False),
}


def _weight_of_file(filename: str) -> str:
    """Weight step a NanumGothic font file holds, from its file name."""
    stem = filename.lower().replace("-", "").replace("_", "").replace(" ", "")
    if "extrabold" in stem:
        return "ExtraBold"
    if "light" in stem:
        return "Light"
    if "bold" in stem:
        return "Bold"
    return "Regular"


def available_weights(fonts_dir: str = FONTS_DIR) -> set[str]:
    """Weights that have a NanumGothic font file in `fonts_dir`."""
    try:
        names = os.listdir(fonts_dir)
    except OSError:
        return set()
    return {
        _weight_of_file(n)
        for n in names
        if n.lower().startswith(FONT_FAMILY.lower())
        and n.lower().endswith((".ttf", ".otf"))
    }


def resolve_faces(needed=WEIGHTS) -> dict[str, tuple[str, bool]]:
    """A face for each weight in `needed`; one with no file uses the closest present.

    Only the weights actually in use need a font file, so a folder holding just
    NanumGothicBold is fine when every speaker is Bold.
    """
    present = available_weights()
    if not present:  # folder unreadable or empty: trust the names, say nothing
        return {w: FACES[w] for w in needed}
    faces = {}
    for weight in needed:
        i = WEIGHTS.index(weight)
        if weight in present:
            faces[weight] = FACES[weight]
            continue
        nearest = min(present, key=lambda w: abs(WEIGHTS.index(w) - i))
        print(f"font: no {weight} file in {FONTS_DIR}, using {nearest}", file=sys.stderr)
        faces[weight] = FACES[nearest]
    return faces


def speaker_weights(
    transcript: Transcript, overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """speaker -> weight label, by talkativeness unless overridden.

    With FIXED_WEIGHT set, every speaker gets that weight instead.
    """
    counts = Counter(w.speaker for w in transcript.words)
    if FIXED_WEIGHT is not None:
        return {speaker: FIXED_WEIGHT for speaker in counts}
    fixed = {**SPEAKER_WEIGHT_OVERRIDES, **(overrides or {})}
    result: dict[str, str] = {}
    free = [s for s, _ in counts.most_common() if s not in fixed]
    for i, speaker in enumerate(free):
        result[speaker] = AUTO_ORDER[i % len(AUTO_ORDER)]
    for speaker, weight in fixed.items():
        if weight not in WEIGHTS:
            raise ValueError(f"unknown weight {weight!r}; use one of {WEIGHTS}")
        result[speaker] = weight
    return result
