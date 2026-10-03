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
import struct
import sys
from collections import Counter

from hearing_to_seeing.design.layout import FONTS_DIR
from hearing_to_seeing.schema import Transcript

# Only fonts whose family name starts with this are considered, so any other
# family dropped into the fonts folder never ends up in the weight table.
FONT_FAMILY = "NanumGothic"

# Light -> thin strokes ... ExtraBold -> heavy strokes.
WEIGHTS = ("Light", "Regular", "Bold", "ExtraBold")

# One font file for every speaker, in `layout.FONTS_DIR`. Its family name and
# bold flag are read from the file itself (no font library needed), because ASS
# finds a face by that internal name -- not by the file name -- and a wrong name
# silently falls back to a default font. Set to None to use the NanumGothic
# weights below instead.
FIXED_FONT_FILE: str | None = "KoPub Dotum Bold.ttf"

# Used only if FIXED_FONT_FILE can't be read; KoPubDotum ships each weight as a
# family of its own, so the weight is in the name and the Bold flag stays off.
FIXED_FONT_FALLBACK = ("KoPubDotum Bold", False)

# One weight label for every speaker. It only names the ASS styles (Fill-Bold,
# ...); the font is the fixed file above. Set to None, together with
# FIXED_FONT_FILE, to give each speaker a weight of their own again.
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


def read_font_face(path: str) -> tuple[str, bool]:
    """`(ASS font name, bold flag)` of a TrueType/OpenType file, from its name table.

    Reads the family (name ID 1) and style (ID 2), preferring the Windows English
    entries. The Bold flag is on only when the style says Bold, i.e. the face is
    the bold member of its family; a face whose family name already carries the
    weight ("KoPubDotum Bold", style Regular) is asked for with the flag off.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] == b"ttcf":  # a collection: take its first font
        base = struct.unpack(">I", data[12:16])[0]
    else:
        base = 0
    (num_tables,) = struct.unpack(">H", data[base + 4:base + 6])
    table = None
    for i in range(num_tables):
        tag, _, offset, _ = struct.unpack(">4sIII", data[base + 12 + 16 * i:base + 28 + 16 * i])
        if tag == b"name":
            table = offset
            break
    if table is None:
        raise ValueError("no name table")
    _, count, string_offset = struct.unpack(">HHH", data[table:table + 6])
    found: dict[int, tuple[int, str]] = {}
    for k in range(count):
        platform, _, lang, name_id, length, off = struct.unpack(
            ">6H", data[table + 6 + 12 * k:table + 18 + 12 * k]
        )
        if name_id not in (1, 2) or platform not in (1, 3):
            continue
        raw = data[table + string_offset + off:table + string_offset + off + length]
        text = raw.decode("utf-16-be" if platform == 3 else "mac_roman", "ignore").strip()
        rank = 0 if (platform == 3 and lang == 0x409) else 1 if platform == 3 else 2
        if text and (name_id not in found or rank < found[name_id][0]):
            found[name_id] = (rank, text)
    if 1 not in found:
        raise ValueError("no family name")
    style = found.get(2, (0, "Regular"))[1].lower()
    return found[1][1], "bold" in style


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
    if FIXED_FONT_FILE is not None:
        path = os.path.join(FONTS_DIR, FIXED_FONT_FILE)
        try:
            face = read_font_face(path)
        except (OSError, ValueError, struct.error) as exc:
            print(f"font: could not read {path} ({exc}); using the name "
                  f"{FIXED_FONT_FALLBACK[0]!r}", file=sys.stderr)
            face = FIXED_FONT_FALLBACK
        return {w: face for w in needed}
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
