"""Speaker colours, as the ASS file draws them.

Who each speaker is and what colour they get is decided by the pipeline
(`pipeline.assign_speaker_colors`) and stored in the transcript's
`speaker_profiles` as `#RRGGBB`. This module only reads those values and
turns them into ASS colours, so the ASS converter and the web legend show
the same ones.
"""

import re

from hearing_to_seeing.schema import Transcript

# Colour every word starts out in, before it is spoken — and the colour of a
# speaker the profiles say nothing about. The pipeline gives every speaker a
# colour with chroma, so none collides with it.
BASE_COLOUR = "&H00FFFFFF"  # white


def hex_to_ass(value: str | None) -> str:
    """`#RRGGBB` → ASS `&H00BBGGRR`; BASE_COLOUR if it is not a colour.

    An ASS colour (`&H…`, as older transcripts stored it) is passed through.
    """
    value = (value or "").strip()
    if value.upper().startswith("&H"):
        return value
    match = re.fullmatch(r"#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})", value)
    if not match:
        return BASE_COLOUR
    r, g, b = (digits.upper() for digits in match.groups())
    return f"&H00{b}{g}{r}"


# Used only when the transcript carries no colour for a speaker (e.g. a JSON run
# locally without the pipeline's RAG / assign_speaker_colors step). Without it
# every colour falls back to BASE_COLOUR = white = the colour a word is wiped
# *from*, so the karaoke fill goes white -> white and cannot be seen.
FALLBACK_PALETTE = [
    "#FFD60A",  # yellow
    "#4FC3F7",  # sky blue
    "#FF8A80",  # coral
    "#A5D6A7",  # green
    "#CE93D8",  # violet
    "#FFB74D",  # orange
    "#80DEEA",  # cyan
]


def speaker_colors(transcript: Transcript) -> dict[str, str]:
    """Label -> ASS colour, for every speaker in the transcript.

    A speaker whose profile has no usable colour (missing, malformed, or plain
    white) gets one from FALLBACK_PALETTE so the fill stays visible.
    """
    profiles = transcript.speaker_profiles or {}
    colours, missing = {}, []
    for label in transcript.speakers():
        colour = hex_to_ass((profiles.get(label) or {}).get("color"))
        if colour.upper().replace("&", "") == BASE_COLOUR.replace("&", ""):
            missing.append(label)
        else:
            colours[label] = colour
    for i, label in enumerate(missing):
        colours[label] = hex_to_ass(FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)])
    if missing:
        print(f"[speaker] no colour in speaker_profiles for {missing}; using fallback palette")
    return colours
