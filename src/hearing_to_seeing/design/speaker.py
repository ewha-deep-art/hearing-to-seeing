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


def speaker_colors(transcript: Transcript) -> dict[str, str]:
    """Label → ASS colour, for every speaker in the transcript."""
    profiles = transcript.speaker_profiles
    return {
        label: hex_to_ass((profiles.get(label) or {}).get("color"))
        for label in transcript.speakers()
    }
