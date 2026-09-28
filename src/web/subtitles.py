"""Transcript views for the player page: speaker colours and the script panel."""

from hearing_to_seeing.converter.ass import subtitle_events
from hearing_to_seeing.design.speaker import assign_speaker_colors
from hearing_to_seeing.schema import Transcript


def ass_to_hex(colour: str) -> str:
    """`&HAABBGGRR` (ASS) → `#RRGGBB` (CSS)."""
    value = colour.removeprefix("&H").rstrip("&").rjust(8, "0")
    bb, gg, rr = value[2:4], value[4:6], value[6:8]
    return f"#{rr}{gg}{bb}".upper()


def speakers(transcript: Transcript) -> list[dict]:
    """Legend entries in label order, named 화자 1, 화자 2, … for display."""
    colours = assign_speaker_colors(transcript.speakers())
    return [
        {"label": label, "name": f"화자 {i}", "color": ass_to_hex(colours[label])}
        for i, label in enumerate(transcript.speakers(), start=1)
    ]


def cues(transcript: Transcript) -> list[dict]:
    """One entry per on-screen subtitle, timed exactly as the ASS file is."""
    return [
        {
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": words[0].speaker,
            "text": " ".join(w.text for w in words),
        }
        for words, start, end, _ in subtitle_events(transcript)
    ]
