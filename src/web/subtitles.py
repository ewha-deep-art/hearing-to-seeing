"""Transcript views for the player page: speaker colours, the script panel and
the subtitle files."""

import json

from hearing_to_seeing.converter.ass import generate_ass, generate_plain_ass, subtitle_events
from hearing_to_seeing.design.speaker import speaker_colors
from hearing_to_seeing.schema import Transcript


def ass_to_hex(colour: str) -> str:
    """`&HAABBGGRR` (ASS) → `#RRGGBB` (CSS)."""
    value = colour.removeprefix("&H").rstrip("&").rjust(8, "0")
    bb, gg, rr = value[2:4], value[4:6], value[6:8]
    return f"#{rr}{gg}{bb}".upper()


def speakers(transcript: Transcript) -> list[dict]:
    """Legend entries, most speaking time first, named by the character the
    pipeline matched (speaker_profiles) or 화자 1, 화자 2, … in label order."""
    colours = speaker_colors(transcript)
    talk: dict[str, float] = {}
    for w in transcript.words:
        talk[w.speaker] = talk.get(w.speaker, 0.0) + w.duration
    legend = [
        {
            "label": label,
            "name": (transcript.speaker_profiles.get(label) or {}).get("name") or f"화자 {i}",
            "color": ass_to_hex(colours[label]),
        }
        for i, label in enumerate(transcript.speakers(), start=1)
    ]
    return sorted(legend, key=lambda s: -talk[s["label"]])


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


def export(transcript: Transcript, kind: str) -> tuple[str, str, str]:
    """The transcript as a file: (content, filename suffix, media type)."""
    if kind == "ass":
        return generate_ass(transcript), ".ass", "text/plain; charset=utf-8"
    if kind == "plain":
        return generate_plain_ass(transcript), ".plain.ass", "text/plain; charset=utf-8"
    if kind == "json":
        content = json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2)
        return content, ".json", "application/json"
    raise KeyError(kind)
