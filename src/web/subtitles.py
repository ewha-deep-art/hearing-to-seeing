"""Transcript views for the player page: speaker colours, the script panel and
the subtitle files."""

from hearing_to_seeing.converter.ass import (
    Effects, generate_ass, generate_plain_ass, subtitle_events,
)
from hearing_to_seeing.design.speaker import speaker_colors, speaker_names
from hearing_to_seeing.schema import Transcript


def ass_to_hex(colour: str) -> str:
    """`&HAABBGGRR` (ASS) → `#RRGGBB` (CSS)."""
    value = colour.removeprefix("&H").rstrip("&").rjust(8, "0")
    bb, gg, rr = value[2:4], value[4:6], value[6:8]
    return f"#{rr}{gg}{bb}".upper()


def speakers(transcript: Transcript) -> list[dict]:
    """Legend entries, most speaking time first, named as the subtitles name
    them (the character the pipeline matched, else 여자 1 / 남자 1 / 인물 1)."""
    colours = speaker_colors(transcript)
    names = speaker_names(transcript)
    talk: dict[str, float] = {}
    for w in transcript.words:
        talk[w.speaker] = talk.get(w.speaker, 0.0) + w.duration
    legend = [
        {
            "label": label,
            "name": names[label],
            "color": ass_to_hex(colours[label]),
        }
        for label in transcript.speakers()
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


def export(
    transcript: Transcript, kind: str, effects: Effects = Effects(),
) -> tuple[str, str, str]:
    """The transcript as a subtitle file: (content, filename suffix, media type).

    `effects` are the viewer's choices (the plain file only takes `names`).
    """
    if kind == "ass":
        return generate_ass(transcript, effects), ".ass", "text/plain; charset=utf-8"
    if kind == "plain":
        return generate_plain_ass(transcript, effects=effects), ".plain.ass", "text/plain; charset=utf-8"
    raise KeyError(kind)
