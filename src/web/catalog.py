"""What the catalogue knows about a clip besides its subtitles: the YouTube id
of a link, the genre shelf it goes on, and the text search matches."""

import re
import sys

from hearing_to_seeing.knowledge import llm, prompts
from web.db import GENRES

OTHER = GENRES[-1]
GENRE_SCHEMA = {**prompts.GENRE_SCHEMA, "properties": {"genre": {"type": "string", "enum": list(GENRES)}}}
DESCRIPTION_CHARS = 500

YOUTUBE_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/))([A-Za-z0-9_-]{11})"
)


def youtube_id(url: str) -> str | None:
    match = YOUTUBE_ID_RE.search(url or "")
    return match.group(1) if match else None


def genre_of(info: dict) -> str:
    """The genre shelf for a clip, from what YouTube says about it (`media.youtube_info`)."""
    if not llm.configured():
        return OTHER
    prompt = prompts.GENRE_PROMPT.format(
        title=info.get("title") or "",
        channel=info.get("channel") or "",
        categories=", ".join(info.get("categories") or []),
        tags=", ".join((info.get("tags") or [])[:20]),
        description=(info.get("description") or "")[:DESCRIPTION_CHARS],
    )
    try:
        genre = llm.ask(prompts.GENRE_SYSTEM, prompt, GENRE_SCHEMA).get("genre")
    except Exception as exc:  # a missing shelf never fails the clip
        print(f"… 장르 분류 실패: {exc}", file=sys.stderr, flush=True)
        return OTHER
    return genre if genre in GENRES else OTHER


def search_text(title: str | None, channel: str | None, transcript: dict) -> str:
    """Title, channel, character names and dialogue, lower-cased for `ilike`."""
    names = [p.get("name") for p in (transcript.get("speaker_profiles") or {}).values()]
    words = [w["text"] for w in transcript.get("words", [])]
    parts = [title, channel, *names, " ".join(words)]
    return "\n".join(p for p in parts if p).lower()
