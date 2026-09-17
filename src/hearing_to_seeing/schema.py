from dataclasses import dataclass, field, replace

# Character-level exports sometimes omit the space at an utterance boundary,
# leaving two separate utterances joined into a single token that spans the
# silence between them. A gap this long between characters is a word break.
MAX_CHAR_GAP = 0.5

# A YouTube description can run to thousands of characters of links and
# boilerplate. The part worth keeping is the top, where a cast or guest list
# usually sits, so the field is capped rather than stored whole.
MAX_DESCRIPTION_CHARS = 2000


@dataclass
class MediaInfo:
    """Where the media came from, and what is known about it.

    None of these fields affect the current three effects — they exist for the
    speaker-colour step, which cannot look up who appears in a video without
    first knowing which video it is. Keeping them on the transcript puts them
    in the intermediate JSON too, so a later step can use them without having
    to fetch the source again.
    """

    title: str | None = None
    source_url: str | None = None
    uploader: str | None = None
    description: str | None = None

    def merge(self, other: "MediaInfo") -> "MediaInfo":
        """Returns a copy with `other`'s values filling in whatever is unset here.

        Used when both an explicit option and a stored transcript carry media
        info: the caller's value wins field by field, rather than a half-filled
        `--title` wiping out a URL and channel that were already known.
        """
        filled = {
            name: getattr(other, name)
            for name in ("title", "source_url", "uploader", "description")
            if getattr(self, name) is None
        }
        return replace(self, **filled)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source_url": self.source_url,
            "uploader": self.uploader,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "MediaInfo":
        data = data or {}
        description = data.get("description")
        if description:
            description = description[:MAX_DESCRIPTION_CHARS]
        return cls(
            title=data.get("title"),
            source_url=data.get("source_url"),
            uploader=data.get("uploader"),
            description=description,
        )


@dataclass
class WordEntry:
    text: str
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    volume: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


@dataclass
class Transcript:
    words: list[WordEntry] = field(default_factory=list)
    language: str = "ko"
    media: MediaInfo = field(default_factory=MediaInfo)

    def speakers(self) -> list[str]:
        return sorted({w.speaker for w in self.words})

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "media": self.media.to_dict(),
            "words": [
                {
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "speaker": w.speaker,
                    "volume": w.volume,
                }
                for w in self.words
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        words = [
            WordEntry(
                text=w["text"],
                start=w["start"],
                end=w["end"],
                speaker=w.get("speaker", "SPEAKER_00"),
                volume=w.get("volume", 0.0),
            )
            for w in data.get("words", [])
        ]
        return cls(
            words=words,
            language=data.get("language", "ko"),
            media=MediaInfo.from_dict(data.get("media")),
        )

    @classmethod
    def from_char_timestamps(cls, data: dict) -> "Transcript":
        """Builds a Transcript from a character-level timestamp export.

        Some upstream tools emit one entry per character rather than per word.
        Whitespace entries are the word separators — and their durations carry
        the inter-word silence, so they are dropped rather than merged. Neither
        speaker nor volume is present in this format; both keep their defaults
        until the diarization and volume steps annotate them.
        """
        words: list[WordEntry] = []
        buffer: list[dict] = []

        def flush() -> None:
            if buffer:
                words.append(
                    WordEntry(
                        text="".join(c["char"] for c in buffer),
                        start=round(buffer[0]["start"], 3),
                        end=round(buffer[-1]["end"], 3),
                    )
                )
                buffer.clear()

        for char in data.get("characters", []):
            if char.get("char", "").isspace():
                flush()
                continue
            if buffer and char["start"] - buffer[-1]["end"] > MAX_CHAR_GAP:
                flush()
            buffer.append(char)
        flush()

        return cls(words=words, language=data.get("language", "ko"))
