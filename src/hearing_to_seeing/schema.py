from dataclasses import dataclass, field, fields, replace

# Character-level exports sometimes omit the space at an utterance boundary,
# leaving two separate utterances joined into a single token that spans the
# silence between them. A gap this long between characters is a word break.
MAX_CHAR_GAP = 0.5


@dataclass
class MediaInfo:
    """Which work the media holds, as told to us by the user.

    Neither field affects the current three effects — they exist for the
    speaker-colour step, which cannot look up who appears in a video without
    first knowing which video it is. Keeping them on the transcript puts them
    in the intermediate JSON too, so that step can read them later without the
    user having to supply them a second time.
    """

    title: str | None = None
    source_url: str | None = None

    def merge(self, other: "MediaInfo") -> "MediaInfo":
        """Returns a copy with `other`'s values filling in whatever is unset here.

        Used when both an explicit option and a stored transcript carry media
        info: the caller's value wins field by field, rather than a lone
        `--title` wiping out a URL that was already known.
        """
        filled = {
            f.name: getattr(other, f.name)
            for f in fields(self)
            if getattr(self, f.name) is None
        }
        return replace(self, **filled)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "MediaInfo":
        data = data or {}
        return cls(**{f.name: data.get(f.name) for f in fields(cls)})


@dataclass
class SpeakerProfile:
    """One speaker's resolved identity and subtitle colour.

    Written by the speaker-colour step and read by the ASS converter, which
    does no colour reasoning of its own. `confidence`, `source` and `note`
    record where the answer came from: a colour that turns out to be wrong is
    otherwise indistinguishable from one that was merely arbitrary, and the
    two call for different fixes.
    """

    label: str
    color: str | None = None
    name: str | None = None
    confidence: float = 0.0
    source: str = "palette"
    note: str | None = None

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, label: str, data: dict) -> "SpeakerProfile":
        # `label` comes from the key it was stored under, which cannot
        # disagree with itself the way a duplicated field could.
        return cls(
            label=label,
            color=data.get("color"),
            name=data.get("name"),
            confidence=data.get("confidence") or 0.0,
            source=data.get("source") or "palette",
            note=data.get("note"),
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
    # Keyed by speaker label. Per-speaker facts belong here rather than copied
    # onto every word that speaker says.
    speaker_profiles: dict[str, SpeakerProfile] = field(default_factory=dict)

    def speakers(self) -> list[str]:
        return sorted({w.speaker for w in self.words})

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "media": self.media.to_dict(),
            "speaker_profiles": {
                label: profile.to_dict()
                for label, profile in self.speaker_profiles.items()
            },
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
            speaker_profiles={
                label: SpeakerProfile.from_dict(label, profile)
                for label, profile in (data.get("speaker_profiles") or {}).items()
            },
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
