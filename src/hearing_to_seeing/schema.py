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
class Character:
    """One character of the identified work, as the lookup step found them.

    Everything here is about the *character*, not the actor: the subtitle
    names 기택, and 기택's clothes and temperament are what a colour can be
    argued from. `actor` is kept only because it is the most reliable handle
    for finding the character again in another source.

    The `hue_*` fields are written later, by the colour step, and start out
    empty. They sit on the character rather than on the speaker because the
    answer belongs to the character: whichever diarization label turns out to
    be 기택, the reason 기택 is red does not change.
    """

    name: str
    aliases: list[str] = field(default_factory=list)
    actor: str | None = None
    # 주연/조연 — decides who keeps their wish when two hues collide.
    importance: str = "supporting"
    role: str | None = None
    personality: str | None = None
    relationships: str | None = None
    # How the character dresses, as the sources describe it. The primary
    # colour basis when it names a colour at all.
    appearance: str | None = None
    # Best-effort reference photo. Missing far more often than present.
    image_url: str | None = None

    hue: float | None = None
    # "costume" | "symbolic" | "achromatic" | None (not decided yet)
    hue_basis: str | None = None
    hue_evidence: str | None = None
    hue_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "Character":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs.setdefault("aliases", [])
        kwargs["aliases"] = list(kwargs["aliases"] or [])
        kwargs["importance"] = kwargs.get("importance") or "supporting"
        kwargs["hue_confidence"] = kwargs.get("hue_confidence") or 0.0
        return cls(**kwargs)


@dataclass
class WorkInfo:
    """The work the clip was identified as, and who appears in it.

    `MediaInfo` is what the user typed; this is what the lookup step made of
    it — a clean title where the clip title was "[4K] 기생충 명장면 모음
    #shorts", and the cast that title led to. Stored in the intermediate JSON
    so a second run over the same clip reuses it instead of searching again,
    and so the mapping step can read the cast without redoing the lookup.
    """

    title: str | None = None
    year: int | None = None
    scene_hint: str | None = None
    scene: str | None = None
    summary: str | None = None
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    characters: list[Character] = field(default_factory=list)

    def find(self, name: str) -> "Character | None":
        """The character called `name`, by name or alias, ignoring spaces."""
        wanted = _fold(name)
        for character in self.characters:
            if _fold(character.name) == wanted:
                return character
        for character in self.characters:
            if any(_fold(alias) == wanted for alias in character.aliases):
                return character
        return None

    def to_dict(self) -> dict:
        data = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "characters"
        }
        data["characters"] = [c.to_dict() for c in self.characters]
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "WorkInfo | None":
        if not data:
            return None
        return cls(
            title=data.get("title"),
            year=data.get("year"),
            scene_hint=data.get("scene_hint"),
            scene=data.get("scene"),
            summary=data.get("summary"),
            sources=list(data.get("sources") or []),
            confidence=data.get("confidence") or 0.0,
            characters=[Character.from_dict(c) for c in data.get("characters") or []],
        )


def _fold(name: str) -> str:
    return "".join(name.split()).casefold()


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
    # What the lookup step identified the clip as. None until it has run — a
    # transcript that never went through lookup is not the same as one where
    # lookup ran and found no characters.
    work: WorkInfo | None = None
    # The confirmed speaker → character mapping, label → name. The same fact
    # is inside `speaker_profiles`, but this block is the one meant to be read
    # and *edited* by a person: fix a name here, run `from-json` again, and the
    # correction is taken as certain. `speaker_profiles` stays the machine's
    # record of how each answer was reached.
    character_map: dict[str, str] = field(default_factory=dict)

    def speakers(self) -> list[str]:
        return sorted({w.speaker for w in self.words})

    def turns(self) -> list[tuple[str, float, float, str]]:
        """Consecutive same-speaker words joined into (speaker, start, end, text)."""
        turns: list[tuple[str, float, float, str]] = []
        for word in self.words:
            if turns and turns[-1][0] == word.speaker:
                speaker, start, _, text = turns[-1]
                turns[-1] = (speaker, start, word.end, f"{text} {word.text}")
            else:
                turns.append((word.speaker, word.start, word.end, word.text))
        return turns

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "media": self.media.to_dict(),
            "work": self.work.to_dict() if self.work else None,
            "character_map": dict(self.character_map),
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
            work=WorkInfo.from_dict(data.get("work")),
            character_map={
                str(label): str(name).strip()
                for label, name in (data.get("character_map") or {}).items()
                if str(name).strip()
            },
            speaker_profiles={
                label: SpeakerProfile.from_dict(label, profile)
                for label, profile in (data.get("speaker_profiles") or {}).items()
            },
        )

    @classmethod
    def from_segments(cls, data: dict) -> "Transcript":
        """Builds a Transcript from a segment-level export.

        The shape the project's input files take: `video_title`, `video_url`
        and a `segments` list of `{speaker, start, end, text}`. Segments carry
        one timestamp pair for a whole utterance, so each word is given an
        equal share of the segment's span — good enough for the sync effect
        to read left to right, not a real alignment. `run` gets true
        word timings from WhisperX; this is the `from-json` stand-in.
        """
        words: list[WordEntry] = []
        for segment in data.get("segments", []):
            tokens = str(segment.get("text", "")).split()
            if not tokens:
                continue
            start, end = float(segment["start"]), float(segment["end"])
            step = max(end - start, 0.0) / len(tokens)
            speaker = segment.get("speaker") or "SPEAKER_00"
            for i, token in enumerate(tokens):
                words.append(
                    WordEntry(
                        text=token,
                        start=round(start + i * step, 3),
                        end=round(start + (i + 1) * step, 3),
                        speaker=speaker,
                    )
                )
        return cls(
            words=words,
            language=data.get("language", "ko"),
            media=MediaInfo(
                title=data.get("video_title") or None,
                source_url=data.get("video_url") or None,
            ),
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
