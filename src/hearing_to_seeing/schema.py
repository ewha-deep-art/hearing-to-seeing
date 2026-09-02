from dataclasses import dataclass, field

# Character-level exports sometimes omit the space at an utterance boundary,
# leaving two separate utterances joined into a single token that spans the
# silence between them. A gap this long between characters is a word break.
MAX_CHAR_GAP = 0.5


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

    def speakers(self) -> list[str]:
        return sorted({w.speaker for w in self.words})

    def to_dict(self) -> dict:
        return {
            "language": self.language,
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
        return cls(words=words, language=data.get("language", "ko"))

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
