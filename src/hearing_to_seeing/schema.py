from dataclasses import dataclass, field


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
    language: str | None = None
    # Label → {"color" (#RRGGBB), "name", "confidence", "reason"}, decided by
    # the pipeline (pipeline.assign_speaker_colors) so every reader — the ASS
    # converter via design/speaker.py, the web legend — shows the same ones.
    speaker_profiles: dict[str, dict] = field(default_factory=dict)

    def speakers(self) -> list[str]:
        return sorted({w.speaker for w in self.words})

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "speaker_profiles": self.speaker_profiles,
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
            language=data.get("language"),
            speaker_profiles=data.get("speaker_profiles") or {},
        )
