"""Features of a clip that may explain how well the pipeline does on it.

Shared by export_clips.py (on the pipeline's transcripts, to draw a sample)
and by_feature.py (on the gold transcripts, so pipeline errors do not leak into
the features). Metrics: README.md 6.
"""

import contextlib
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hearing_to_seeing.design.volume import classify_volume, relative_volumes  # noqa: E402
from hearing_to_seeing.schema import Transcript  # noqa: E402

ANIMATION = "애니메이션"
MINOR_WORDS = 10  # a speaker with fewer words than this is a bit part
REVIEWED = "영상 검토"  # speaker_profiles reason when the video review named the speaker

PUNCT = re.compile(r"[^\w\s]")


def _words(data: dict) -> list[dict]:
    words = [w for w in data.get("words", []) if PUNCT.sub("", w["text"]).strip()]
    return sorted(words, key=lambda w: w["start"])


def joined(data: dict) -> str:
    """The transcript as plain text, for pairing transcripts by similarity."""
    return " ".join(PUNCT.sub("", w["text"]).strip() for w in _words(data))


def _union_seconds(words: list[dict]) -> float:
    total, end = 0.0, float("-inf")
    for w in words:  # sorted by start
        if w["end"] <= end:
            continue
        total += w["end"] - max(w["start"], end)
        end = w["end"]
    return total


def _overlapped(words: list[dict]) -> int:
    """Words that overlap in time a word of another speaker."""
    hit = set()
    for i, w in enumerate(words):
        for j in range(i + 1, len(words)):
            v = words[j]
            if v["start"] >= w["end"]:
                break
            if v["speaker"] != w["speaker"] and v["end"] > w["start"]:
                hit.update((i, j))
    return len(hit)


def features(data: dict, meta: dict | None = None) -> dict:
    """Video features of one transcript dict (schema.Transcript.to_dict).

    `meta` (a clip's meta.json) gives the genre and the video's duration;
    without it the duration is where the last word ends.
    """
    meta = meta or {}
    words = _words(data)
    n = len(words)
    duration = meta.get("duration") or (max((w["end"] for w in words), default=0.0))
    minutes = duration / 60 if duration else 0.0

    per_speaker: dict[str, int] = {}
    for w in words:
        per_speaker[w["speaker"]] = per_speaker.get(w["speaker"], 0) + 1
    minor = [s for s, c in per_speaker.items() if c < MINOR_WORDS]

    speech = _union_seconds(words)
    chars = sum(len(PUNCT.sub("", w["text"]).replace(" ", "")) for w in words)
    turns = sum(1 for a, b in zip(words, words[1:]) if a["speaker"] != b["speaker"])

    with contextlib.redirect_stdout(io.StringIO()):  # its note on words without volume
        relative = relative_volumes(Transcript.from_dict({"words": words}).words)
    classes = [classify_volume(w.volume) for w in relative]

    profiles = data.get("speaker_profiles") or {}
    confidences = [p["confidence"] for p in profiles.values() if p.get("confidence") is not None]

    def share(k: int) -> float:
        return round(k / n, 4) if n else 0.0

    return {
        "genre": meta.get("genre"),
        "animation": meta.get("genre") == ANIMATION,
        "duration": round(duration, 1),
        "words": n,
        "speakers": len(per_speaker),
        "minor_speakers": len(minor),
        "minor_word_share": share(sum(per_speaker[s] for s in minor)),
        "speech_ratio": round(speech / duration, 4) if duration else 0.0,
        "chars_per_sec": round(chars / speech, 2) if speech else 0.0,
        "turns_per_min": round(turns / minutes, 2) if minutes else 0.0,
        "overlap_share": share(_overlapped(words)),
        "whisper_share": share(classes.count("whisper")),
        "loud_share": share(classes.count("loud")),
        "reviewed": any(REVIEWED in (p.get("reason") or "") for p in profiles.values()),
        "character_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
    }


def duration_bin(seconds: float) -> str:
    return "≤5분" if seconds <= 300 else "5–10분" if seconds <= 600 else ">10분"


def speakers_bin(count: int) -> str:
    return "≤3명" if count <= 3 else "4–6명" if count <= 6 else "≥7명"
