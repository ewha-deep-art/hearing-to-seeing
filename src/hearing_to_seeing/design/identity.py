"""Working out which character each diarization label belongs to.

WhisperX names speakers `SPEAKER_00`, `SPEAKER_01`, … — cluster numbers with
no identity attached. Deciding that `SPEAKER_00` is 기택 is a separate problem
from deciding what colour 기택's subtitles should be, and this module owns the
first one; `speaker.py` owns the second.

Only the manual path lives here so far: the user states the mapping and it is
taken at face value. It is not a placeholder for the automatic ones — it stays
as the correction the user reaches for when an automatic answer is wrong, and
a hand-written mapping is also the only ground truth an automatic one can be
measured against.
"""

from hearing_to_seeing.design.speaker import SpeakerCandidate, SpeakerStrategy
from hearing_to_seeing.schema import Transcript

# Nothing is inferred, so there is nothing to be unsure about.
MANUAL_CONFIDENCE = 1.0
SOURCE = "manual"


class SpeakerMapError(ValueError):
    pass


def parse_speaker_map(text: str) -> dict[str, str]:
    """Parses `SPEAKER_00=기택,SPEAKER_01=충숙` into a label → name mapping.

    Names may contain spaces; they are separated by commas, so a name
    containing a comma cannot be expressed. Errors are raised rather than
    skipped — a typo that silently drops one speaker is the failure mode worth
    avoiding, since the subtitles still come out looking fine without it.
    """
    mapping: dict[str, str] = {}
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, separator, name = entry.partition("=")
        label, name = label.strip(), name.strip()
        if not separator or not label or not name:
            raise SpeakerMapError(
                f"화자 매핑 형식이 잘못되었습니다 (SPEAKER_00=이름): {entry!r}"
            )
        if label in mapping:
            raise SpeakerMapError(f"화자 라벨이 중복되었습니다: {label}")
        mapping[label] = name
    return mapping


def manual_mapping(mapping: dict[str, str]) -> SpeakerStrategy:
    """Builds a strategy that reports `mapping` as certain.

    No colour preference is expressed: naming a speaker says nothing about
    which colour suits them, and that is the colour step's decision.
    """

    def resolve(transcript: Transcript, media_path: str) -> dict[str, SpeakerCandidate]:
        return {
            label: SpeakerCandidate(
                label=label,
                name=name,
                confidence=MANUAL_CONFIDENCE,
                source=SOURCE,
                note="사용자 지정",
            )
            for label, name in mapping.items()
        }

    return resolve
