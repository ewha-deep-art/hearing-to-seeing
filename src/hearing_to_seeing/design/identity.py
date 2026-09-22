"""Working out which character each diarization label belongs to.

WhisperX names speakers `SPEAKER_00`, `SPEAKER_01`, … — cluster numbers with
no identity attached. Deciding that `SPEAKER_00` is 기택 is a separate problem
from deciding what colour 기택's subtitles should be, and this module owns the
first one; `speaker.py` owns the second.

Two certain paths live here, plus the way of stacking them with the inferred
one:

  * **manual** — the user states the mapping on the command line;
  * **stored** — a mapping read back from a transcript's `character_map`,
    which is either what an earlier run confirmed or what the user corrected
    in the file by hand;
  * `layered()` — the first strategy with an answer for a label wins, so the
    order in which they are stacked is the order of trust.

Neither certain path is a placeholder for the automatic one (`inference.py`).
They are the correction the user reaches for when an automatic answer is
wrong, and a hand-written mapping is also the only ground truth an automatic
one can be measured against.
"""

from collections.abc import Mapping

from hearing_to_seeing.design.speaker import SpeakerCandidate, SpeakerStrategy
from hearing_to_seeing.schema import Transcript

# Nothing is inferred, so there is nothing to be unsure about.
MANUAL_CONFIDENCE = 1.0
SOURCE = "manual"
STORED_SOURCE = "stored"


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


def stored_mapping(mapping: Mapping[str, str]) -> SpeakerStrategy:
    """A strategy that replays a transcript's confirmed `character_map`.

    Confirmed once is confirmed: the mapping was either accepted by an earlier
    run or edited into the file by the user, and in both cases redoing the
    inference would only risk changing an answer that was already right.
    """

    def resolve(transcript: Transcript, media_path: str) -> dict[str, SpeakerCandidate]:
        return {
            label: SpeakerCandidate(
                label=label,
                name=name,
                confidence=MANUAL_CONFIDENCE,
                source=STORED_SOURCE,
                note="이전 실행에서 확정된 매핑",
            )
            for label, name in mapping.items()
        }

    return resolve


def layered(*strategies: SpeakerStrategy | None) -> SpeakerStrategy | None:
    """Stacks strategies so the first to name a label decides it.

    Later strategies only fill labels the earlier ones left out. A manual
    mapping therefore sits on top of a stored one, and both on top of the
    inferred one — which is also why the inferred strategy is only asked at
    all when something is left to ask about.
    """
    stack = [s for s in strategies if s is not None]
    if not stack:
        return None
    if len(stack) == 1:
        return stack[0]

    def resolve(transcript: Transcript, media_path: str) -> dict[str, SpeakerCandidate]:
        candidates: dict[str, SpeakerCandidate] = {}
        remaining = set(transcript.speakers())
        for strategy in stack:
            if not remaining:
                break  # nothing left to ask — and asking may cost a model call
            for label, candidate in strategy(transcript, media_path).items():
                candidates.setdefault(label, candidate)
            remaining -= {label for label, c in candidates.items() if c.name}
        return candidates

    return resolve
