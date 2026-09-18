"""Deciding which colour each speaker's subtitles are drawn in.

Three things live here, in the order the pipeline uses them:

  * **the palette** — the colours a subtitle may be drawn in at all;
  * **assignment** — turning per-speaker colour *preferences* into one colour
    each, and settling the case where two speakers want the same one;
  * **resolution** — running a strategy that produces those preferences,
    keeping only what it was confident about, and filling the rest from the
    palette.

A strategy never picks a colour. It says what it would prefer and how sure it
is, and assignment decides — so however a strategy arrives at its answer, it
cannot hand two speakers the same colour, or one that cannot be read against
video.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from hearing_to_seeing.schema import SpeakerProfile, Transcript

# TODO: 색상 배정 방식 최종 미확정 — 단순 팔레트 순서 배정(현재) vs RAG 기반
#       (인물 정보 → LLM 색상 결정). 기획서 §다음 논의 필요 사항 참조.
# Wong colorblind-friendly palette in ASS BGR hex format (&HAABBGGRR&).
# ASS stores colours byte-reversed relative to RGB, so #E69F00 → &H00009FE6.
_PALETTE = [
    "&H00009FE6",  # Orange        #E69F00
    "&H00E9B456",  # Sky Blue      #56B4E9
    "&H00739E00",  # Bluish Green  #009E73
    "&H0042E4F0",  # Yellow        #F0E442
    "&H00B27200",  # Blue          #0072B2
    "&H00005ED5",  # Vermillion    #D55E00
    "&H00A779CC",  # Reddish Purple #CC79A7
]

# Colour every word starts out in, before the karaoke fill reaches it.
BASE_COLOUR = "&H00FFFFFF"  # white

# How sure a strategy has to be before its answer is used at all. A wrong
# colour is worse than an arbitrary one: a viewer who cannot hear the audio
# has no way to catch it, so an unconvincing answer is dropped rather than
# shown. Deliberately high to start with.
# TODO: 임계값 0.7은 경험값 — 실제 전략을 붙인 뒤 정확도를 측정해 재조정 필요.
MIN_CONFIDENCE = 0.7


@dataclass
class SpeakerCandidate:
    """What a strategy has to say about one speaker.

    `preference` scores palette colours from 0 to 1 — how well each would suit
    this speaker — instead of naming a colour outright, which is what leaves
    assignment free to break ties. Scores of zero or less are treated as "no
    opinion" and fall through to the palette.
    """

    label: str
    name: str | None = None
    confidence: float = 0.0
    source: str = ""
    note: str | None = None
    preference: dict[str, float] = field(default_factory=dict)


# Takes the transcript and the media path, returns one candidate per speaker it
# has something to say about — a strategy may leave speakers out entirely.
SpeakerStrategy = Callable[[Transcript, str], dict[str, SpeakerCandidate]]


def assign_from_preferences(
    preferences: Mapping[str, Mapping[str, float]],
    palette: Sequence[str] = _PALETTE,
) -> dict[str, str]:
    """Gives each speaker the colour it prefers, as far as that is possible.

    Preferences collide — two characters in orange shirts both score orange
    highest and only one can have it. The highest score takes the colour
    outright and the loser drops to its own next best, which keeps both
    assignments meaningful without ever letting two speakers share a colour.

    Speakers whose preferences are all used up, or who expressed none, are
    left out for the caller to fill.
    """
    order = {colour: i for i, colour in enumerate(palette)}
    # Sorting on the whole tuple keeps ties broken the same way every run:
    # by score, then speaker label, then palette order.
    ranked = sorted(
        (-score, label, order[colour], colour)
        for label, scores in preferences.items()
        for colour, score in scores.items()
        if colour in order and score > 0
    )

    assigned: dict[str, str] = {}
    used: set[str] = set()
    for _, label, _, colour in ranked:
        if label in assigned or colour in used:
            continue
        assigned[label] = colour
        used.add(colour)
    return assigned


def fill_from_palette(
    labels: Iterable[str],
    taken: Iterable[str] = (),
    palette: Sequence[str] = _PALETTE,
) -> dict[str, str]:
    """Hands out the colours nothing else is using, in palette order.

    Once every colour is in use the palette cycles, so an eighth speaker
    shares with the first. Two speakers in one scene sharing a colour is the
    one failure seven colours cannot avoid; cycling at least makes which pair
    it is predictable.
    """
    if not palette:
        return {}

    used = set(taken)
    free = [colour for colour in palette if colour not in used]
    assigned: dict[str, str] = {}
    # TODO: 화자가 알파벳순(SPEAKER_00, SPEAKER_01, …)으로 정렬되어 색상이
    #       WhisperX 라벨 순서에 의존함. 대화 첫 등장 순서 기반 배정 방식 검토 필요.
    for i, label in enumerate(sorted(labels)):
        if i < len(free):
            assigned[label] = free[i]
        else:
            assigned[label] = palette[(i - len(free)) % len(palette)]
    return assigned


def assign_speaker_colors(speakers: list[str]) -> dict[str, str]:
    """Palette-order assignment with nothing else to go on — the fallback."""
    return fill_from_palette(set(speakers))


def resolve_speakers(
    transcript: Transcript,
    candidates: Mapping[str, SpeakerCandidate] | None = None,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    palette: Sequence[str] = _PALETTE,
) -> dict[str, SpeakerProfile]:
    """Produces one profile per speaker in `transcript`.

    Confidence is judged per speaker, not for the run as a whole: a strategy
    usually knows the leads and not the bit parts, and taking what it is sure
    of beats discarding the lot. Rejected speakers fall back to the palette,
    which picks from the colours the accepted ones did not take.

    With no candidates at all this is exactly the old palette-order behaviour.
    """
    labels = transcript.speakers()
    candidates = candidates or {}

    # Identity and colour arrive from different steps: a manual mapping knows
    # exactly who a speaker is and has no opinion on colour, so confidence
    # alone decides whether a candidate is believed. A candidate with no
    # preference keeps its name and takes its colour from the palette.
    accepted = {
        label: candidate
        for label, candidate in candidates.items()
        if label in labels and candidate.confidence >= min_confidence
    }

    colours = assign_from_preferences(
        {label: candidate.preference for label, candidate in accepted.items()},
        palette,
    )
    colours.update(
        fill_from_palette(
            [label for label in labels if label not in colours],
            taken=colours.values(),
            palette=palette,
        )
    )

    profiles: dict[str, SpeakerProfile] = {}
    for label in labels:
        candidate = accepted.get(label)
        if candidate is not None:
            profiles[label] = SpeakerProfile(
                label=label,
                color=colours.get(label),
                name=candidate.name,
                confidence=candidate.confidence,
                source=candidate.source or "strategy",
                note=candidate.note,
            )
            continue

        # Not believed, or nothing was proposed. A name we are unsure of is a
        # claim rather than a label, so it is not kept — but the discarded one
        # is recorded, since a rejection is the thing worth seeing when the
        # threshold turns out to be set wrong.
        rejected = candidates.get(label)
        note = None
        if rejected is not None:
            note = (
                f"discarded {rejected.name or '(unnamed)'} "
                f"at confidence {rejected.confidence:.2f}"
            )
        profiles[label] = SpeakerProfile(label=label, color=colours.get(label), note=note)

    return profiles
