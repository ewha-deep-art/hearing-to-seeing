"""Deciding which colour each speaker's subtitles are drawn in.

Three things live here, in the order the pipeline uses them:

  * **the colour model** — one fixed lightness, hue free, chroma by tier;
  * **hue assignment** — placing each speaker on the hue circle as near its
    preference as it can get while staying a minimum arc from everyone else;
  * **resolution** — running a strategy, keeping only what it was confident
    about, and spacing out whatever is left.

A strategy never picks a colour. It names a hue it would prefer and says how
sure it is, and assignment decides — so however a strategy arrives at its
answer, it cannot hand two speakers the same colour or one that cannot be
read against video.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from hearing_to_seeing.design.oklch import hue_to_ass
from hearing_to_seeing.schema import SpeakerProfile, Transcript

# One lightness for every speaker colour. Fixing it is what makes hue free:
# legibility is settled once, here, instead of being re-argued for each colour.
# 0.75 sits in the same band as the colours the project used before (the Wong
# palette's orange measures 0.753, its sky blue 0.735) and still leaves every
# hue at least 0.128 chroma to work with, so no hue comes out washed grey.
# TODO: L·채도 값은 실제 영상 위에서 확인 후 확정 필요 — 밝은 장면에서의 가독성 미검증.
LIGHTNESS = 0.75

# Chroma as a fraction of the most that hue can hold at this lightness. The two
# tiers are how a named character is told apart from an unidentified one at a
# glance: vivid means "we know who this is".
MAIN_CHROMA_RATIO = 0.95
MINOR_CHROMA_RATIO = 0.32

# How far apart two speakers' hues must stay. Enough that neighbouring
# assignments do not read as the same colour; loosened automatically when there
# are more speakers than the circle can space that widely.
MIN_HUE_GAP = 40.0

# Colour every word starts out in, before the karaoke fill reaches it. Nothing
# generated here can collide with it: a speaker colour always carries chroma,
# and white has none.
BASE_COLOUR = "&H00FFFFFF"  # white

# How sure a strategy has to be before its answer is used at all. A wrong
# colour is worse than an arbitrary one: a viewer who cannot hear the audio has
# no way to catch it, so an unconvincing answer is dropped rather than shown.
# Deliberately high to start with.
# TODO: 임계값 0.7은 경험값 — 실제 전략을 붙인 뒤 정확도를 측정해 재조정 필요.
MIN_CONFIDENCE = 0.7

_EPSILON = 1e-9


@dataclass
class SpeakerCandidate:
    """What a strategy has to say about one speaker.

    `preferred_hue` is a position on the colour circle in degrees — where this
    speaker's colour would ideally sit — rather than a finished colour, which
    is what leaves assignment free to move it when two speakers want the same
    place. `hue_weight` decides who gets their wish when they collide.
    """

    label: str
    name: str | None = None
    confidence: float = 0.0
    source: str = ""
    note: str | None = None
    preferred_hue: float | None = None
    hue_weight: float = 1.0


# Takes the transcript and the media path, returns one candidate per speaker it
# has something to say about — a strategy may leave speakers out entirely.
SpeakerStrategy = Callable[[Transcript, str], dict[str, SpeakerCandidate]]


def hue_distance(a: float, b: float) -> float:
    """The shorter way round the circle between two hues, in degrees."""
    apart = abs(a - b) % 360
    return min(apart, 360 - apart)


def _is_free(hue: float, placed: Sequence[float], gap: float) -> bool:
    return all(hue_distance(hue, other) >= gap - _EPSILON for other in placed)


def _nearest_free_hue(
    preferred: float, placed: Sequence[float], gap: float
) -> float | None:
    """The hue closest to `preferred` that clears `gap` from everything placed.

    Only the preference itself and the points exactly `gap` to either side of
    an occupied hue are worth testing: when the preference is blocked, the
    nearest legal spot is always flush against whatever is blocking it.
    """
    preferred %= 360
    if _is_free(preferred, placed, gap):
        return preferred

    candidates = [
        (occupied + offset) % 360
        for occupied in placed
        for offset in (gap, -gap)
    ]
    free = [hue for hue in candidates if _is_free(hue, placed, gap)]
    if not free:
        return None
    return min(free, key=lambda hue: (hue_distance(hue, preferred), hue))


def _largest_gap_midpoint(placed: Sequence[float]) -> float:
    """The emptiest spot on the circle — where a speaker with no wish goes."""
    if not placed:
        return 0.0
    if len(placed) == 1:
        return (placed[0] + 180) % 360

    ordered = sorted(placed)
    widest_start, widest_size = ordered[0], -1.0
    for i, start in enumerate(ordered):
        size = (ordered[(i + 1) % len(ordered)] - start) % 360
        if size > widest_size:
            widest_start, widest_size = start, size
    return (widest_start + widest_size / 2) % 360


def assign_hues(
    labels: Sequence[str],
    preferences: Mapping[str, float] | None = None,
    weights: Mapping[str, float] | None = None,
    min_gap: float = MIN_HUE_GAP,
) -> dict[str, float]:
    """Places every label on the hue circle.

    Preferences are honoured in weight order, each taken as close to its wish
    as the ones already placed allow. Labels without a preference drop into the
    widest remaining gap, which spreads them out instead of letting them crowd
    whatever was assigned first.

    With no preferences at all this is an even distribution — which is what
    keeps an entirely unidentified cast distinguishable anyway.
    """
    labels = sorted(set(labels))
    if not labels:
        return {}

    preferences = preferences or {}
    weights = weights or {}
    # More speakers than the circle can space at `min_gap`: share out what
    # there is rather than failing to place anyone. Dividing by one more than
    # the count leaves slack, since placements land flush against each other
    # and an exactly-tight circle can strand the last speaker.
    gap = min(min_gap, 360 / (len(labels) + 1))

    if not preferences:
        step = 360 / len(labels)
        return {label: (i * step) % 360 for i, label in enumerate(labels)}

    wanted = sorted(
        (label for label in labels if label in preferences),
        key=lambda label: (-weights.get(label, 1.0), label),
    )
    unwanted = [label for label in labels if label not in preferences]

    assigned: dict[str, float] = {}
    placed: list[float] = []
    for label in wanted + unwanted:
        target = (
            preferences[label]
            if label in preferences
            else _largest_gap_midpoint(placed)
        )
        hue = _nearest_free_hue(target, placed, gap)
        if hue is None:  # pragma: no cover - gap is chosen to keep this unreachable
            hue = _largest_gap_midpoint(placed)
        assigned[label] = hue
        placed.append(hue)
    return assigned


def resolve_speakers(
    transcript: Transcript,
    candidates: Mapping[str, SpeakerCandidate] | None = None,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    lightness: float = LIGHTNESS,
    min_gap: float = MIN_HUE_GAP,
) -> dict[str, SpeakerProfile]:
    """Produces one profile per speaker in `transcript`.

    Confidence is judged per speaker, not for the run as a whole: a strategy
    usually knows the leads and not the bit parts, and taking what it is sure
    of beats discarding the lot. Speakers it could not place still get a hue of
    their own — so they stay distinguishable from one another — but at low
    chroma, which reads as "not one of the named characters".
    """
    labels = transcript.speakers()
    candidates = candidates or {}

    # Identity and colour arrive from different steps: a manual mapping knows
    # exactly who a speaker is and has no opinion on hue, so confidence alone
    # decides whether a candidate is believed.
    accepted = {
        label: candidate
        for label, candidate in candidates.items()
        if label in labels and candidate.confidence >= min_confidence
    }

    hues = assign_hues(
        labels,
        preferences={
            label: candidate.preferred_hue
            for label, candidate in accepted.items()
            if candidate.preferred_hue is not None
        },
        weights={label: candidate.hue_weight for label, candidate in accepted.items()},
        min_gap=min_gap,
    )

    profiles: dict[str, SpeakerProfile] = {}
    for label in labels:
        candidate = accepted.get(label)
        # Muted only says "not one of the named characters", which is a
        # statement about nothing when no character was named at all — so with
        # an entirely unidentified cast everyone keeps full chroma rather than
        # the whole film going pale for a contrast that is not being drawn.
        muted = candidate is None and bool(accepted)
        ratio = MINOR_CHROMA_RATIO if muted else MAIN_CHROMA_RATIO
        color = hue_to_ass(hues[label], lightness, ratio)

        if candidate is not None:
            profiles[label] = SpeakerProfile(
                label=label,
                color=color,
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
        profiles[label] = SpeakerProfile(
            label=label, color=color, source="unidentified", note=note,
        )

    return profiles
