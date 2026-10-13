"""Word timing → motion in step with the voice.

Two effects follow the moment each word is said:
  - fill: every word is wiped karaoke-style from white (the style's
    SecondaryColour) into its colour (\\c, the PrimaryColour) with \\kf over
    the time it is said — a whole-word wipe, or one character after another
    for a waved word.
  - wave: the characters of a word lift and land one after another in a ripple
    that fits inside the word's spoken time.

Which words wave, and the colour they fill into, are decided elsewhere
(volume.py, speaker.py); converter/ass.py puts the pieces together.
"""

from typing import NamedTuple

from hearing_to_seeing.design.layout import BASE_FONT_SIZE
from hearing_to_seeing.schema import WordEntry

# Forced aligners routinely stretch the final character of an utterance across
# the silence that follows it. Capping how long a word counts as "being said"
# keeps such a word from animating (or hiding its fill) for tens of seconds.
MAX_TRAILING_HOLD = 2.0

WAVE_RISE_PX = round(18 * BASE_FONT_SIZE / 44)  # how far each character lifts (18 px at size 44)

# Each character of a waved word gets an equal share ("slot") of the time the
# word is spoken, and starts rising when its slot starts. A rise + fall lasts
# as long as its slot, so in a long word the characters go up and down one
# after another, slowly, in step with the voice. A short word has slots too
# brief for a visible motion, so each rise + fall is kept at least
# WAVE_MIN_MOTION_MS and the characters' motions overlap into a ripple.
# WAVE_MAX_MOTION_MS stops a single drawn-out character from rising for
# seconds. The rise takes WAVE_RISE_SHARE of the motion, the fall the rest.
WAVE_MIN_MOTION_MS = 240
WAVE_MAX_MOTION_MS = 900
WAVE_RISE_SHARE = 0.45


def seconds_to_centiseconds(seconds: float) -> int:
    return max(1, round(seconds * 100))


def spoken_end(word: WordEntry) -> float:
    """When `word` stops being said, with an aligner-stretched tail cut off."""
    return min(word.end, word.start + MAX_TRAILING_HOLD)


def karaoke_durations(words: list[WordEntry]) -> list[int]:
    """Per-word durations (centiseconds) that span the whole line.

    Each word holds until the next word begins, so the sum is the time from the
    first word's start to the last word's (capped) end — the span the line
    needs to stay on screen for every effect to finish.
    """
    durations = []
    for i, word in enumerate(words):
        if i + 1 < len(words):
            span = words[i + 1].start - word.start
        else:
            span = spoken_end(word) - word.start
        durations.append(seconds_to_centiseconds(span))
    return durations


# --- fill ------------------------------------------------------------------------

def wipe_tags(event_start: float, wipe_start: float, wipe_end: float) -> tuple[str, str]:
    """`(lead, sweep)` karaoke tags that wipe a text from `wipe_start` to `wipe_end`.

    Karaoke times count from the event start, so an empty `\\k` (`lead`, ""
    when there is nothing to wait for) first holds the wipe back until
    `wipe_start`; `sweep` is the `\\kf` that then runs it.
    """
    lead_cs = round((wipe_start - event_start) * 100)
    lead = f"\\k{lead_cs}" if lead_cs > 0 else ""
    return lead, f"\\kf{seconds_to_centiseconds(wipe_end - wipe_start)}"


def hide_tag(at_ms: int) -> str:
    """Makes text vanish `at_ms` after the event start (the Fill layer hand-over)."""
    return f"\\t({at_ms},{at_ms + 1},\\alpha&HFF&)"


def hand_over_ms(word: WordEntry, line_start: float) -> int:
    """When (ms into the line) a word moves from the Fill layer to its own event."""
    return round((word.start - line_start) * 1000)


def wave_hand_over_ms(word: WordEntry, line_start: float) -> list[int]:
    """`hand_over_ms` for each character of a waved word.

    The characters start one after another; hiding the whole word at its start
    made the ones that hadn't started yet vanish for a few frames. Each is
    handed over at its own start instead, on the same centisecond grid its
    event starts on (the hide ends exactly then), so there is neither a gap nor
    a double draw.
    """
    return [
        max(0, (round(c.rise_start * 100) - round(line_start * 100)) * 10 - 1)
        for c in wave_chars(word)
    ]


# --- wave --------------------------------------------------------------------------

class WaveChar(NamedTuple):
    """One character's motion and wipe, in seconds (durations in ms)."""
    rise_start: float
    rise_end: float
    rise_ms: int
    fall_ms: int
    wipe_start: float
    wipe_end: float


def _wave_timing(duration_ms: int, n: int) -> tuple[float, int, int]:
    """`(stagger, rise, fall)` in ms for a wave of `n` characters in `duration_ms`.

    The stagger is each character's share of the word's time. Rise + fall equals
    that share (one character after the other) unless it is shorter than
    WAVE_MIN_MOTION_MS, in which case the motions overlap, or longer than
    WAVE_MAX_MOTION_MS, in which case the character waits for its turn.
    """
    slot = duration_ms / max(1, n)
    motion = min(WAVE_MAX_MOTION_MS, max(WAVE_MIN_MOTION_MS, slot))
    rise = max(1, round(motion * WAVE_RISE_SHARE))
    fall = max(1, round(motion) - rise)
    return slot, rise, fall


def wave_chars(word: WordEntry) -> list[WaveChar]:
    """How each character of `word` lifts, lands and fills.

    Characters start `stagger` ms apart (one share of the word's time each): in
    a long word one lands before the next rises, in a short one the motions
    overlap. Each character also gets an equal share of the spoken time in
    which it is wiped. The wipe belongs to the fall — the long part, which
    keeps the character resting in place afterwards — so it is never cut in
    half by the hand-over from rise to fall; a character still rising when its
    share begins is wiped from the moment it starts falling (at most one rise
    late).
    """
    n = len(word.text)
    if n == 0:
        return []
    end = spoken_end(word)
    duration_ms = max(1, round((end - word.start) * 1000))
    stagger, rise_ms, fall_ms = _wave_timing(duration_ms, n)
    slot = (end - word.start) / n

    chars = []
    for i in range(n):
        rise_start = word.start + i * stagger / 1000
        rise_end = rise_start + rise_ms / 1000
        wipe_start = max(word.start + i * slot, rise_end)
        wipe_end = max(word.start + (i + 1) * slot, wipe_start)
        chars.append(WaveChar(rise_start, rise_end, rise_ms, fall_ms, wipe_start, wipe_end))
    return chars


def rise_move(anchor: tuple[int, int], ms: int) -> str:
    """`\\move` that lifts an event anchored at `anchor` by WAVE_RISE_PX."""
    x, y = anchor
    return f"\\move({x},{y},{x},{y - WAVE_RISE_PX},0,{ms})"


def fall_move(anchor: tuple[int, int], ms: int) -> str:
    """`\\move` that lands a lifted event back on `anchor` (and stays there)."""
    x, y = anchor
    return f"\\move({x},{y - WAVE_RISE_PX},{x},{y},0,{ms})"
