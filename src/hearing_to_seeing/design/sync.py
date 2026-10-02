from hearing_to_seeing.schema import WordEntry
from hearing_to_seeing.design.volume import (
    BASE_FONT_SIZE,
    LOUD_THRESHOLD,
    QUIET_THRESHOLD,
    classify_volume,
)

# Forced aligners routinely stretch the final character of an utterance across
# the silence that follows it. Capping how long a word counts as "being said"
# keeps such a word from animating (or hiding its fill) for tens of seconds.
MAX_TRAILING_HOLD = 2.0

# --- The three layers of a subtitle -------------------------------------------
# converter/ass.py draws each on-screen subtitle as stacked Dialogue events:
#   0. Box  — the whole line in an opaque black box, so it reads over any video.
#   1. Fill — the whole line again: white before a word is said, hidden from the
#             moment it is said. Layer 2 draws it from then until the line ends,
#             so the word is never handed back to this layer (the hand-over of
#             the same glyphs between two layers is what made words twitch).
#   2. Pop  — one event per word (per character for the wave), positioned by
#             coordinate, carrying the motion that shows how the word was said:
#               loud    → quickly grows, holds, quickly returns; the peak size follows
#                         the volume (louder → bigger); then rests at 100%
#               whisper → small from the moment the line appears, with no animation;
#                         white until it is said, then wiped into the speaker colour.
#                         The size
#                         follows the volume (quieter → smaller). Fill and Box keep
#                         a placeholder narrowed to the same width (\fscx only, so
#                         the line height is untouched) and ass._compute_word_layout
#                         measures it that way, so no empty gap is left around it.
#               normal  → its characters rise and fall in an overlapping ripple that
#                         fits inside the word's spoken time, then rest
#             On top of the motion, every word is filled karaoke-style: white
#             (the style's SecondaryColour) wipes left to right into the speaker
#             colour (\c, the PrimaryColour) with \kf over the word's spoken
#             time — a whole-word wipe for loud/whisper, character by character
#             (one equal share of the word each) for the wave.
#             Every Pop event lasts until the line ends (see ass._pop_events).
# ASS moves (\pos, \move) apply to a whole event, which is why layer 2 splits
# words out instead of animating inside the line.
# Peak size (percent) is interpolated from the word's volume:
#   loud    : volume LOUD_THRESHOLD → LOUD_SCALE_MIN, volume 1.0 → LOUD_SCALE_MAX
#   whisper : volume 0.0 → WHISPER_SCALE_MIN, volume QUIET_THRESHOLD → WHISPER_SCALE_MAX
#             (applied from the first frame of the line, never animated)
LOUD_SCALE_MIN = 110
LOUD_SCALE_MAX = 140
WHISPER_SCALE_MIN = 80
WHISPER_SCALE_MAX = 90

# The space next to a whispered word is narrowed to that word's width scale
# (see word_separator). Multiply it further to tighten (<1.0) or loosen (>1.0)
# the gaps around whispers; 1.0 = same scale as the whisper itself.
WHISPER_SPACE_FACTOR = 1.0

# Every space in the Box / Fill / Pop lines is narrowed to this share of its
# normal width (1.0 = untouched). Applied on top of the whisper narrowing above.
SPACE_TIGHTNESS = 0.92

# A whispered word is shrunk in both directions. Laid out by libass it would sit
# on the baseline; raising it by this share of the height it lost
# (font size * (1 - scale)) puts it back at the vertical centre of the row.
# 0.0 = on the baseline, ~0.3 = centred (half of ascent - descent over the font
# size, which is about 0.6 for most fonts).
WHISPER_V_CENTER = 0.3

# A scaled word has three phases within the time it is said:
#   grow/shrink (short) → hold at the scaled size (long) → return to 100% (short)
# Each transition takes this share of the word's duration, but never less than
# SCALE_MIN_TRANSITION_MS (unless that would exceed SCALE_MAX_TRANSITION_SHARE).
# Whatever is left over is the hold.
SCALE_IN_RATIO = 0.2
SCALE_OUT_RATIO = 0.2
SCALE_MIN_TRANSITION_MS = 80
SCALE_MAX_TRANSITION_SHARE = 0.3

WAVE_RISE_PX = round(18 * BASE_FONT_SIZE / 44)  # how far each character lifts (18 px at size 44)
WAVE_RISE_MS = 70
WAVE_FALL_MS = 90

# Each character starts rising WAVE_STAGGER ms after the previous one — well
# before the previous one has landed (rise + fall = 160 ms), so the motions
# overlap into a ripple. The stagger is chosen so the whole wave ends when the
# word does, then clamped to this range; a very short word first shrinks the
# rise/fall (never below WAVE_MIN_SHRINK of their length) to still fit.
WAVE_STAGGER_MIN_MS = 25
WAVE_STAGGER_MAX_MS = 55
WAVE_MIN_SHRINK = 0.4

# Each character is its own event, so spacing comes from measured widths and can
# look looser than libass' own. Characters are pulled this share closer to the
# word's centre (0 = off, 0.04 = 4% tighter).
WAVE_TIGHTEN = 0.04


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


def build_fill_text(words: list[WordEntry], break_at: int | None = None) -> str:
    """Layer 1: white before a word is said, hidden from the moment it is.

    \\t offsets are milliseconds from the Dialogue event start, which is the
    first word's start. `break_at` is the index of the word that starts the
    second row, if any.

    Once hidden a word is never shown again here: the Pop layer keeps drawing it
    (in the speaker colour) until the line ends.

    A whispered word is hidden from the very first frame: the Pop layer draws it,
    already small, for the whole line. It stays here as an invisible placeholder
    narrowed to the word's scaled width (\\fscx only, so the row keeps its height),
    which is the width `ass._compute_word_layout` assumes for it.
    """
    if not words:
        return ""

    chunk_start = words[0].start
    parts = []
    for i, word in enumerate(words):
        if i:
            parts.append("\\N" if i == break_at else word_separator(words[i - 1], word))
        if classify_volume(word.volume) == "whisper":
            parts.append(
                f"{{\\fs{BASE_FONT_SIZE}\\alpha&HFF&\\fscx{whisper_scale(word.volume)}}}"
                f"{word.text}{{\\fscx100}}"
            )
            continue
        if classify_volume(word.volume) == "normal":
            # A wave word is drawn character by character in the Pop layer, and
            # the characters start one after another. Hiding the whole word at
            # word.start made the characters that hadn't started yet vanish for
            # a few frames. Each character is instead handed over at its own
            # start, on the same centisecond grid the Pop event starts on (the
            # \\t ends exactly then), so there is neither a gap nor a double draw.
            for ch, st in zip(word.text, wave_rise_starts(word)):
                c0 = max(0, (round(st * 100) - round(chunk_start * 100)) * 10 - 1)
                parts.append(
                    f"{{\\fs{BASE_FONT_SIZE}\\c&H00FFFFFF&\\alpha&H00&"
                    f"\\t({c0},{c0 + 1},\\alpha&HFF&)}}{ch}"
                )
            continue
        t0 = round((word.start - chunk_start) * 1000)
        parts.append(
            f"{{\\fs{BASE_FONT_SIZE}\\c&H00FFFFFF&\\alpha&H00&"
            f"\\t({t0},{t0 + 1},\\alpha&HFF&)}}{word.text}"
        )
    return "".join(parts)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def loud_scale(volume: float) -> int:
    """Peak size (percent) of a shouted word: louder is bigger."""
    t = _clamp01((volume - LOUD_THRESHOLD) / (1.0 - LOUD_THRESHOLD))
    return round(LOUD_SCALE_MIN + (LOUD_SCALE_MAX - LOUD_SCALE_MIN) * t)


def whisper_scale(volume: float) -> int:
    """Lowest size (percent) of a whispered word: quieter is smaller."""
    t = _clamp01(volume / QUIET_THRESHOLD)
    return round(WHISPER_SCALE_MIN + (WHISPER_SCALE_MAX - WHISPER_SCALE_MIN) * t)


def width_scale(word: WordEntry) -> float:
    """Horizontal scale (1.0 = normal) `word` has in the Box and Fill lines."""
    if classify_volume(word.volume) == "whisper":
        return whisper_scale(word.volume) / 100
    return 1.0


def space_scale(prev: WordEntry, nxt: WordEntry) -> int:
    """Width (percent) of the space between two words.

    The smaller neighbour's scale (whispers are narrower), times SPACE_TIGHTNESS.
    """
    pct = round(
        min(width_scale(prev), width_scale(nxt)) * 100
        * WHISPER_SPACE_FACTOR * SPACE_TIGHTNESS
    )
    return min(100, pct)


def word_separator(prev: WordEntry, nxt: WordEntry) -> str:
    """The space between two words in the Box / Fill / invisible-context lines.

    A whispered word is narrowed to 80-90% (and drawn at that size in the Pop
    layer), but a plain space stays at 100%, so the gaps around small words
    looked wider than the words themselves. The space is therefore narrowed
    with \\fscx too, to the smaller of the two neighbouring words' scales
    (whisper next to normal -> the whisper's scale; two whispers -> the smaller
    one). \\fscx is reset right after so the next word is unaffected.
    """
    pct = space_scale(prev, nxt)
    if pct >= 100:
        return " "
    return f"{{\\fscx{pct}}} {{\\fscx100}}"


def _transition_ms(duration_ms: int, ratio: float) -> int:
    wanted = max(SCALE_MIN_TRANSITION_MS, round(duration_ms * ratio))
    return max(1, min(wanted, round(duration_ms * SCALE_MAX_TRANSITION_SHARE)))


def _scale_event_text(
    word: WordEntry, color: str, cx: float, cy: float, scale: int,
    context: "WordLineContext | None" = None,
) -> str:
    """grow/shrink to `scale` → hold → back to 100%, all within the spoken span.

    With `context` the word is not placed by coordinate: the event carries the
    whole subtitle line, every other word invisible, so libass puts the visible
    word at the very pixel the Fill layer drew it at. Because the (invisible)
    line is re-centred as the word's width changes, a horizontal \\fscx growth
    stays centred on the word by itself; vertically it grows up from the
    baseline.
    """
    duration_ms = max(3, round((spoken_end(word) - word.start) * 1000))
    in_ms = _transition_ms(duration_ms, SCALE_IN_RATIO)
    back_start = duration_ms - _transition_ms(duration_ms, SCALE_OUT_RATIO)
    sweep_cs = seconds_to_centiseconds(spoken_end(word) - word.start)
    anim = (
        f"\\t(0,{in_ms},\\fscx{scale}\\fscy{scale})"
        f"\\t({back_start},{duration_ms},\\fscx100\\fscy100)"
    )
    if context is None:
        return (
            f"{{\\an5\\pos({cx:.0f},{cy:.0f})\\fs{BASE_FONT_SIZE}\\c{color}\\kf{sweep_cs}"
            f"{anim}}}{word.text}"
        )
    ax, ay = context.anchor
    return (
        f"{{\\an2\\pos({ax},{ay})\\fs{BASE_FONT_SIZE}\\alpha&HFF&}}{context.before}"
        f"{{\\alpha&H00&\\c{color}{anim}}}{{\\kf{sweep_cs}}}{word.text}"
        f"{{\\alpha&HFF&\\fscx100\\fscy100}}{context.after}"
    )


def build_loud_event_text(
    word: WordEntry, color: str, cx: float, cy: float,
    context: "WordLineContext | None" = None,
) -> str:
    """Layer 2, shouted word: grows (more when louder), holds, then returns."""
    return _scale_event_text(word, color, cx, cy, loud_scale(word.volume), context)


def build_whisper_event_text(
    word: WordEntry, color: str, cx: float, cy: float, line_start: float,
    context: "WordLineContext | None" = None,
) -> str:
    """Layer 2, whispered word: small from the first frame of the line.

    The event starts with the line, not with the word, so the word never has to
    change size: it is already at its volume-based scale. It shows white (the
    style's SecondaryColour) until it is said, then `color` wipes in over it
    with \\kf. An empty `\\k` first holds the wipe back until the word starts;
    karaoke times count from the event start, which is `line_start`.

    With `context` libass lays the word out inside the invisible line (see
    `_scale_event_text`). Its Fill placeholder is narrowed with \\fscx only, so
    the row keeps its height and the shrunk word sits on the same baseline as
    its neighbours; WHISPER_V_CENTER lifts it back to the middle of the row.
    """
    scale = whisper_scale(word.volume)
    lead_cs = round((word.start - line_start) * 100)
    lead = f"\\k{lead_cs}" if lead_cs > 0 else ""
    sweep_cs = seconds_to_centiseconds(spoken_end(word) - word.start)
    if context is None:
        return (
            f"{{\\an5\\pos({cx:.0f},{cy:.0f})\\fs{BASE_FONT_SIZE}\\c{color}"
            f"\\fscx{scale}\\fscy{scale}{lead}}}"
            f"{{\\kf{sweep_cs}}}{word.text}"
        )
    ax, ay = context.anchor
    # The line is bottom-anchored, so the shrunk word would sit on the baseline;
    # shift the whole (otherwise invisible) line up to centre it in the row.
    ay -= round(BASE_FONT_SIZE * (1 - scale / 100) * WHISPER_V_CENTER)
    return (
        f"{{\\an2\\pos({ax},{ay})\\fs{BASE_FONT_SIZE}\\alpha&HFF&}}{context.before}"
        f"{{\\alpha&H00&\\c{color}\\fscx{scale}\\fscy{scale}{lead}}}"
        f"{{\\kf{sweep_cs}}}{word.text}"
        f"{{\\alpha&HFF&\\fscx100\\fscy100}}{context.after}"
    )


class WaveLineContext:
    """What a wave character needs to be laid out by libass instead of measured.

    `anchor` is the `\\an2` position of the whole line (bottom centre, the same
    anchor the Fill layer gets from its style margins). `before[i]` and
    `after[i]` are the line text before and after character i of the word, as
    invisible text (spaces, row breaks and narrowed whispers included).
    """

    def __init__(self, anchor: tuple[int, int], before: list[str], after: list[str]):
        self.anchor = anchor
        self.before = before
        self.after = after


class WordLineContext:
    """What a loud or whispered word needs to be laid out by libass itself.

    Same idea as `WaveLineContext`, for a whole word: `anchor` is the `\\an2`
    position of the line, `before` / `after` the line text around the word as
    invisible text (spaces, row breaks and narrowed whispers included).
    """

    def __init__(self, anchor: tuple[int, int], before: str, after: str):
        self.anchor = anchor
        self.before = before
        self.after = after


def wave_rise_starts(word: WordEntry) -> list[float]:
    """When each character of a normally spoken word starts to rise."""
    n = len(word.text)
    duration_ms = max(1, round((spoken_end(word) - word.start) * 1000))
    stagger, _, _ = _wave_timing(duration_ms, n)
    return [word.start + i * stagger / 1000 for i in range(n)]


def _wave_timing(duration_ms: int, n: int) -> tuple[float, int, int]:
    """`(stagger, rise, fall)` in ms for a wave of `n` characters in `duration_ms`.

    The wave spans (n - 1) * stagger + rise + fall. The stagger is whatever makes
    that equal the word's duration, kept within the configured range; if even the
    minimum stagger doesn't leave room, the rise and fall are shortened.
    """
    rise, fall = WAVE_RISE_MS, WAVE_FALL_MS
    stagger = 0.0
    if n > 1:
        stagger = (duration_ms - rise - fall) / (n - 1)
        stagger = max(WAVE_STAGGER_MIN_MS, min(WAVE_STAGGER_MAX_MS, stagger))
    room = duration_ms - (n - 1) * stagger
    if room < rise + fall:
        k = max(WAVE_MIN_SHRINK, room / (rise + fall))
        rise, fall = max(1, round(rise * k)), max(1, round(fall * k))
    return stagger, rise, fall


def build_wave_events(
    word: WordEntry,
    color: str,
    char_positions: list[tuple[str, float, float]],
    hold_until: float | None = None,
    line_context: "WaveLineContext | None" = None,
) -> list[tuple[float, float, str]]:
    """Layer 2, normally spoken word: its characters lift and land in a ripple.

    `char_positions` is `(character, centre x, centre y)` for each character.

    With `line_context`, positions are not taken from `char_positions` at all:
    each event carries the WHOLE subtitle line, every other character made
    invisible, anchored exactly where the Fill layer anchors it. libass then
    lays the visible character out itself, so it sits at the very pixel the Fill
    layer drew it at (same font, same kerning, same rounding) and is only moved
    by the \\move. Measured positions can't match libass exactly; any mismatch
    showed up as a small jump when the word was handed from Fill to Pop.
    ASS allows one \\move per event, so each character is two events — the rise,
    then the fall — back to back. Characters start `stagger` ms apart, so one
    character's fall overlaps the next one's rise. Returns `(start, end, text)`
    per event.

    A \\move stays at its end point once it has run, so the fall event can simply
    last longer: it keeps each character resting in place until `hold_until`
    (the end of the line) instead of vanishing when the word ends.

    Each character also gets an equal share of the word's spoken time in which
    `color` wipes across it (\\kf). The wipe lives in the fall event — the long
    one — so it is never cut in half by the hand-over from the rise event; a
    character still rising when its share begins is wiped from the moment it
    starts falling (at most one rise, `WAVE_RISE_MS`, late). The rise event
    shows the character white, which is what the unwiped fall event shows too.
    """
    n = len(char_positions)
    if n == 0:
        return []

    end = spoken_end(word)
    duration_ms = max(1, round((end - word.start) * 1000))
    stagger, rise_ms, fall_ms = _wave_timing(duration_ms, n)

    slot = (end - word.start) / n
    mid = (char_positions[0][1] + char_positions[-1][1]) / 2
    hold = hold_until if hold_until is not None else 0.0
    events = []
    for i, (ch, cx, cy) in enumerate(char_positions):
        if line_context is None:
            cx = mid + (cx - mid) * (1 - WAVE_TIGHTEN)
        rise_start = word.start + i * stagger / 1000
        rise_end = rise_start + rise_ms / 1000

        if line_context is None:
            rise_head = (
                f"{{\\an5\\move({cx:.0f},{cy:.0f},{cx:.0f},{cy - WAVE_RISE_PX:.0f},0,{rise_ms})"
                f"\\fs{BASE_FONT_SIZE}\\c&H00FFFFFF&}}"
            )
            body = ch
        else:
            ax, ay = line_context.anchor
            before, after = line_context.before[i], line_context.after[i]
            rise_head = (
                f"{{\\an2\\move({ax},{ay},{ax},{ay - WAVE_RISE_PX},0,{rise_ms})"
                f"\\fs{BASE_FONT_SIZE}\\alpha&HFF&}}{before}"
                f"{{\\alpha&H00&\\c&H00FFFFFF&}}"
            )
            body = f"{ch}{{\\alpha&HFF&}}{after}"
        rise_text = rise_head + body
        events.append((rise_start, rise_end, rise_text))

        fill_from = max(word.start + i * slot, rise_end)
        fill_to = max(word.start + (i + 1) * slot, fill_from)
        lead_cs = round((fill_from - rise_end) * 100)
        lead = f"\\k{lead_cs}" if lead_cs > 0 else ""
        sweep_cs = seconds_to_centiseconds(fill_to - fill_from)

        if line_context is None:
            fall_text = (
                f"{{\\an5\\move({cx:.0f},{cy - WAVE_RISE_PX:.0f},{cx:.0f},{cy:.0f},0,{fall_ms})"
                f"\\fs{BASE_FONT_SIZE}\\c{color}{lead}}}{{\\kf{sweep_cs}}}{ch}"
            )
        else:
            fall_text = (
                f"{{\\an2\\move({ax},{ay - WAVE_RISE_PX},{ax},{ay},0,{fall_ms})"
                f"\\fs{BASE_FONT_SIZE}\\alpha&HFF&}}{before}"
                f"{{\\alpha&H00&\\c{color}{lead}}}{{\\kf{sweep_cs}}}{ch}"
                f"{{\\alpha&HFF&}}{after}"
            )
        fall_end = max(rise_end + fall_ms / 1000, end, hold)
        events.append((rise_end, fall_end, fall_text))

    return events
