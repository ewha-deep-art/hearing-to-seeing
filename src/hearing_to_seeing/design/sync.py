from hearing_to_seeing.schema import WordEntry
from hearing_to_seeing.design.volume import BASE_FONT_SIZE

# Forced aligners routinely stretch the final character of an utterance across
# the silence that follows it. Capping how long a word counts as "being said"
# keeps such a word from animating (or hiding its fill) for tens of seconds.
MAX_TRAILING_HOLD = 2.0

# --- The three layers of a subtitle -------------------------------------------
# converter/ass.py draws each on-screen subtitle as stacked Dialogue events:
#   0. Box  — the whole line in an opaque black box, so it reads over any video.
#   1. Fill — the whole line again: white before a word is said, hidden while it
#             is being said (layer 2 draws it instead), speaker colour after.
#   2. Pop  — one event per word (per character for the wave), positioned by
#             coordinate, carrying the motion that shows how the word was said:
#               loud    → grows to LOUD_SCALE and settles back
#               whisper → shrinks to WHISPER_SCALE and settles back
#               normal  → its characters rise and fall one after another
# ASS moves (\pos, \move) apply to a whole event, which is why layer 2 splits
# words out instead of animating inside the line.
LOUD_SCALE = 150           # \fscx/\fscy at the peak, percent
LOUD_GROW_RATIO = 0.4      # share of the word's duration spent growing

WHISPER_SCALE = 65
WHISPER_SHRINK_RATIO = 0.4

WAVE_RISE_PX = 18          # how far each character lifts
WAVE_RISE_MS = 70
WAVE_FALL_MS = 90


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


def build_fill_text(words: list[WordEntry], color: str, break_at: int | None = None) -> str:
    """Layer 1: white before a word is said, hidden while it is, `color` after.

    \\t offsets are milliseconds from the Dialogue event start, which is the
    first word's start. `break_at` is the index of the word that starts the
    second row, if any.
    """
    if not words:
        return ""

    chunk_start = words[0].start
    parts = []
    for i, word in enumerate(words):
        if i:
            parts.append("\\N" if i == break_at else " ")
        t0 = round((word.start - chunk_start) * 1000)
        t1 = round((spoken_end(word) - chunk_start) * 1000)
        parts.append(
            f"{{\\fs{BASE_FONT_SIZE}\\c&H00FFFFFF&\\alpha&H00&"
            f"\\t({t0},{t0 + 1},\\alpha&HFF&)"
            f"\\t({t1},{t1 + 1},\\alpha&H00&\\c{color})}}{word.text}"
        )
    return "".join(parts)


def _scale_event_text(
    word: WordEntry, color: str, cx: float, cy: float, scale: int, ratio: float,
) -> str:
    duration_ms = max(1, round((spoken_end(word) - word.start) * 1000))
    turn_ms = max(1, round(duration_ms * ratio))
    return (
        f"{{\\an5\\pos({cx:.0f},{cy:.0f})\\fs{BASE_FONT_SIZE}\\c{color}"
        f"\\t(0,{turn_ms},\\fscx{scale}\\fscy{scale})"
        f"\\t({turn_ms},{duration_ms},\\fscx100\\fscy100)}}{word.text}"
    )


def build_loud_event_text(word: WordEntry, color: str, cx: float, cy: float) -> str:
    """Layer 2, shouted word: grows while it is said, then back to size."""
    return _scale_event_text(word, color, cx, cy, LOUD_SCALE, LOUD_GROW_RATIO)


def build_whisper_event_text(word: WordEntry, color: str, cx: float, cy: float) -> str:
    """Layer 2, whispered word: shrinks while it is said, then back to size."""
    return _scale_event_text(word, color, cx, cy, WHISPER_SCALE, WHISPER_SHRINK_RATIO)


def build_wave_events(
    word: WordEntry, color: str, char_positions: list[tuple[str, float, float]]
) -> list[tuple[float, float, str]]:
    """Layer 2, normally spoken word: its characters lift and land in turn.

    `char_positions` is `(character, centre x, centre y)` for each character.
    ASS allows one \\move per event, so each character is two events — the rise,
    then the fall — back to back. Returns `(start, end, text)` per event.
    """
    n = len(char_positions)
    if n == 0:
        return []

    end = spoken_end(word)
    slot = (end - word.start) / n
    events = []
    for i, (ch, cx, cy) in enumerate(char_positions):
        rise_start = word.start + i * slot
        rise_dur = max(1, min(WAVE_RISE_MS, round(slot * 1000)))
        fall_dur = max(1, min(WAVE_FALL_MS, round(slot * 1000)))
        rise_end = rise_start + rise_dur / 1000

        rise_text = (
            f"{{\\an5\\move({cx:.0f},{cy:.0f},{cx:.0f},{cy - WAVE_RISE_PX:.0f},0,{rise_dur})"
            f"\\fs{BASE_FONT_SIZE}\\c{color}}}{ch}"
        )
        events.append((rise_start, rise_end, rise_text))

        fall_text = (
            f"{{\\an5\\move({cx:.0f},{cy - WAVE_RISE_PX:.0f},{cx:.0f},{cy:.0f},0,{fall_dur})"
            f"\\fs{BASE_FONT_SIZE}\\c{color}}}{ch}"
        )
        events.append((rise_end, max(end, rise_end), fall_text))

    return events
