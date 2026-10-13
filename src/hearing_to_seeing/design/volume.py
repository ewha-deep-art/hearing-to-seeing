"""Loudness → size: how big each word is drawn.

Measures each word's volume, sorts it into loud / whisper / normal, and turns
that into a scale: a shouted word quickly grows, holds and returns; a whispered
one does the opposite — it quickly shrinks as it is said, holds, and returns to
full size. The font size itself never changes (layout.BASE_FONT_SIZE) — only
this scale on top of it.
"""

import numpy as np

from hearing_to_seeing.schema import Transcript

# Normalised volume (0–1) at or above which a word counts as shouted, and at or
# below which it counts as whispered.
# TODO: 임계값 0.7/0.3은 경험값 — 실제 영상으로 분류 결과를 확인한 뒤 조정 필요.
LOUD_THRESHOLD = 0.7
QUIET_THRESHOLD = 0.3

# Peak size (percent) is interpolated from the word's volume:
#   loud    : volume LOUD_THRESHOLD → LOUD_SCALE_MIN, volume 1.0 → LOUD_SCALE_MAX
#   whisper : volume 0.0 → WHISPER_SCALE_MIN, volume QUIET_THRESHOLD → WHISPER_SCALE_MAX
#             (reached as the word is said, held, then back to 100%)
LOUD_SCALE_MIN = 120
LOUD_SCALE_MAX = 160
WHISPER_SCALE_MIN = 65
WHISPER_SCALE_MAX = 80

# A whispered word is shrunk in both directions. Laid out by libass it would sit
# on the baseline; raising it by this share of the height it lost
# (font size * (1 - scale)) keeps it centred on the row's vertical centre. The
# raise follows the shrink and the return, so the word scales about its own
# centre point.
# 0.0 = on the baseline, ~0.3 = centred (half of ascent - descent over the font
# size, which is about 0.6 for most fonts).
WHISPER_V_CENTER = 0.3

# A shouted or whispered word has three phases within the time it is said:
#   grow/shrink (short) → hold at the scaled size (long) → return to 100% (short)
# Each transition takes this share of the word's duration, but never less than
# SCALE_MIN_TRANSITION_MS (unless that would exceed SCALE_MAX_TRANSITION_SHARE).
# Whatever is left over is the hold.
SCALE_IN_RATIO = 0.2
SCALE_OUT_RATIO = 0.2
SCALE_MIN_TRANSITION_MS = 80
SCALE_MAX_TRANSITION_SHARE = 0.3


def calculate_rms(audio_array: np.ndarray) -> float:
    if audio_array is None or len(audio_array) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_array.astype(np.float64)))))


def normalize(values: list[float]) -> list[float]:
    # TODO: 정규화가 전체 단어 기준 전역 적용됨 — 속삭이는 장면과 소리치는 장면이
    #       동일한 기준으로 분류됨. 장면별 또는 화자별 정규화 방식 검토 필요.
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo == 0:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def classify_volume(volume: float) -> str:
    """Sorts a normalised volume (0–1) into "loud", "whisper" or "normal"."""
    if volume >= LOUD_THRESHOLD:
        return "loud"
    if volume <= QUIET_THRESHOLD:
        return "whisper"
    return "normal"


def annotate_volumes(
    transcript: Transcript,
    sample_rate: int,
    audio_data: np.ndarray,
) -> None:
    rms_values = []
    for word in transcript.words:
        start_idx = int(word.start * sample_rate)
        end_idx = int(word.end * sample_rate)
        segment = audio_data[start_idx:end_idx]
        rms_values.append(calculate_rms(segment))

    normalized = normalize(rms_values)
    for word, vol in zip(transcript.words, normalized):
        word.volume = round(vol, 4)


# --- size ----------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def loud_scale(volume: float) -> int:
    """Peak size (percent) of a shouted word: louder is bigger."""
    t = _clamp01((volume - LOUD_THRESHOLD) / (1.0 - LOUD_THRESHOLD))
    return round(LOUD_SCALE_MIN + (LOUD_SCALE_MAX - LOUD_SCALE_MIN) * t)


def whisper_scale(volume: float) -> int:
    """Size (percent) of a whispered word: quieter is smaller."""
    t = _clamp01(volume / QUIET_THRESHOLD)
    return round(WHISPER_SCALE_MIN + (WHISPER_SCALE_MAX - WHISPER_SCALE_MIN) * t)


def width_scale(volume: float) -> float:
    """Horizontal scale (1.0 = normal) a word takes up in the line.

    Always 1.0: the line, and the black box behind it, keep the room every word
    has at full size. A whispered word shrinks inside that room about its own
    centre, so it leaves equal margins on both sides instead of pulling the
    words around it closer, then returns to full size; a shouted word grows over
    its neighbours and gives the room back. Kept as a function so layout.py's
    spacing keeps one place to ask.
    """
    return 1.0


def narrow_tag(volume: float) -> str:
    """`\\fscx` that narrows a word to `width_scale` ("" when it keeps 100%).

    Width only, so the row keeps its height.
    """
    pct = round(width_scale(volume) * 100)
    return f"\\fscx{pct}" if pct != 100 else ""


def _transition_ms(duration_ms: int, ratio: float) -> int:
    wanted = max(SCALE_MIN_TRANSITION_MS, round(duration_ms * ratio))
    return max(1, min(wanted, round(duration_ms * SCALE_MAX_TRANSITION_SHARE)))


def loud_tags(volume: float, spoken_seconds: float) -> str:
    """Grow to `loud_scale` → hold → back to 100%, all within `spoken_seconds`.

    `\\t` times count from the event start, which is when the word is said.
    """
    scale = loud_scale(volume)
    duration_ms = max(3, round(spoken_seconds * 1000))
    in_ms = _transition_ms(duration_ms, SCALE_IN_RATIO)
    back_start = duration_ms - _transition_ms(duration_ms, SCALE_OUT_RATIO)
    return (
        f"\\t(0,{in_ms},\\fscx{scale}\\fscy{scale})"
        f"\\t({back_start},{duration_ms},\\fscx100\\fscy100)"
    )


def whisper_phases(spoken_seconds: float) -> tuple[int, int, int]:
    """`(shrink_ms, return_start_ms, return_ms)` of a whispered word.

    All counted from the moment the word is said: it shrinks during the first
    `shrink_ms`, holds until `return_start_ms`, then takes `return_ms` to get
    back to full size (the same phases as a shouted word, see `loud_tags`).
    """
    duration_ms = max(3, round(spoken_seconds * 1000))
    in_ms = _transition_ms(duration_ms, SCALE_IN_RATIO)
    out_ms = _transition_ms(duration_ms, SCALE_OUT_RATIO)
    return in_ms, duration_ms - out_ms, out_ms


def whisper_tags(volume: float, spoken_seconds: float) -> str:
    """First part of a whispered word: shrink to `whisper_scale`, then hold.

    `\\t` times count from the event start, which is when the word is said. The
    return to 100% is a second event (`whisper_return_tags`): ASS allows only one
    `\\move`, and the word has to be lifted while it shrinks and lowered again
    while it returns.
    """
    scale = whisper_scale(volume)
    in_ms = whisper_phases(spoken_seconds)[0]
    return f"\\t(0,{in_ms},\\fscx{scale}\\fscy{scale})"


def whisper_return_tags(volume: float, spoken_seconds: float) -> str:
    """Second part: start at `whisper_scale`, return to 100%, stay there."""
    scale = whisper_scale(volume)
    out_ms = whisper_phases(spoken_seconds)[2]
    return f"\\fscx{scale}\\fscy{scale}\\t(0,{out_ms},\\fscx100\\fscy100)"


def whisper_lift(volume: float, font_size: float) -> int:
    """Pixels to raise a whispered word so it sits mid-row, not on the baseline."""
    return round(font_size * (1 - whisper_scale(volume) / 100) * WHISPER_V_CENTER)
