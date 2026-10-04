"""Loudness → size: how big each word is drawn.

Measures each word's volume, sorts it into loud / whisper / normal, and turns
that into a scale: a shouted word quickly grows, holds and returns; a whispered
one is small for the whole line. The font size itself never changes
(layout.BASE_FONT_SIZE) — only this scale on top of it.
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
#             (applied from the first frame of the line, never animated)
LOUD_SCALE_MIN = 120
LOUD_SCALE_MAX = 160
WHISPER_SCALE_MIN = 65
WHISPER_SCALE_MAX = 80

# A whispered word is shrunk in both directions. Laid out by libass it would sit
# on the baseline; raising it by this share of the height it lost
# (font size * (1 - scale)) puts it back at the vertical centre of the row.
# 0.0 = on the baseline, ~0.3 = centred (half of ascent - descent over the font
# size, which is about 0.6 for most fonts).
WHISPER_V_CENTER = 0.3

# A shouted word has three phases within the time it is said:
#   grow (short) → hold at the scaled size (long) → return to 100% (short)
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

    A whispered word stays small for the whole line, so the line keeps only
    that much room for it; a shouted word grows over its neighbours and gives
    its room back, so it keeps its normal width.
    """
    if classify_volume(volume) == "whisper":
        return whisper_scale(volume) / 100
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


def whisper_tags(volume: float) -> str:
    """A whispered word's size, held unchanged from the first frame."""
    scale = whisper_scale(volume)
    return f"\\fscx{scale}\\fscy{scale}"


def whisper_lift(volume: float, font_size: float) -> int:
    """Pixels to raise a whispered word so it sits mid-row, not on the baseline."""
    return round(font_size * (1 - whisper_scale(volume) / 100) * WHISPER_V_CENTER)
