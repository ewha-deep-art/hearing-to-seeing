"""Loudness → size: how big each word is drawn.

Measures each word's volume, sorts it into loud / whisper / normal, and turns
that into a scale: a shouted word quickly grows, holds and returns; a whispered
one does the opposite — it quickly shrinks as it is said, holds, and returns to
full size. The font size itself never changes (layout.BASE_FONT_SIZE) — only
this scale on top of it.
"""

from dataclasses import replace

import numpy as np

from hearing_to_seeing.schema import Transcript

# Relative volume (0–1, see relative_volumes) at or above which a word counts as
# shouted, and at or below which as whispered.
# Loud side: anchored to a z-score. A word whose raw volume is LOUD_Z standard
# deviations above its speaker's mean is mapped to exactly LOUD_THRESHOLD, so
# `volume >= LOUD_THRESHOLD` means "z >= LOUD_Z".
# Quiet side: anchored to an absolute raw volume. A word whose raw (stored)
# volume is QUIET_VOLUME or lower is mapped to at most QUIET_THRESHOLD, so
# `volume <= QUIET_THRESHOLD` means "raw volume <= QUIET_VOLUME", whoever speaks.
# TODO: LOUD_Z / LOUD_Z_FULL / QUIET_VOLUME는 경험값 — 실제 영상으로 분류 결과를 확인한 뒤 조정 필요.
LOUD_THRESHOLD = 0.9
QUIET_THRESHOLD = 0.1

# Raw volume (the stored 0–1 value, min-max normalised over the whole video) at or
# below which a word is whispered, for every speaker alike. The stored value is
# scaled by each video's own loudest and quietest word, so the same number means
# something different in every video.
QUIET_VOLUME = 0.08

# Loud side of relative_volumes: z >= LOUD_Z counts as shouted; z >= LOUD_Z_FULL
# gets the full relative volume 1.0, i.e. LOUD_SCALE_MAX (loud_scale interpolates
# between the two). LOUD_Z_FULL is a free parameter, not derived from data.
LOUD_Z = 1.5
LOUD_Z_FULL = 2.5

# Peak size (percent) is interpolated from the word's volume:
#   loud    : volume LOUD_THRESHOLD → LOUD_SCALE_MIN, volume 1.0 → LOUD_SCALE_MAX
#   whisper : volume 0.0 → WHISPER_SCALE_MIN, volume QUIET_THRESHOLD → WHISPER_SCALE_MAX
#             (reached as the word is said, held, then back to 100%)
LOUD_SCALE_MIN = 106
LOUD_SCALE_MAX = 118
WHISPER_SCALE_MIN = 86
WHISPER_SCALE_MAX = 92

# Extra transparency of a whispered word (ASS alpha, 0x30 = 19 % transparent).
QUIET_ALPHA = 0x30

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


def classify_volume(
    volume: float, loud_thr: float = LOUD_THRESHOLD, quiet_thr: float = QUIET_THRESHOLD,
) -> str:
    """Sorts a normalised volume (0–1) into "loud", "whisper" or "normal"."""
    if volume >= loud_thr:
        return "loud"
    if volume <= quiet_thr:
        return "whisper"
    return "normal"


def relative_volumes(
    words, min_words: int = 12, lo_pct: float = 5, hi_pct: float = 95,
    loud_z: float = LOUD_Z, loud_z_full: float = LOUD_Z_FULL,
    quiet_volume: float = QUIET_VOLUME,
):
    """Copies of `words` with `volume` re-expressed on the 0–1 scale the rest of
    the code classifies and sizes by (QUIET_THRESHOLD / LOUD_THRESHOLD).

    The stored volume is min-max normalised over the whole video, so one very
    loud word squeezes everything else towards 0. The two ends are therefore
    anchored differently:

    Quiet end, absolute and the same for every speaker (so also for a speaker
    with fewer than `min_words` words):
        raw 0            -> 0.0
        raw quiet_volume -> QUIET_THRESHOLD  (so raw <= quiet_volume is whispered)

    Everything above that is relative to the speaker, piecewise linear:
        raw quiet_volume                 -> QUIET_THRESHOLD
        the speaker's median             -> 0.5
        mean + loud_z * std              -> LOUD_THRESHOLD  (so z >= loud_z is loud)
        mean + loud_z_full * std         -> 1.0             (and above, clipped)
    If the speaker's median is itself at or below `quiet_volume`, 0.5 sits at
    `quiet_volume` instead and the first segment is empty (the scale jumps from
    QUIET_THRESHOLD to 0.5 there). A speaker with fewer than `min_words` words
    has no reliable baseline, and a speaker whose volume never varies (std 0)
    has none either: their words above `quiet_volume` stay at 0.5.

    `loud_z` has to be above 1: the median is never more than one standard
    deviation from the mean, which keeps the loud segment well defined.

    `lo_pct` and `hi_pct` are no longer used (the scale used to run between
    those percentiles); they are kept so existing callers do not break.

    A proper fix belongs in `annotate_volumes` (per-speaker dB z-score from the
    raw RMS); this works on the volumes already stored in the JSON.
    """
    if not 1 < loud_z < loud_z_full:
        raise ValueError(f"need 1 < loud_z < loud_z_full, got {loud_z}, {loud_z_full}")
    if quiet_volume <= 0:
        raise ValueError(f"quiet_volume has to be above 0, got {quiet_volume}")
    by_speaker: dict[str, list[float]] = {}
    missing = 0
    for w in words:
        if w.volume is None:  # never measured (e.g. no audio for this word)
            missing += 1
            continue
        by_speaker.setdefault(w.speaker, []).append(w.volume)
    if missing:
        print(f"[volume] {missing} of {len(words)} words have no volume; they are drawn without volume effects")
    stats = {}
    for spk, vols in by_speaker.items():
        if len(vols) < min_words:
            stats[spk] = None
            continue
        stats[spk] = (float(np.median(vols)), float(np.mean(vols)), float(np.std(vols)))

    def rel(w):
        if w.volume is None:
            return 0.5
        if w.volume <= quiet_volume:
            return QUIET_THRESHOLD * (w.volume / quiet_volume)
        st = stats.get(w.speaker)
        if st is None:
            return 0.5
        med, mean, std = st
        if std <= 0:
            return 0.5
        mid = max(med, quiet_volume)  # where the scale reaches 0.5
        loud_at = mean + loud_z * std
        full_at = mean + loud_z_full * std
        if w.volume >= loud_at:
            share = min(1.0, (w.volume - loud_at) / (full_at - loud_at))
            return LOUD_THRESHOLD + (1.0 - LOUD_THRESHOLD) * share
        if w.volume >= mid:
            r = 0.5 + (LOUD_THRESHOLD - 0.5) * (w.volume - mid) / (loud_at - mid)
            # rounding to 4 decimals must not lift a word below loud_z onto the threshold
            return min(r, LOUD_THRESHOLD - 1e-4)
        r = QUIET_THRESHOLD + (0.5 - QUIET_THRESHOLD) * (w.volume - quiet_volume) / (mid - quiet_volume)
        # ... nor drop a word above quiet_volume onto QUIET_THRESHOLD
        return max(r, QUIET_THRESHOLD + 1e-4)

    return [replace(w, volume=round(rel(w), 4)) for w in words]


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


def loud_scale(
    volume: float, lo: int = LOUD_SCALE_MIN, hi: int = LOUD_SCALE_MAX,
    thr: float = LOUD_THRESHOLD,
) -> int:
    """Peak size (percent) of a shouted word: louder is bigger."""
    t = _clamp01((volume - thr) / (1.0 - thr))
    return round(lo + (hi - lo) * t)


def whisper_scale(
    volume: float, lo: int = WHISPER_SCALE_MIN, hi: int = WHISPER_SCALE_MAX,
    thr: float = QUIET_THRESHOLD,
) -> int:
    """Size (percent) of a whispered word: quieter is smaller."""
    t = _clamp01(volume / thr)
    return round(lo + (hi - lo) * t)


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


def loud_tags(volume: float, spoken_seconds: float, scale: int | None = None) -> str:
    """Grow to `loud_scale` → hold → back to 100%, all within `spoken_seconds`.

    `\\t` times count from the event start, which is when the word is said.
    """
    scale = loud_scale(volume) if scale is None else scale
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


def whisper_tags(volume: float, spoken_seconds: float, scale: int | None = None) -> str:
    """First part of a whispered word: shrink to `whisper_scale`, then hold.

    `\\t` times count from the event start, which is when the word is said. The
    return to 100% is a second event (`whisper_return_tags`): ASS allows only one
    `\\move`, and the word has to be lifted while it shrinks and lowered again
    while it returns.
    """
    scale = whisper_scale(volume) if scale is None else scale
    in_ms = whisper_phases(spoken_seconds)[0]
    return f"\\t(0,{in_ms},\\fscx{scale}\\fscy{scale})"


def whisper_return_tags(volume: float, spoken_seconds: float, scale: int | None = None) -> str:
    """Second part: start at `whisper_scale`, return to 100%, stay there."""
    scale = whisper_scale(volume) if scale is None else scale
    out_ms = whisper_phases(spoken_seconds)[2]
    return f"\\fscx{scale}\\fscy{scale}\\t(0,{out_ms},\\fscx100\\fscy100)"


def whisper_lift(volume: float, font_size: float, scale: int | None = None) -> int:
    """Pixels to raise a whispered word so it sits mid-row, not on the baseline."""
    scale = whisper_scale(volume) if scale is None else scale
    return round(font_size * (1 - scale / 100) * WHISPER_V_CENTER)
