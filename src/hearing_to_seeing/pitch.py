"""How high each speaker's voice is, and the gender that suggests.

A speaker the pipeline cannot name is shown as "여자 1" / "남자 1"
(design/speaker.py `speaker_names`) from the gender guessed here, with no
model and no API: each label's median fundamental frequency (F0) over its
voiced speech, measured with YIN (de Cheveigné & Kawahara, 2002). Adult
speaking voices sit around 85–155 Hz (male) and 165–255 Hz (female); between
MALE_MAX_HZ and FEMALE_MIN_HZ the guess is left unknown, and children,
shouting and falsetto will be misread.
"""

import numpy as np

from hearing_to_seeing.schema import Transcript
from hearing_to_seeing.voice import MIN_SPEAKER_SECONDS, RATE, _mono_16k, _utterances

MALE_MAX_HZ = 160.0
FEMALE_MIN_HZ = 185.0

F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
WINDOW = 512             # samples at RATE (32 ms) compared against their lagged copy
HOP = 160                # 10 ms between frames
YIN_THRESHOLD = 0.15     # a dip of the normalised difference below this is a period
SILENCE_RATIO = 0.05     # frames quieter than this share of the loudest RMS are skipped
MAX_FRAMES = 3000        # 30 s of frames per speaker is plenty for a median
MIN_VOICED_FRAMES = 50   # half a second of voiced frames, or no guess
CHUNK_FRAMES = 300       # frames per FFT batch (~20 MB) rather than all of them (~200 MB)


def _frames(samples: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    """Every frame (WINDOW + the longest lag) starting inside `spans`, at most MAX_FRAMES."""
    length = WINDOW + int(RATE / F0_MIN_HZ)
    starts = [
        np.arange(int(a * RATE), int(b * RATE) - length, HOP)
        for a, b in spans
    ]
    starts = np.concatenate(starts) if starts else np.empty(0, dtype=int)
    starts = starts[(starts >= 0) & (starts + length <= len(samples))]
    if len(starts) > MAX_FRAMES:
        starts = starts[np.linspace(0, len(starts) - 1, MAX_FRAMES).astype(int)]
    return samples[starts[:, None] + np.arange(length)]


def _rms(frames: np.ndarray) -> np.ndarray:
    """Loudness of each frame's compared window, its DC offset removed."""
    window = frames - frames.mean(axis=1, keepdims=True)
    return np.sqrt((window[:, :WINDOW] ** 2).mean(axis=1))


def frame_f0(frames: np.ndarray, loudest: float | None = None) -> np.ndarray:
    """F0 (Hz) of each frame, NaN where it is unvoiced or silent (YIN).

    Silent means quieter than SILENCE_RATIO of `loudest` — by default the
    loudest of `frames`; pass the speaker's when `frames` is one batch of many.
    """
    if not len(frames):
        return np.empty(0)
    frames = frames - frames.mean(axis=1, keepdims=True)
    lag_min, lag_max = int(RATE / F0_MAX_HZ), int(RATE / F0_MIN_HZ)
    n = 1 << int(np.ceil(np.log2(frames.shape[1] + WINDOW)))
    # d(τ) = Σ x_j² + Σ x_{j+τ}² − 2 Σ x_j x_{j+τ}, over j < WINDOW: the
    # cross term for every lag at once by FFT, the energies by running sums.
    spectrum = np.fft.rfft(frames, n) * np.conj(np.fft.rfft(frames[:, :WINDOW], n))
    cross = np.fft.irfft(spectrum, n)[:, :lag_max + 1]
    squares = np.cumsum(np.pad(frames ** 2, ((0, 0), (1, 0))), axis=1)
    lags = np.arange(lag_max + 1)
    energy0 = squares[:, WINDOW][:, None]
    energy = squares[:, lags + WINDOW] - squares[:, lags]
    diff = np.maximum(energy0 + energy - 2 * cross, 0.0)
    # Cumulative mean normalised difference: d'(0) = 1, d'(τ) = d(τ)·τ / Σ_{1..τ} d.
    running = np.cumsum(diff[:, 1:], axis=1)
    cmnd = np.ones_like(diff)
    cmnd[:, 1:] = diff[:, 1:] * lags[1:] / np.maximum(running, 1e-12)

    rms = np.sqrt(energy0[:, 0] / WINDOW)
    loud_enough = rms >= SILENCE_RATIO * ((rms.max() if loudest is None else loudest) or 1.0)
    f0 = np.full(len(frames), np.nan)
    for i in np.flatnonzero(loud_enough):
        row = cmnd[i]
        below = np.flatnonzero(row[lag_min:] < YIN_THRESHOLD)
        if not len(below):
            continue
        lag = lag_min + below[0]
        while lag + 1 <= lag_max and row[lag + 1] < row[lag]:  # down to the bottom of the dip
            lag += 1
        f0[i] = RATE / lag
    return f0


def speaker_pitches(
    transcript: Transcript, sample_rate: int, audio: np.ndarray,
) -> dict[str, float | None]:
    """Label → median F0 (Hz) of its voiced speech; None with too little of it."""
    samples = _mono_16k(sample_rate, audio)
    spans: dict[str, list[tuple[float, float]]] = {}
    seconds: dict[str, float] = {}
    for u in _utterances(transcript.words):
        label = u[0].speaker
        spans.setdefault(label, []).append((u[0].start, u[-1].end))
        seconds[label] = seconds.get(label, 0.0) + u[-1].end - u[0].start
    pitches = {}
    for label in transcript.speakers():
        frames = _frames(samples, spans.get(label, []))
        loudest = float(_rms(frames).max()) if len(frames) else None
        f0 = np.concatenate([np.empty(0)] + [
            frame_f0(frames[i:i + CHUNK_FRAMES], loudest) for i in range(0, len(frames), CHUNK_FRAMES)
        ])
        voiced = f0[~np.isnan(f0)]
        enough = seconds.get(label, 0.0) >= MIN_SPEAKER_SECONDS and len(voiced) >= MIN_VOICED_FRAMES
        pitches[label] = float(np.median(voiced)) if enough else None
    return pitches


def gender(f0: float | None) -> str | None:
    """"male" / "female" from a median F0, None when it could be either."""
    if f0 is None:
        return None
    if f0 < MALE_MAX_HZ:
        return "male"
    if f0 > FEMALE_MIN_HZ:
        return "female"
    return None

