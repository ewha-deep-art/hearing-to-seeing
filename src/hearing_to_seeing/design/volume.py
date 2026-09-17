import numpy as np

from hearing_to_seeing.schema import Transcript, WordEntry

FONT_SIZE_MIN = 36
FONT_SIZE_MAX = 64


def calculate_rms(audio_array: np.ndarray) -> float:
    if audio_array is None or len(audio_array) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_array.astype(np.float64)))))


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo == 0:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def compute_font_size(
    normalized_volume: float,
    min_size: int = FONT_SIZE_MIN,
    max_size: int = FONT_SIZE_MAX,
) -> int:
    return round(min_size + (max_size - min_size) * normalized_volume)


def annotate_volumes(transcript: Transcript, sample_rate: int, audio_data: np.ndarray) -> None:
    rms_values = []
    for word in transcript.words:
        start_idx = int(word.start * sample_rate)
        end_idx = int(word.end * sample_rate)
        segment = audio_data[start_idx:end_idx]
        rms_values.append(calculate_rms(segment))

    normalized = normalize(rms_values)
    for word, vol in zip(transcript.words, normalized):
        word.volume = round(vol, 4)