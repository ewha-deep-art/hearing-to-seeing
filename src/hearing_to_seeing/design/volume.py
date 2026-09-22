import numpy as np

from hearing_to_seeing.schema import Transcript, WordEntry

BASE_FONT_SIZE = 44  # 모든 글자 공통 크기 — 음량에 따라 크기 자체는 더 이상 안 바꿈

LOUD_THRESHOLD = 0.7
QUIET_THRESHOLD = 0.3


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


def classify_volume(volume: float) -> str:
    """정규화된 음량(0~1)을 세 구간으로 분류: loud / whisper / normal."""
    if volume >= LOUD_THRESHOLD:
        return "loud"
    if volume <= QUIET_THRESHOLD:
        return "whisper"
    return "normal"


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