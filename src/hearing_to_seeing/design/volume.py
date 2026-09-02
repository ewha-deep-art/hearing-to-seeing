import numpy as np

from hearing_to_seeing.schema import Transcript, WordEntry

# TODO: FONT_SIZE_MIN/MAX는 임시 값 — Netflix Timed Text Style Guide, BBC Subtitle
#       Guidelines, WCAG 등 접근성 표준 조사 후 기획서 §다음 논의 필요 사항에 따라 최종값 확정 필요.
FONT_SIZE_MIN = 24
FONT_SIZE_MAX = 48


def calculate_rms(audio_array: np.ndarray) -> float:
    if audio_array is None or len(audio_array) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_array.astype(np.float64)))))


def normalize(values: list[float]) -> list[float]:
    # TODO: 정규화가 전체 단어 기준 전역 적용됨 — 속삭이는 장면과 소리치는 장면이
    #       동일한 크기 범위에 매핑됨. 장면별 또는 화자별 정규화 방식 검토 필요.
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
