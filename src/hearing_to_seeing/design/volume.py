import numpy as np

from hearing_to_seeing.schema import Transcript

# Every word is drawn at one size; loudness shows as motion instead (sync.py:
# a loud word pops bigger, a quiet one shrinks, and settles back). Keeping the
# size fixed stops the line from reflowing word by word.
# TODO: BASE_FONT_SIZE는 임시 값 — Netflix Timed Text Style Guide, BBC Subtitle
#       Guidelines, WCAG 등 접근성 표준 조사 후 기획서 §다음 논의 필요 사항에 따라 최종값 확정 필요.
BASE_FONT_SIZE = 44

# Normalised volume (0–1) at or above which a word counts as shouted, and at or
# below which it counts as whispered.
# TODO: 임계값 0.7/0.3은 경험값 — 실제 영상으로 분류 결과를 확인한 뒤 조정 필요.
LOUD_THRESHOLD = 0.7
QUIET_THRESHOLD = 0.3


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
