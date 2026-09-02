import io
import subprocess

import numpy as np

from hearing_to_seeing.schema import Transcript
from hearing_to_seeing.stt import load_models, transcribe
from hearing_to_seeing.design.volume import annotate_volumes
from hearing_to_seeing.converter.ass import write_ass


def _extract_audio(media_path: str) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

    # TODO: 오디오가 44100 Hz로 리샘플링되지만 WhisperX load_audio는 16000 Hz를 기대함 —
    #       올바른 샘플레이트 확인 후 ffmpeg -ar 값 수정 필요.
    cmd = [
        "ffmpeg", "-i", media_path,
        "-f", "wav", "-ac", "1", "-ar", "44100",
        "-loglevel", "quiet", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode()}")
    sample_rate, audio_data = wavfile.read(io.BytesIO(out))
    return sample_rate, audio_data


def run(media_path: str, output_ass: str, language: str | None = None) -> Transcript:
    models = load_models(language=language or "ko")
    transcript = transcribe(media_path, models=models, language=language)

    sample_rate, audio_data = _extract_audio(media_path)
    annotate_volumes(transcript, sample_rate, audio_data)

    # TODO: 화자 분리 이후 라벨 수동 보정 기능 없음 — 기획서 §10에서 화자 오분류에 대한
    #       수동 보정 옵션을 리스크 대응방안으로 제시했으나 미구현.
    write_ass(transcript, output_ass)
    return transcript
