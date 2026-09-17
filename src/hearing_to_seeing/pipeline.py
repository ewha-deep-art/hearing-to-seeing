# pipeline.py
import io
import json
import subprocess

import numpy as np

from hearing_to_seeing.schema import Transcript
from hearing_to_seeing.stt import load_models, transcribe
from hearing_to_seeing.design.volume import annotate_volumes
from hearing_to_seeing.converter.ass import write_ass


def _extract_audio(media_path: str) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

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


def transcribe_to_json(media_path: str, json_path: str, language: str | None = None) -> Transcript:
    """무거운 STT+화자분리+음량분석을 한 번 실행하고 결과를 JSON으로 저장."""
    models = load_models(language=language or "ko")
    transcript = transcribe(media_path, models=models, language=language)

    sample_rate, audio_data = _extract_audio(media_path)
    annotate_volumes(transcript, sample_rate, audio_data)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)

    return transcript


def render_from_json(json_path: str, output_ass: str) -> Transcript:
    """저장된 JSON에서 ASS만 빠르게 재생성 (STT 재실행 없음)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    transcript = Transcript.from_dict(data)
    write_ass(transcript, output_ass)
    return transcript


def run(media_path: str, output_ass: str, language: str | None = None) -> Transcript:
    """기존처럼 한 번에 다 돌리고 싶을 때 쓰는 편의 함수."""
    json_path = output_ass.rsplit(".", 1)[0] + ".json"
    transcribe_to_json(media_path, json_path, language=language)
    return render_from_json(json_path, output_ass)