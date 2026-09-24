"""STT, forced alignment and speaker diarization via the remote WhisperX API.

The server runs WhisperX and returns its `assign_word_speakers` output as-is:
segments carrying per-word `start`/`end`/`speaker`. Set WHISPERX_API_KEY and
WHISPERX_API_URL in the environment or `.env`.
"""

import os

import requests
from dotenv import load_dotenv

from hearing_to_seeing.schema import Transcript, WordEntry

# The server processes roughly in real time (45 s of audio took ~50 s), so a
# long input needs a generous read timeout.
REQUEST_TIMEOUT = 1800


class STTRequestError(RuntimeError):
    pass


def _request(wav: bytes, language: str | None) -> dict:
    load_dotenv()
    api_key = os.environ.get("WHISPERX_API_KEY")
    if not api_key:
        raise STTRequestError("WHISPERX_API_KEY is not set (see .env.example).")
    url = os.environ.get("WHISPERX_API_URL")
    if not url:
        raise STTRequestError("WHISPERX_API_URL is not set (see .env.example).")

    # TODO: 화자 수 상한이 없음 — 기획서 §10에서 max_speakers 상한 설정으로
    #       유사한 목소리의 오분류를 줄이도록 권고함. API의 max_speakers 파라미터 활용 필요.
    data = {"align": "true", "diarize": "true"}
    if language:
        data["language"] = language

    try:
        response = requests.post(
            url,
            headers={"X-API-Key": api_key},
            # The server accepts only parts typed audio/*.
            files={"file": ("audio.wav", wav, "audio/wav")},
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise STTRequestError(f"WhisperX API request failed: {exc}") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise STTRequestError(f"WhisperX API returned {response.status_code}: {detail}")
    return response.json()


def transcribe(wav_path: str, language: str | None = None) -> Transcript:
    with open(wav_path, "rb") as f:
        result = _request(f.read(), language)
    detected_lang = language or result.get("language")

    words: list[WordEntry] = []
    for segment in result["segments"]:
        speaker = segment.get("speaker", "SPEAKER_00")
        for w in segment.get("words", []):
            start = w.get("start")
            end = w.get("end")
            # TODO: 타임스탬프가 없는 단어는 로그 없이 조용히 삭제됨 — 어느 구간이
            #       강제 정렬에 실패했는지 진단할 수 있도록 경고 로그 또는 플래그 처리 필요.
            if start is None or end is None:
                continue
            words.append(
                WordEntry(
                    text=w.get("word", "").strip(),
                    start=round(start, 3),
                    end=round(end, 3),
                    speaker=speaker,
                )
            )

    return Transcript(words=words, language=detected_lang)
