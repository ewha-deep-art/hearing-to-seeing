import gc
import os
from dataclasses import dataclass

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from hearing_to_seeing.schema import Transcript, WordEntry


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _free(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class WhisperXModels:
    asr: object
    align: object
    align_metadata: object
    diarize: object
    device: str


def load_models(language: str = "ko") -> WhisperXModels:
    device = _device()
    compute_type = "float16" if device == "cuda" else "int8"

    asr = whisperx.load_model("large-v3", device, compute_type=compute_type)
    align, align_metadata = whisperx.load_align_model(language_code=language, device=device)
    diarize = DiarizationPipeline(
        token=os.environ.get("HF_TOKEN"),
        device=device,
    )

    return WhisperXModels(
        asr=asr,
        align=align,
        align_metadata=align_metadata,
        diarize=diarize,
        device=device,
    )

def transcribe(audio_path: str, models: WhisperXModels, language: str | None = None) -> Transcript:
    audio = whisperx.load_audio(audio_path)

    result = models.asr.transcribe(audio, batch_size=16)
    detected_lang = language or result.get("language", "en")

    aligned = whisperx.align(
        result["segments"],
        models.align,
        models.align_metadata,
        audio,
        models.device,
        return_char_alignments=False,
    )

    diarize_segments = models.diarize(audio)
    final = whisperx.assign_word_speakers(diarize_segments, aligned)

    words: list[WordEntry] = []
    for segment in final["segments"]:
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
