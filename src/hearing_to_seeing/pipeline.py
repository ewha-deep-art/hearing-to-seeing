import json

import numpy as np

from hearing_to_seeing.schema import Transcript
from hearing_to_seeing.stt import transcribe
from hearing_to_seeing.design.volume import annotate_volumes
from hearing_to_seeing.converter.ass import write_ass


def decode_audio(wav_path: str) -> tuple[int, np.ndarray]:
    """Reads `wav_path` to mono PCM for the volume step.

    Returns the sample rate reported by the WAV header alongside the samples,
    so callers index by time without assuming a particular rate.
    """
    from scipy.io import wavfile

    rate, audio_data = wavfile.read(wav_path)
    return rate, audio_data


def run(
    wav_path: str,
    output_ass: str,
    language: str | None = None,
    output_json: str | None = None,
) -> Transcript:
    transcript = transcribe(wav_path, language=language)

    sample_rate, audio_data = decode_audio(wav_path)
    annotate_volumes(transcript, sample_rate, audio_data)

    # Persist the intermediate schema (schema.py) — the central data contract —
    # right before handing it to the ASS converter, so it can be inspected
    # independently of the final rendered output.
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)

    write_ass(transcript, output_ass)
    return transcript
