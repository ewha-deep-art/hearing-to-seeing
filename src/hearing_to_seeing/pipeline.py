import json

from hearing_to_seeing.media import decode_audio
from hearing_to_seeing.schema import Transcript
from hearing_to_seeing.design.volume import annotate_volumes
from hearing_to_seeing.converter.ass import write_ass


def _finish(
    transcript: Transcript,
    media_path: str,
    output_ass: str,
    output_json: str | None,
) -> Transcript:
    """Annotates volume and writes the outputs — the tail both entry points share."""
    sample_rate, audio_data = decode_audio(media_path)
    annotate_volumes(transcript, sample_rate, audio_data)

    # Persist the intermediate schema (schema.py) — the central data contract —
    # right before handing it to the ASS converter, so it can be inspected
    # independently of the final rendered output.
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)

    write_ass(transcript, output_ass)
    return transcript


def run_from_json(
    media_path: str,
    transcript_json: str,
    output_ass: str,
    output_json: str | None = None,
) -> Transcript:
    """Builds subtitles from an existing transcript instead of re-running STT.

    A stopgap for machines without a usable GPU: WhisperX's STT, forced
    alignment and diarization are the only steps that need one, so this entry
    point takes their output as a file — produced elsewhere, or by an external
    transcription service — and runs the rest of the pipeline locally on CPU.
    `run()` remains the real entry point wherever a GPU is available.

    Accepts either the intermediate schema written by `run()` or a
    character-level timestamp export. Volume is (re)derived from `media_path`,
    since neither format is guaranteed to carry it. Note that a character-level
    export carries no speaker labels, so every word falls back to the default
    speaker and the colour effect is lost.
    """
    with open(transcript_json, encoding="utf-8") as f:
        data = json.load(f)

    if "characters" in data:
        transcript = Transcript.from_char_timestamps(data)
    else:
        transcript = Transcript.from_dict(data)

    return _finish(transcript, media_path, output_ass, output_json)


def run(
    media_path: str,
    output_ass: str,
    language: str | None = None,
    output_json: str | None = None,
) -> Transcript:
    # Imported here rather than at module scope so run_from_json() — which
    # needs no STT — does not pay for loading torch and whisperx.
    from hearing_to_seeing.stt import load_models, transcribe

    models = load_models(language=language or "ko")
    transcript = transcribe(media_path, models=models, language=language)

    # TODO: 화자 분리 이후 라벨 수동 보정 기능 없음 — 기획서 §10에서 화자 오분류에 대한
    #       수동 보정 옵션을 리스크 대응방안으로 제시했으나 미구현.
    return _finish(transcript, media_path, output_ass, output_json)
