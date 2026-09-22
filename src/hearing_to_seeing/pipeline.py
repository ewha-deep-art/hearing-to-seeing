import json

from hearing_to_seeing.media import decode_audio
from hearing_to_seeing.schema import MediaInfo, Transcript
from hearing_to_seeing.design.character_color import suggest_hues, with_character_hues
from hearing_to_seeing.design.identity import layered, stored_mapping
from hearing_to_seeing.design.inference import infer_from_text
from hearing_to_seeing.design.speaker import SpeakerStrategy, resolve_speakers
from hearing_to_seeing.design.volume import annotate_volumes
from hearing_to_seeing.converter.ass import write_ass
from hearing_to_seeing.knowledge import LLM, lookup, progress


def _load(transcript_json: str) -> Transcript:
    """Reads any of the three transcript shapes `from-json` accepts."""
    with open(transcript_json, encoding="utf-8") as f:
        data = json.load(f)

    if "characters" in data:
        return Transcript.from_char_timestamps(data)
    if "segments" in data and "words" not in data:
        return Transcript.from_segments(data)
    return Transcript.from_dict(data)


def _write_json(transcript: Transcript, output_json: str) -> None:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)


def identify(
    transcript: Transcript,
    media_path: str,
    speaker_strategy: SpeakerStrategy | None = None,
    llm: LLM | None = None,
    fresh: bool = False,
) -> Transcript:
    """Decides who each speaker is and what colour they get — no audio needed.

    Lookup (which work, who is in it, what colour suits each), then the
    identity strategies stacked most-trusted first: `speaker_strategy` is the
    user's own word (the manual mapping); under it a mapping the transcript
    already carries; under that the inferred one, consulted only for labels
    the first two left unnamed. What survives the confidence threshold is
    written back as `character_map`.
    """
    if fresh:
        # Forget what an earlier run concluded and redo lookup and mapping;
        # the transcript itself (words, timings, speakers) is kept.
        transcript.work = None
        transcript.character_map = {}

    # Lookup only with a model to ask — nothing is fetched otherwise — and
    # only once per clip, since a stored transcript carries the answer forward.
    work = transcript.work
    inferred = None
    if llm is not None:
        work = lookup(transcript, llm)
        if work is not None and work.characters:
            if any(c.hue_basis is None for c in work.characters):
                progress("인물별 색상 근거 결정 중")
            suggest_hues(work, llm)
            inferred = infer_from_text(work, llm)

    stored = stored_mapping(transcript.character_map) if transcript.character_map else None
    strategy = layered(speaker_strategy, stored, inferred)
    strategy = with_character_hues(strategy, work)

    candidates = strategy(transcript, media_path) if strategy else None
    transcript.speaker_profiles = resolve_speakers(transcript, candidates)
    # What survived the confidence threshold is the confirmed mapping: the
    # block a person edits, and the one the next run replays.
    transcript.character_map = {
        label: profile.name
        for label, profile in transcript.speaker_profiles.items()
        if profile.name
    }
    return transcript


def _finish(
    transcript: Transcript,
    media_path: str,
    output_ass: str,
    output_json: str | None,
    media_info: MediaInfo | None = None,
    speaker_strategy: SpeakerStrategy | None = None,
    llm: LLM | None = None,
    fresh: bool = False,
) -> Transcript:
    """Annotates volume, identifies speakers and writes the outputs — the tail both entry points share."""
    if media_info is not None:
        # Field by field, so an explicit --title does not discard a URL and
        # channel that a stored transcript already carried.
        transcript.media = media_info.merge(transcript.media)

    sample_rate, audio_data = decode_audio(media_path)
    annotate_volumes(transcript, sample_rate, audio_data)

    # Before the JSON is written, so the reasoning behind each colour is in the
    # file rather than recomputed from scratch by whatever reads it next.
    identify(transcript, media_path, speaker_strategy, llm, fresh)

    # Persist the intermediate schema (schema.py) — the central data contract —
    # right before handing it to the ASS converter, so it can be inspected
    # independently of the final rendered output.
    if output_json:
        _write_json(transcript, output_json)

    write_ass(transcript, output_ass)
    return transcript


def run_lookup(
    transcript_json: str,
    output_json: str,
    output_ass: str | None = None,
    media_info: MediaInfo | None = None,
    speaker_strategy: SpeakerStrategy | None = None,
    llm: LLM | None = None,
    fresh: bool = False,
) -> Transcript:
    """Lookup, colour and speaker mapping from a transcript alone — no media.

    For working on the identification steps without a video to hand: the
    volume step is skipped (every word keeps whatever volume the file had,
    zero for a fresh export), and the result is written as JSON. An ASS file
    is written too when asked for, with the font size flat since there is no
    amplitude to scale it by.
    """
    transcript = _load(transcript_json)
    if media_info is not None:
        transcript.media = media_info.merge(transcript.media)

    identify(transcript, transcript_json, speaker_strategy, llm, fresh)

    _write_json(transcript, output_json)
    if output_ass:
        write_ass(transcript, output_ass)
    return transcript


def run_from_json(
    media_path: str,
    transcript_json: str,
    output_ass: str,
    output_json: str | None = None,
    media_info: MediaInfo | None = None,
    speaker_strategy: SpeakerStrategy | None = None,
    llm: LLM | None = None,
    fresh: bool = False,
) -> Transcript:
    """Builds subtitles from an existing transcript instead of re-running STT.

    A stopgap for machines without a usable GPU: WhisperX's STT, forced
    alignment and diarization are the only steps that need one, so this entry
    point takes their output as a file — produced elsewhere, or by an external
    transcription service — and runs the rest of the pipeline locally on CPU.
    `run()` remains the real entry point wherever a GPU is available.

    Accepts the intermediate schema written by `run()`, a segment-level export
    (`video_title`, `video_url`, `segments`), or a character-level timestamp
    export. Volume is (re)derived from `media_path`, since none of them is
    guaranteed to carry it. A segment export has utterance timings only, so
    word timings are interpolated; a character-level export carries no speaker
    labels, so every word falls back to the default speaker and the colour
    effect is lost.
    """
    transcript = _load(transcript_json)
    return _finish(
        transcript, media_path, output_ass, output_json, media_info, speaker_strategy, llm, fresh,
    )


def run(
    media_path: str,
    output_ass: str,
    language: str | None = None,
    output_json: str | None = None,
    media_info: MediaInfo | None = None,
    speaker_strategy: SpeakerStrategy | None = None,
    llm: LLM | None = None,
    fresh: bool = False,
) -> Transcript:
    # Imported here rather than at module scope so run_from_json() — which
    # needs no STT — does not pay for loading torch and whisperx.
    from hearing_to_seeing.stt import load_models, transcribe

    models = load_models(language=language or "ko")
    transcript = transcribe(media_path, models=models, language=language)

    # TODO: 화자 분리 이후 라벨 수동 보정 기능 없음 — 기획서 §10에서 화자 오분류에 대한
    #       수동 보정 옵션을 리스크 대응방안으로 제시했으나 미구현.
    return _finish(
        transcript, media_path, output_ass, output_json, media_info, speaker_strategy, llm, fresh,
    )
