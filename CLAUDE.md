# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hearing to Seeing** converts audio characteristics (speaker identity, timing, volume) into dynamic/kinetic subtitles for deaf and hard-of-hearing users. See [README.md](README.md) for the full project description.

## Commands

See [README.md](README.md) for installation, running the pipeline, This project uses [`uv`](https://docs.astral.sh/uv/) for package management (Python 3.12).
```
# 테스트 실행
uv run pytest

# 특정 테스트 파일만 실행
uv run pytest tests/path/to/test_file.py

# 의존성 추가
uv add <package>
```

## Architecture

The pipeline flows in this order:

```
Audio/Video input
  → STT + Speaker Diarization (remote WhisperX API) [src/hearing_to_seeing/stt.py]
  → Video review: speakers + disputed words (Gemini) [src/hearing_to_seeing/review.py]
  → Schema / intermediate JSON               [src/hearing_to_seeing/schema.py]
  → Volume → size (loud grows / whisper shrinks) [src/hearing_to_seeing/design/volume.py]
  ∥ Work/character lookup (RAG, parallel to STT) [src/hearing_to_seeing/pipeline.py + knowledge/]
  → Speaker → character → color (OKLCH)      [src/hearing_to_seeing/pipeline.py]
  → Profile colour → ASS colour              [src/hearing_to_seeing/design/speaker.py]
  → Timing → fill wipe + wave motion         [src/hearing_to_seeing/design/sync.py]
  → Font, size, word spacing (shared)        [src/hearing_to_seeing/design/layout.py]
  → ASS subtitle generation                  [src/hearing_to_seeing/converter/ass.py]
  → Pipeline orchestration                   [src/hearing_to_seeing/pipeline.py]
  → Preview renderer (optional)              [src/hearing_to_seeing/render.py]
```

The **intermediate JSON schema** (`schema.py`) is the central data contract: it carries per-word entries with `speaker`, `text`, `start`/`end` timestamps, and `volume`, plus `speaker_profiles` (label → `#RRGGBB` color, matched character name, confidence, reason — plain dicts decided in `pipeline.py`; `design/speaker.py` only converts them to ASS colours). All downstream modules consume this schema and annotate it before the ASS converter renders the final file.

### Key design decisions
- **WhisperX** is used for STT + forced alignment (word-level timestamps) + speaker diarization, via a remote API (`WHISPERX_API_KEY` / `WHISPERX_API_URL`). `stt.transcribe` decodes the audio several times — Whisper models (`large-v3`, `large-v2`, `large-v3-turbo`) × `chunk_size` windows — and merges the passes region by region, each word timed at the median and labelled by the majority of the passes that read it; only the first pass is diarized (the others borrow its labels by time) and the passes go grouped by model, starting with the one the server holds; Korean is aligned with `ALIGN_MODEL` (an unspecified language is detected first); a known speaker count (`--speakers`) goes to diarization as `num_speakers`; `parse_result` also cleans up Whisper's loops, hallucinated credits and stretched word timings. With `GEMINI_API_KEY`, `review.review_transcript` then shows Gemini the source video (or the audio, uploaded alongside STT) with the merged transcript and the spots the passes disagree on: it names each utterance's speaker by who is on screen and picks among the passes' readings — never new text — and its names replace the diarization labels and the dialogue-based speaker → character call. Character names from the RAG cast sheet respell misheard names (`correct_names`). Measurements and rationale: [docs/STT_TUNING.md](docs/STT_TUNING.md).
- **ASS format** is the primary output (over VTT/SRT) because it natively supports per-dialogue color, font size, and `\k` karaoke-style fill animations needed for the sync effect.
- The three visual effects map directly to audio signals: speaker identity → color, word timestamp → fill animation, volume → motion (loud words pop bigger, whispers shrink, normal words wave character by character). Font size is fixed.
- Each effect module owns one signal and nothing else: `sync.py` timing (fill wipe, wave), `volume.py` size, `speaker.py` colour. Code that needs more than one of them goes in `layout.py`, or in `converter/ass.py` if it is about the ASS format (assembling Dialogue events).
- Each subtitle is three ASS layers: Box (opaque background), Fill (white → hidden while spoken), Pop (per-word events wiped into the speaker colour). A Pop event carries the whole line with every other word invisible, so libass places the word itself; nothing is measured.
- One bundled font (`design/fonts/`, named in `design/layout.py`). ASS finds it by the family name inside the file, so `FONT_NAME` must match it (`tests/test_layout.py` checks).
- Speaker colors: with `GEMINI_API_KEY` set, the web job looks up the work's characters (title → Wikipedia/Namu wiki → LLM cast sheet → per-character hue) in parallel with STT, then infers which speaker is which character. Without a key or on failure, hues are spread evenly. See [src/hearing_to_seeing/knowledge/README.md](src/hearing_to_seeing/knowledge/README.md), which covers every LLM call.

### Directory layout
- `src/hearing_to_seeing/` — core pipeline package + CLI entry point (`cli.py`)
- `src/hearing_to_seeing/design/` — speaker color (profile → ASS), sync (fill/wave), volume (size), and layout (font/spacing) modules (+ bundled `fonts/`); `README.md` there documents the current design decisions
- `src/hearing_to_seeing/knowledge/` — what the pipeline's speaker step and the review call out to: `llm.py` (Gemini `ask()`), `retrieve.py` (Wikipedia/Namu wiki), `prompts.py` (the LLM prompts + JSON schemas); `README.md` there documents every LLM call (speaker-color RAG, video review)
- `src/hearing_to_seeing/converter/` — output format converters (ASS, future VTT)
- `src/hearing_to_seeing/render.py` — preview renderer that burns the generated ASS onto the source media (ffmpeg, optional; not the pipeline's real output)
- `data/input/` / `data/output/` — local test media files (not committed)
- `tests/` — test suite
- `scripts/stt_eval.py` — STT measurement on the test clips (references, passes, video review, scores)
- `docs/` — product spec (Korean), `STT_TUNING.md` (STT measurements), `TODO.md` (unresolved decisions)

## Git Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching strategy and commit message conventions. Always follow it when creating branches or commits.
