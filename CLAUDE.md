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
  → Schema / intermediate JSON               [src/hearing_to_seeing/schema.py]
  → Volume → size (loud grows / whisper shrinks) [src/hearing_to_seeing/design/volume.py]
  ∥ Work/character lookup (RAG, parallel to STT) [src/hearing_to_seeing/design/speaker.py + knowledge/]
  → Speaker → character → color (OKLCH)      [src/hearing_to_seeing/design/speaker.py]
  → Timing → fill wipe + wave motion         [src/hearing_to_seeing/design/sync.py]
  → Font, size, word spacing (shared)        [src/hearing_to_seeing/design/layout.py]
  → ASS subtitle generation                  [src/hearing_to_seeing/converter/ass.py]
  → Pipeline orchestration                   [src/hearing_to_seeing/pipeline.py]
  → Preview renderer (optional)              [src/hearing_to_seeing/render.py]
```

The **intermediate JSON schema** (`schema.py`) is the central data contract: it carries per-word entries with `speaker`, `text`, `start`/`end` timestamps, and `volume`, plus `speaker_profiles` (label → color, matched character name, confidence, reason — plain dicts written by `design/speaker.py`). All downstream modules consume this schema and annotate it before the ASS converter renders the final file.

### Key design decisions
- **WhisperX** is used for STT + forced alignment (word-level timestamps) + speaker diarization in a single pass, via a remote API (`WHISPERX_API_KEY` / `WHISPERX_API_URL`).
- **ASS format** is the primary output (over VTT/SRT) because it natively supports per-dialogue color, font size, and `\k` karaoke-style fill animations needed for the sync effect.
- The three visual effects map directly to audio signals: speaker identity → color, word timestamp → fill animation, volume → motion (loud words pop bigger, whispers shrink, normal words wave character by character). Font size is fixed.
- Each effect module owns one signal and nothing else: `sync.py` timing (fill wipe, wave), `volume.py` size, `speaker.py` colour. Code that needs more than one of them goes in `layout.py`, or in `converter/ass.py` if it is about the ASS format (assembling Dialogue events).
- Each subtitle is three ASS layers: Box (opaque background), Fill (white → hidden while spoken), Pop (per-word events wiped into the speaker colour). A Pop event carries the whole line with every other word invisible, so libass places the word itself; nothing is measured.
- One bundled font (`design/fonts/`, named in `design/layout.py`). ASS finds it by the family name inside the file, so `FONT_NAME` must match it (`tests/test_layout.py` checks).
- Speaker colors: with `GEMINI_API_KEY` set, the web job looks up the work's characters (title → Wikipedia/Namu wiki → LLM cast sheet → per-character hue) in parallel with STT, then infers which speaker is which character. Without a key or on failure, hues are spread evenly. See [docs/PIPELINE.md](docs/PIPELINE.md).

### Directory layout
- `src/hearing_to_seeing/` — core pipeline package + CLI entry point (`cli.py`)
- `src/hearing_to_seeing/design/` — speaker color (incl. the RAG lookup), sync (fill/wave), volume (size), and layout (font/spacing) modules (+ bundled `fonts/`); `README.md` there documents the current design decisions
- `src/hearing_to_seeing/knowledge/` — what the speaker step calls out to: `llm.py` (Gemini `ask()`), `retrieve.py` (Wikipedia/Namu wiki), `prompts.py` (the three LLM prompts + JSON schemas)
- `src/hearing_to_seeing/converter/` — output format converters (ASS, future VTT)
- `src/hearing_to_seeing/render.py` — preview renderer that burns the generated ASS onto the source media (ffmpeg, optional; not the pipeline's real output)
- `data/input/` / `data/output/` — local test media files (not committed)
- `tests/` — test suite
- `docs/` — product spec (Korean), `PIPELINE.md` (speaker-color RAG design), `TODO.md` (unresolved decisions)

## Git Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching strategy and commit message conventions. Always follow it when creating branches or commits.
