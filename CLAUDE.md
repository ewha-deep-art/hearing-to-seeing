# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hearing to Seeing** converts audio characteristics (speaker identity, timing, volume) into dynamic/kinetic subtitles for deaf and hard-of-hearing users. See [README.md](README.md) for the full project description.

## Commands

See [README.md](README.md) for installation, running the pipeline, and test commands. This project uses [`uv`](https://docs.astral.sh/uv/) for package management (Python 3.12).

## Architecture

The pipeline flows in this order:

```
Audio/Video input
  → STT + Speaker Diarization (WhisperX)     [src/hearing_to_seeing/stt.py]
  → Schema / intermediate JSON               [src/hearing_to_seeing/schema.py]
  → Speaker color mapping                    [src/hearing_to_seeing/design/speaker.py]
  → Word-level sync timing                   [src/hearing_to_seeing/design/sync.py]
  → Volume → font size scaling               [src/hearing_to_seeing/design/volume.py]
  → ASS subtitle generation                  [src/hearing_to_seeing/converter/ass.py]
  → Pipeline orchestration                   [src/hearing_to_seeing/pipeline.py]
```

The **intermediate JSON schema** (`schema.py`) is the central data contract: it carries per-word entries with `speaker`, `text`, `start`/`end` timestamps, and `amplitude`. All downstream modules (speaker color, sync, volume) consume this schema and annotate it before the ASS converter renders the final file.

### Key design decisions
- **WhisperX** is used for STT + forced alignment (word-level timestamps) + speaker diarization in a single pass.
- **ASS format** is the primary output (over VTT/SRT) because it natively supports per-dialogue color, font size, and `\k` karaoke-style fill animations needed for the sync effect.
- The three visual effects map directly to audio signals: speaker identity → color, word timestamp → fill animation, amplitude → font size.

### Directory layout
- `src/hearing_to_seeing/` — core pipeline package + CLI entry point (`cli.py`)
- `src/hearing_to_seeing/design/` — speaker color, sync timing, and volume→font-size modules
- `src/hearing_to_seeing/converter/` — output format converters (ASS, future VTT)
- `src/web/` — preview renderer (`render.py`) that burns the generated ASS onto the source media
- `data/input/` / `data/output/` — local test media files (not committed)
- `tests/` — test suite
- `docs/` — product spec (Korean) + `TODO.md` (unresolved decisions)

## Git Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching strategy and commit message conventions. Always follow it when creating branches or commits.
