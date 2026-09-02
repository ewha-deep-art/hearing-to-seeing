# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hearing to Seeing** converts audio characteristics (speaker identity, timing, volume) into dynamic/kinetic subtitles — color-coded by speaker, word-by-word sync animations, and volume-scaled font sizes — targeting deaf and hard-of-hearing users. Output format is ASS (Advanced SubStation Alpha), chosen for its animation/formatting capabilities.

## Commands

This project uses [`uv`](https://docs.astral.sh/uv/) for package management (Python 3.12).

```bash
# Install dependencies
uv sync

# Run the CLI entry point
uv run hearing-to-seeing

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/path/to/test_file.py

# Add a dependency
uv add <package>
```

## Architecture

The pipeline flows in this order:

```
Audio/Video input
  → STT + Speaker Diarization (WhisperX)     [src/hearing_to_seeing/stt.py]
  → Schema / intermediate JSON               [src/hearing_to_seeing/schema.py]
  → Speaker color mapping                    [src/hearing_to_seeing/speaker.py]
  → Word-level sync timing                   [src/hearing_to_seeing/sync.py]
  → Volume → font size scaling               [src/hearing_to_seeing/volume.py]
  → ASS subtitle generation                  [src/hearing_to_seeing/converter/ass.py]
  → Pipeline orchestration                   [src/hearing_to_seeing/pipeline.py]
```

The **intermediate JSON schema** (`schema.py`) is the central data contract: it carries per-word entries with `speaker`, `text`, `start`/`end` timestamps, and `amplitude`. All downstream modules (speaker color, sync, volume) consume this schema and annotate it before the ASS converter renders the final file.

### Key design decisions
- **WhisperX** is used for STT + forced alignment (word-level timestamps at ±50ms precision) + speaker diarization in a single pass.
- **ASS format** is the primary output (over VTT/SRT) because it natively supports per-dialogue color, font size, and `\k` karaoke-style fill animations needed for the sync effect.
- The three visual effects map directly to audio signals: speaker identity → color, word timestamp → fill animation, amplitude → font size.

### Directory layout
- `src/hearing_to_seeing/` — core pipeline package + CLI entry point (`main()`)
- `src/hearing_to_seeing/converter/` — output format converters (ASS, future VTT)
- `src/web/` — future web UI (not yet implemented)
- `data/input/` / `data/output/` — local test media files (not committed)
- `tests/` — test suite (not yet populated)
- `docs/` — product spec (Korean)
