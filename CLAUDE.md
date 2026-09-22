# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hearing to Seeing** converts audio characteristics (speaker identity, timing, volume) into dynamic/kinetic subtitles for deaf and hard-of-hearing users. See [README.md](README.md) for the full project description.

## Commands

See [README.md](README.md) for installation, running the pipeline, and test commands. This project uses [`uv`](https://docs.astral.sh/uv/) for package management (Python 3.12).

## Architecture

The pipeline flows in this order:

```
Audio/Video input + work metadata
  → Input validation + media metadata        [src/hearing_to_seeing/source.py]
  → STT + Speaker Diarization (WhisperX)     [src/hearing_to_seeing/stt.py]
  → Schema / intermediate JSON               [src/hearing_to_seeing/schema.py]
  → Work + character lookup (--lookup)       [src/hearing_to_seeing/knowledge/]
  → Character → preferred hue                [src/hearing_to_seeing/design/character_color.py]
  → Speaker → character mapping              [src/hearing_to_seeing/design/identity.py (manual/stored/layered),
                                              src/hearing_to_seeing/design/inference.py (text inference)]
  → Speaker color mapping                    [src/hearing_to_seeing/design/speaker.py]
  → Word-level sync timing                   [src/hearing_to_seeing/design/sync.py]
  → Volume → font size scaling               [src/hearing_to_seeing/design/volume.py]
  → ASS subtitle generation                  [src/hearing_to_seeing/converter/ass.py]
  → Pipeline orchestration                   [src/hearing_to_seeing/pipeline.py]
```

The **intermediate JSON schema** (`schema.py`) is the central data contract: it carries per-word entries with `speaker`, `text`, `start`/`end` timestamps, and `amplitude`, plus a `MediaInfo` block (title, source URL) taken as CLI input, a `WorkInfo` block (the identified work, its cast as `Character`s with per-character hue and the evidence for it) written by the lookup and colour steps, a `SpeakerProfile` per speaker (colour, resolved name, confidence, source) written by the speaker step, and a `character_map` (label → name) holding the confirmed mapping. All downstream modules (speaker color, sync, volume) consume this schema and annotate it before the ASS converter renders the final file. A transcript that already carries `work` is not looked up again, and a `character_map` it carries is replayed as certain (the `stored` source) — which is how a person corrects a mapping: edit that block and rerun `from-json`. `--fresh` discards both.

### Key design decisions
- **WhisperX** is used for STT + forced alignment (word-level timestamps) + speaker diarization in a single pass.
- **Speaker colour is one fixed OKLCH lightness with hue left free** (`design/oklch.py`). Fixing lightness settles legibility once for every colour, which is what allows hue to be a whole circle rather than a short palette; chroma is a ratio of the per-hue gamut maximum, and its two tiers mark identified characters apart from unidentified ones.
- **Colour is decided behind a strategy seam.** A strategy (`SpeakerStrategy`) returns a `SpeakerCandidate` per speaker — a name, a confidence, and a *preferred hue* — and never a colour. `design/speaker.resolve_speakers()` places the hues, keeping every speaker a minimum arc apart, so no strategy can give two speakers the same colour or one that cannot be read on video. Candidates below `MIN_CONFIDENCE` are dropped; with no strategy at all the speakers are simply spread evenly around the circle.
- **Hue preferences belong to characters, not speakers.** The lookup step (`knowledge/`) identifies the work from the clip title, fetches Korean Wikipedia and Namu wiki articles itself (`retrieve.py`, no model, sections cut to a budget) and has Gemini (free tier) read them into a cast; `design/character_color.py` then asks once for the whole cast and gets a costume colour and a symbolic colour per character as sRGB hex. The tiering is done in code on OKLCH chroma: costume if it is vivid enough (`ACHROMATIC_CHROMA`), else symbolic, else no preference. `with_character_hues()` wraps whatever identity strategy is in force so a named speaker inherits their character's hue. All model calls go through the `knowledge/llm.LLM` protocol — `structured(system, prompt, schema, web_search=)` returning one schema-conforming dict — and run only under `--lookup`. `Gemini` (`knowledge/llm.py`, default, `GEMINI_API_KEY`) is the shipped backend; `Claude` (`knowledge/claude.py`, `H2S_LLM_PROVIDER=claude`, `uv sync --extra claude`) is the alternative. There is no web-search variant: the free tier has no search-grounding quota, so retrieval is our own and the documents go inside the prompt. Default model `gemini-3.6-flash`; a 429 fails over immediately and a 503 after short SDK retries to `gemini-3.5-flash-lite`; the whole pipeline is four model requests per clip.
- **Identity strategies stack, most trusted first.** `identity.layered(manual, stored, inferred)`: the first strategy to name a label decides it, and later strategies are not consulted once every speaker is named (the inferred one costs a model call). `design/inference.py` is the automatic path: rule-based name-calling anchors (who said a character's name, who answered right after) are counted from the transcript and handed to one model call as constraints alongside the cast sheet and the dialogue; the model may leave a speaker unnamed, and `resolve_speakers()` drops anything under `MIN_CONFIDENCE`.
- **ASS format** is the primary output (over VTT/SRT) because it natively supports per-dialogue color, font size, and `\k` karaoke-style fill animations needed for the sync effect.
- The three visual effects map directly to audio signals: speaker identity → color, word timestamp → fill animation, amplitude → font size.

### Directory layout
- `src/hearing_to_seeing/` — core pipeline package + CLI entry point (`cli.py`: `run`, `from-json`, and `lookup` — the identification steps alone, from a transcript with no media); `source.py` validates the input file and carries the work metadata (`--title`, `--url`). `pipeline.identify()` is the media-free part (lookup → hues → mapping) that both `_finish()` and `run_lookup()` call.
- `src/hearing_to_seeing/knowledge/` — work + character lookup: clip-title parsing (`title.py`), document retrieval (`retrieve.py`), cast extraction (`characters.py`), the model wrappers (`llm.py` Gemini + protocol, `claude.py` optional)
- `src/hearing_to_seeing/design/` — speaker identity (`identity.py` manual/stored/layered, `inference.py` text inference), character→hue (`character_color.py`), speaker color, sync timing, and volume→font-size modules
- `src/hearing_to_seeing/converter/` — output format converters (ASS, future VTT)
- `src/web/` — preview renderer (`render.py`) that burns the generated ASS onto the source media
- `data/input/` / `data/output/` — local test media and transcripts in, generated `.ass` / `.json` out (default output location for every CLI command; neither is committed)
- `tests/` — test suite
- `docs/` — product spec (Korean) + `PIPELINE.md` (end-to-end design, per-step status) + `TODO.md` (unresolved decisions)

## Git Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching strategy and commit message conventions. Always follow it when creating branches or commits.
