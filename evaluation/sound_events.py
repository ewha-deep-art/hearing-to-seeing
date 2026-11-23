# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "librosa>=0.10",
#     "numpy>=1.26",
#     "panns-inference>=0.1.1",
#     "torch>=2.4",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Explore what a local sound event detector hears in the test clips.

    uv run evaluation/sound_events.py                    # data/input/stt_eval/*.wav
    uv run evaluation/sound_events.py path/to/a.wav ...
    uv run evaluation/sound_events.py --threshold 0.4 --top 50

PANNs Cnn14_DecisionLevelMax (AudioSet, 527 classes, 100 frames/s) on the
CPU; no LLM calls. The frame scores are cached per clip as
data/output/sound_events/<stem>.npz, so changing the thresholds only re-reads
them. For each clip it writes <stem>.events.txt (every detected event, in time
order) and prints the classes ranked by peak score, then a summary across
clips — the material for choosing a whitelist and per-class thresholds.
"""

import argparse
from pathlib import Path

import numpy as np

SR = 32000
FPS = 100  # hop 320 at 32 kHz
CHUNK_SECONDS = 60
OUT_DIR = Path("data/output/sound_events")
DEFAULT_INPUTS = sorted(
    p for p in Path("data/input/stt_eval").glob("*.wav") if p.stem.count(".") == 0
)

# Classes that never become a caption: speech itself, room/recording
# character, generic noise.
EXCLUDE = {
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Speech synthesizer", "Babbling", "Hubbub, speech noise, speech babble",
    "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, urban or manmade", "Outside, rural or natural",
    "Reverberation", "Echo", "Noise", "Environmental noise", "Static",
    "Mains hum", "Distortion", "Sidetone", "White noise", "Pink noise",
    "Silence", "Sine wave", "Harmonic", "Pulse", "Sound effect",
    "Effects unit", "Chorus effect", "Field recording", "Television", "Radio",
}
MOODS = [
    "Happy music", "Funny music", "Sad music", "Tender music",
    "Exciting music", "Angry music", "Scary music",
]


def labels() -> list[str]:
    from panns_inference import labels as panns_labels

    return list(panns_labels)


def frame_scores(wav: Path) -> np.ndarray:
    """(frames, 527) scores, cached."""
    cache = OUT_DIR / f"{wav.stem}.npz"
    if cache.exists():
        return np.load(cache)["scores"].astype(np.float32)
    import librosa
    from panns_inference import SoundEventDetection

    audio, _ = librosa.load(wav, sr=SR, mono=True)
    sed = SoundEventDetection(checkpoint_path=None, device="cpu")
    step = CHUNK_SECONDS * SR
    parts = []
    for i in range(0, len(audio), step):
        chunk = audio[i : i + step]
        out = sed.inference(chunk[None, :])  # (1, frames, 527)
        parts.append(out[0][: int(np.ceil(len(chunk) / SR * FPS))])
    scores = np.concatenate(parts)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, scores=scores.astype(np.float16))
    return scores


def events(track: np.ndarray, on: float, off: float, min_s: float, gap_s: float):
    """Hysteresis segments of one class: [(start_s, end_s, peak)]."""
    segs, start = [], None
    for t, v in enumerate(track):
        if start is None and v >= on:
            start = t
            # walk back to where it rose above `off`
            while start > 0 and track[start - 1] >= off:
                start -= 1
        elif start is not None and v < off:
            segs.append([start, t])
            start = None
    if start is not None:
        segs.append([start, len(track)])
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= gap_s * FPS:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [
        (s / FPS, e / FPS, float(track[s:e].max()))
        for s, e in merged
        if (e - s) >= min_s * FPS
    ]


def clock(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:05.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--threshold", type=float, default=0.3, help="event onset score")
    ap.add_argument("--off", type=float, default=0.15, help="event release score")
    ap.add_argument("--min", type=float, default=0.2, help="min event seconds")
    ap.add_argument("--gap", type=float, default=0.3, help="merge gaps up to (s)")
    ap.add_argument("--top", type=int, default=30, help="classes printed per clip")
    args = ap.parse_args()

    names = labels()
    music = names.index("Music")
    speech = names.index("Speech")
    overall: dict[str, dict] = {}

    for wav in args.inputs:
        scores = frame_scores(wav)
        seconds = len(scores) / FPS
        print(f"\n=== {wav.stem}  ({clock(seconds)}) ===")
        print(
            f"speech≥{args.threshold}: {np.mean(scores[:, speech] >= args.threshold):.0%}"
            f"  music≥{args.threshold}: {np.mean(scores[:, music] >= args.threshold):.0%}"
            f"  both: {np.mean((scores[:, speech] >= args.threshold) & (scores[:, music] >= args.threshold)):.0%}"
        )

        found = []
        for k, name in enumerate(names):
            if name in EXCLUDE:
                continue
            evs = events(scores[:, k], args.threshold, args.off, args.min, args.gap)
            if not evs:
                continue
            # how much of the event time is also speech
            over = np.mean(
                [scores[int(s * FPS) : int(e * FPS), speech].mean() for s, e, _ in evs]
            )
            found.append((name, evs, over))
            o = overall.setdefault(name, {"clips": 0, "events": 0, "peak": 0.0, "sec": 0.0})
            o["clips"] += 1
            o["events"] += len(evs)
            o["peak"] = max(o["peak"], max(p for _, _, p in evs))
            o["sec"] += sum(e - s for s, e, _ in evs)

        found.sort(key=lambda f: -max(p for _, _, p in f[1]))
        print(f"{'class':38} {'n':>3} {'sec':>6} {'peak':>5} {'speech':>6}  first")
        for name, evs, over in found[: args.top]:
            print(
                f"{name[:38]:38} {len(evs):3d} {sum(e - s for s, e, _ in evs):6.1f}"
                f" {max(p for _, _, p in evs):5.2f} {over:6.2f}  {clock(evs[0][0])}"
            )

        # music mood: averaged over frames where music is on
        on = scores[:, music] >= args.threshold
        if on.any():
            mood = {m: float(scores[on, names.index(m)].mean()) for m in MOODS}
            print("mood (mean while music):", "  ".join(
                f"{m.split()[0]} {v:.2f}" for m, v in sorted(mood.items(), key=lambda x: -x[1])
            ))

        timeline = sorted(
            (s, e, p, name) for name, evs, _ in found for s, e, p in evs
        )
        log = OUT_DIR / f"{wav.stem}.events.txt"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "".join(
                f"{clock(s)}–{clock(e)}  {p:.2f}  sp={scores[int(s * FPS):int(e * FPS), speech].mean():.2f}  {name}\n"
                for s, e, p, name in timeline
            )
        )
        print(f"→ {log} ({len(timeline)} events)")

    print(f"\n=== all clips: classes by peak (threshold {args.threshold}) ===")
    print(f"{'class':38} {'clips':>5} {'n':>4} {'sec':>6} {'peak':>5}")
    for name, o in sorted(overall.items(), key=lambda x: -x[1]["peak"]):
        print(f"{name[:38]:38} {o['clips']:5d} {o['events']:4d} {o['sec']:6.1f} {o['peak']:5.2f}")


if __name__ == "__main__":
    main()
