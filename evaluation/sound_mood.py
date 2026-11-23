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
"""Can AudioSet's music-mood classes name the music ([경쾌한 음악])?

    uv run evaluation/sound_mood.py                      # data/input/stt_eval/*.wav
    uv run evaluation/sound_mood.py path/to/a.wav ...

PANNs Cnn14 clip-level tagger (not the frame-level one in sound_events.py)
over 10 s windows, hop 5 s, on the CPU; no LLM calls. Prints, per clip, the
peak score of Music, the seven mood classes and a few events, then the top
mood of every window where Music >= --music.

The checkpoint is not fetched by panns_inference's default (that is the
frame-level model); download it once:

    wget -O ~/panns_data/Cnn14_mAP=0.431.pth \
      "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

Result on the four test clips (2026-11-24): mood scores stay below 0.04, so
the classes are unusable as-is; only their ranking looks plausible.
"""

import argparse
from pathlib import Path

import numpy as np

SR = 32000
WINDOW, HOP = 10, 5
CHECKPOINT = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
DEFAULT_INPUTS = sorted(
    p for p in Path("data/input/stt_eval").glob("*.wav") if p.stem.count(".") == 0
)
MOODS = [
    "Happy music", "Funny music", "Sad music", "Tender music",
    "Exciting music", "Angry music", "Scary music",
]
SHOWN = ["Music", *MOODS, "Laughter", "Applause", "Glass", "Shatter"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--music", type=float, default=0.5, help="window counts as music")
    args = ap.parse_args()

    import librosa
    from panns_inference import AudioTagging, labels

    tagger = AudioTagging(checkpoint_path=str(CHECKPOINT), device="cpu")
    shown = [labels.index(n) for n in SHOWN]
    moods = [labels.index(m) for m in MOODS]
    music = labels.index("Music")

    for wav in args.inputs:
        audio, _ = librosa.load(wav, sr=SR, mono=True)
        starts = list(range(0, max(1, len(audio) - WINDOW * SR), HOP * SR))
        out, _ = tagger.inference(np.stack([audio[s : s + WINDOW * SR] for s in starts]))
        print(f"\n=== {wav.stem}  ({len(starts)} windows)")
        print("peak  " + "  ".join(f"{n.split()[0]}={out[:, i].max():.2f}" for n, i in zip(SHOWN, shown)))
        on = out[:, music] >= args.music
        print(f"music windows {on.sum()}/{len(starts)}")
        for k, s in enumerate(starts):
            if on[k]:
                j = int(np.argmax(out[k, moods]))
                print(f"  {s / SR:6.0f}s music={out[k, music]:.2f}  top mood {MOODS[j]} {out[k, moods[j]]:.3f}")


if __name__ == "__main__":
    main()
