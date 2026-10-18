"""Speaker labels checked against voice prints (WeSpeaker ResNet34, ONNX, CPU).

`merge_labels` joins same-voice labels when there is no review;
`reassign_utterances` corrects the review. Without the model the labels stay
as they are. Measurements: evaluation/README.md 5.3.
"""

import os
import sys
import urllib.request
from functools import lru_cache

import numpy as np

from hearing_to_seeing.review import UTTERANCE_PAUSE
from hearing_to_seeing.schema import Transcript, WordEntry

MODEL_URL = (
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/"
    "voxceleb_resnet34_LM.onnx"
)
MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "hearing-to-seeing", "voxceleb_resnet34_LM.onnx")
RATE = 16_000

# Shorter stretches give no usable voice print.
MIN_EMBED_SECONDS = 0.4
# Two labels are one voice from this voice-print similarity up.
MERGE_SIMILARITY = 0.65
# An utterance this long moves when another speaker's voice is closer by this margin.
REASSIGN_SECONDS = 1.0
REASSIGN_MARGIN = 0.15
# Speech a speaker needs for a reliable voice print.
MIN_SPEAKER_SECONDS = 3.0
REASSIGN_ROUNDS = 3


@lru_cache(maxsize=1)
def _session():
    import onnxruntime

    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
        os.replace(MODEL_PATH + ".part", MODEL_PATH)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = min(8, os.cpu_count() or 1)
    return onnxruntime.InferenceSession(MODEL_PATH, options, providers=["CPUExecutionProvider"])


def _mono_16k(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    """Mono samples at 16 kHz on the int16 scale the model was trained on."""
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.floating):
        audio = audio * 32768.0
    elif audio.dtype != np.int16:
        audio = audio / (np.iinfo(audio.dtype).max / 32767.0)
    if sample_rate != RATE:
        from scipy.signal import resample_poly

        g = np.gcd(sample_rate, RATE)
        audio = resample_poly(audio, RATE // g, sample_rate // g)
    return audio.astype(np.float32)


def _fbank(samples: np.ndarray) -> np.ndarray:
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.samp_freq = RATE
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(RATE, samples.tolist())
    fbank.input_finished()
    feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
    return feats - feats.mean(axis=0, keepdims=True)


def _embed(samples: np.ndarray, start: float, end: float) -> np.ndarray | None:
    """The unit-length voice print of a stretch, or None when it is too short."""
    if end - start < MIN_EMBED_SECONDS:
        return None
    feats = _fbank(samples[int(start * RATE): int(end * RATE)])[None].astype(np.float32)
    return _unit(_session().run(None, {"feats": feats})[0][0])


def _utterances(words: list[WordEntry]) -> list[list[WordEntry]]:
    """One label's words with no pause longer than UTTERANCE_PAUSE."""
    utterances: list[list[WordEntry]] = []
    for w in words:
        prev = utterances[-1][-1] if utterances else None
        if prev is None or prev.speaker != w.speaker or w.start - prev.end > UTTERANCE_PAUSE:
            utterances.append([])
        utterances[-1].append(w)
    return utterances


def _voices(transcript: Transcript, sample_rate: int, audio: np.ndarray):
    """Utterances, their voice prints (None if too short) and lengths; None without the model."""
    utterances = _utterances(transcript.words)
    try:
        samples = _mono_16k(sample_rate, audio)
        prints = [_embed(samples, u[0].start, u[-1].end) for u in utterances]
    except Exception as exc:  # no model, no network — keep the labels
        print(f"… 목소리 비교 생략: {exc}", file=sys.stderr, flush=True)
        return None
    return utterances, prints, [u[-1].end - u[0].start for u in utterances]


def _label_sums(labels, prints, seconds) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Each label's length-weighted sum of voice prints, and its seconds of speech."""
    sums: dict[str, np.ndarray] = {}
    totals: dict[str, float] = {}
    for label, p, s in zip(labels, prints, seconds):
        if p is not None:
            sums[label] = sums.get(label, 0) + p * s
            totals[label] = totals.get(label, 0.0) + s
    return sums, totals


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def merge_labels(transcript: Transcript, sample_rate: int, audio: np.ndarray) -> dict[str, str]:
    """Merges labels with one voice; returns merged label → kept label.

    Average linkage on length-weighted prints, until no pair reaches MERGE_SIMILARITY.
    """
    voices = _voices(transcript, sample_rate, audio)
    if voices is None:
        return {}
    utterances, prints, seconds = voices
    sums, _ = _label_sums([u[0].speaker for u in utterances], prints, seconds)

    merged: dict[str, str] = {}
    while True:
        labels = sorted(sums)
        best = None
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                similarity = float(_unit(sums[a]) @ _unit(sums[b]))
                if similarity >= MERGE_SIMILARITY and (best is None or similarity > best[0]):
                    best = (similarity, a, b)
        if best is None:
            break
        _, keep, gone = best
        sums[keep] = sums[keep] + sums.pop(gone)
        merged = {label: keep if to == gone else to for label, to in merged.items()}
        merged[gone] = keep
    for w in transcript.words:
        w.speaker = merged.get(w.speaker, w.speaker)
    if merged:
        print(f"… 목소리가 같은 화자 병합: {merged}", file=sys.stderr, flush=True)
    return merged


def reassign_utterances(transcript: Transcript, sample_rate: int, audio: np.ndarray) -> int:
    """Moves utterances whose voice is clearly another speaker's; returns how many.

    Its own label's print is taken without it; repeats up to REASSIGN_ROUNDS.
    """
    voices = _voices(transcript, sample_rate, audio)
    if voices is None:
        return 0
    utterances, prints, seconds = voices
    labels = [u[0].speaker for u in utterances]
    moved = 0
    for _ in range(REASSIGN_ROUNDS):
        sums, totals = _label_sums(labels, prints, seconds)
        changes = {}
        for k, (label, p, s) in enumerate(zip(labels, prints, seconds)):
            if p is None or s < REASSIGN_SECONDS:
                continue
            scores = {}
            for other, total in sums.items():
                rest = total - (p * s if other == label else 0)
                if totals[other] - (s if other == label else 0) >= MIN_SPEAKER_SECONDS:
                    scores[other] = float(p @ _unit(rest))
            if label not in scores:
                continue
            best = max(scores, key=scores.get)
            if best != label and scores[best] - scores[label] > REASSIGN_MARGIN:
                changes[k] = best
        if not changes:
            break
        for k, label in changes.items():
            labels[k] = label
        moved += len(changes)
    for u, label in zip(utterances, labels):
        for w in u:
            w.speaker = label
    if moved:
        print(f"… 목소리로 발화 {moved}개 화자 수정", file=sys.stderr, flush=True)
    return moved
