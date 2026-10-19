"""Speaker labels from voice prints (WeSpeaker ResNet34, ONNX, CPU).

`diarize` labels the first STT pass in place of the server's diarization;
`merge_labels` joins same-voice labels when there is no review;
`reassign_utterances` corrects the review. Without the model the labels stay
as they are. Measurements: evaluation/README.md 5.3, 5.8.
"""

import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from hearing_to_seeing.review import SENTENCE_END, UTTERANCE_PAUSE
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

# `diarize`: the server's diarization took 7–141 s a clip and was most of a
# long clip's wait; this takes a few seconds and labelled as well
# (evaluation/LOG.md 6.7). Utterances are cut where the voice on either side of
# a word boundary (CHANGE_WINDOW s each) is less alike than CHANGE_SIMILARITY,
# then those of CLUSTER_SECONDS or more are grouped (average linkage) while
# alike from CLUSTER_SIMILARITY up; shorter ones join the nearest group.
CHANGE_WINDOW = 1.5
CHANGE_SIMILARITY = 0.2
CLUSTER_SECONDS = 1.0
CLUSTER_SIMILARITY = 0.3


@lru_cache(maxsize=1)
def _session():
    import onnxruntime

    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
        os.replace(MODEL_PATH + ".part", MODEL_PATH)
    options = onnxruntime.SessionOptions()
    # Utterances are embedded in parallel (`_voices`), one thread each:
    # 2.1 s for a 5-minute clip against 3.5 s one at a time on eight threads.
    options.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(MODEL_PATH, options, providers=["CPUExecutionProvider"])


def available() -> bool:
    """Whether the model loads (downloading it the first time)."""
    try:
        _session()
        return True
    except Exception as exc:  # no network, no onnxruntime
        print(f"… 목소리 모델 없음: {exc}", file=sys.stderr, flush=True)
        return False


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
        _session()  # loaded (or downloaded) once, before the threads ask for it
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
            prints = list(pool.map(lambda u: _embed(samples, u[0].start, u[-1].end), utterances))
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


def _embed_all(samples: np.ndarray, spans: list[tuple[float, float]]) -> list[np.ndarray | None]:
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        return list(pool.map(lambda span: _embed(samples, *span), spans))


def _split_at_changes(pieces: list[list[dict]], samples: np.ndarray) -> list[list[dict]]:
    """Each piece cut at word boundaries where the voice changes, a second or more apart."""
    candidates = []  # (piece, word index, before, at, after)
    for p, words in enumerate(pieces):
        for k in range(1, len(words)):
            at = words[k]["start"]
            if at - words[0]["start"] >= 1.0 and words[-1]["end"] - at >= 1.0:
                candidates.append((p, k, max(words[0]["start"], at - CHANGE_WINDOW), at,
                                   min(words[-1]["end"], at + CHANGE_WINDOW)))
    spans = [(b, a) for _, _, b, a, _ in candidates] + [(a, c) for _, _, _, a, c in candidates]
    prints = _embed_all(samples, spans)
    cuts: dict[int, list[int]] = {}
    scored = []
    for n, (p, k, *_) in enumerate(candidates):
        left, right = prints[n], prints[n + len(candidates)]
        if left is not None and right is not None and float(left @ right) < CHANGE_SIMILARITY:
            scored.append((float(left @ right), p, k))
    for _, p, k in sorted(scored):  # the clearest changes first
        at = pieces[p][k]["start"]
        if all(abs(pieces[p][c]["start"] - at) >= 1.0 for c in cuts.get(p, [])):
            cuts.setdefault(p, []).append(k)
    out = []
    for p, words in enumerate(pieces):
        bounds = [0, *sorted(cuts.get(p, [])), len(words)]
        out += [words[a:b] for a, b in zip(bounds, bounds[1:])]
    return out


def diarize(result: dict, sample_rate: int, audio: np.ndarray, num_speakers: int | None = None) -> None:
    """Labels a server response's words by voice, in place (see CHANGE_WINDOW).

    `num_speakers`, when known, is the number of groups instead of
    CLUSTER_SIMILARITY. Words without timing are left for the merge to fill.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    words = [w for seg in result.get("segments", []) for w in seg.get("words", [])
             if w.get("start") is not None and w.get("end") is not None]
    for seg in result.get("segments", []):
        seg["speaker"] = None
        for w in seg.get("words", []):
            w["speaker"] = None
    pieces: list[list[dict]] = []
    for w in words:
        prev = pieces[-1][-1] if pieces else None
        if (prev is None or w["start"] - prev["end"] > UTTERANCE_PAUSE
                or SENTENCE_END.search(prev.get("word", "").strip())):
            pieces.append([])
        pieces[-1].append(w)
    if not pieces:
        return
    samples = _mono_16k(sample_rate, audio)
    pieces = _split_at_changes(pieces, samples)
    spans = [(p[0]["start"], p[-1]["end"]) for p in pieces]
    prints = _embed_all(samples, spans)

    long = [i for i, (p, (a, b)) in enumerate(zip(prints, spans)) if p is not None and b - a >= CLUSTER_SECONDS]
    group: dict[int, int] = {}
    if len(long) >= 2:
        tree = linkage(np.stack([prints[i] for i in long]), method="average", metric="cosine")
        found = (fcluster(tree, t=num_speakers, criterion="maxclust") if num_speakers
                 else fcluster(tree, t=1 - CLUSTER_SIMILARITY, criterion="distance"))
        group = {i: int(g) - 1 for i, g in zip(long, found)}
    elif long:
        group = {long[0]: 0}
    if group:
        sums: dict[int, np.ndarray] = {}
        for i, g in group.items():
            sums[g] = sums.get(g, 0) + prints[i] * (spans[i][1] - spans[i][0])
        centres = {g: _unit(v) for g, v in sums.items()}
        for i, p in enumerate(prints):
            if i not in group and p is not None:
                group[i] = max(centres, key=lambda g: float(p @ centres[g]))
        mids = [(a + b) / 2 for a, b in spans]
        for i in range(len(pieces)):
            if i not in group:
                group[i] = group[min(group, key=lambda j: abs(mids[j] - mids[i]))]
    for i, piece in enumerate(pieces):
        for w in piece:
            w["speaker"] = f"SPEAKER_{group.get(i, 0):02d}"
