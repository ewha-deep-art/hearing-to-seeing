# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jiwer>=3.1",
#     "pyannote.metrics>=3.2",
#     "rapidfuzz>=3.9",
#     "scikit-learn>=1.5",
#     "scipy>=1.13",
# ]
# ///
"""Compare library transcripts against gold transcripts.

    uv run evaluation/evaluate.py                       # today's library runs
    uv run evaluation/evaluate.py --date 2026-10-13
    uv run evaluation/evaluate.py --pair "브래드 이발소.json=220ae5338289"

Metrics: README.md 2. --pair overrides the similarity-based pairing for a gold file.
"""

import argparse
import csv
import json
import warnings
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import jiwer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import JaccardErrorRate
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.cluster import contingency_matrix

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data/output/gold"
LIBRARY_DIR = ROOT / "data/output/library"
OUT_DIR = ROOT / "data/output/eval"

NORMALIZE = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])
TO_WORDS = jiwer.Compose([NORMALIZE, jiwer.ReduceToListOfListOfWords()])
TO_CHARS = jiwer.Compose([NORMALIZE, jiwer.ReduceToListOfListOfChars()])
UNMAPPED = "<unmapped>"

# JER scores the union of both transcripts' extents; pyannote warns about it every call.
warnings.filterwarnings("ignore", message="'uem' was approximated")


def load_words(path: Path) -> list[dict]:
    """Words whose text survives normalization, in time order."""
    words = json.loads(path.read_text(encoding="utf-8"))["words"]
    words = [w for w in words if NORMALIZE(w["text"])]
    return sorted(words, key=lambda w: w["start"])


def joined(words: list[dict]) -> str:
    return " ".join(NORMALIZE(w["text"]) for w in words)


# --- selecting and pairing files -------------------------------------------

def library_runs(day: date) -> dict[str, Path]:
    """Finished library runs created on `day`: id → transcript path."""
    runs = {}
    for meta_path in sorted(LIBRARY_DIR.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        transcript = meta_path.parent / "transcript.json"
        created = datetime.fromtimestamp(meta["created_at"]).date()
        if created == day and meta.get("status") == "ready" and transcript.exists():
            runs[meta["id"]] = transcript
    return runs


def pair_files(runs: dict[str, Path], forced: dict[str, str], min_score: float):
    """Gold path → (library id | None, similarity score)."""
    texts = {rid: joined(load_words(p)) for rid, p in runs.items()}
    pairs = {}
    for gold in sorted(GOLD_DIR.glob("*.json")):
        if gold.name in forced:
            rid = forced[gold.name]
            pairs[gold] = (rid if rid in runs else None, None)
            continue
        gold_text = joined(load_words(gold))
        scored = [(fuzz.ratio(gold_text, t), rid) for rid, t in texts.items()]
        score, rid = max(scored, default=(0.0, None))
        pairs[gold] = (rid if score >= min_score else None, round(score, 1))
    return pairs


# --- text metrics -----------------------------------------------------------

def ngram_dup(tokens: list[str], n: int) -> int:
    """Extra occurrences of every n-gram seen more than once."""
    grams = Counter(zip(*(tokens[i:] for i in range(n))))
    return sum(c - 1 for c in grams.values() if c > 1)


def text_metrics(ref: str, hyp: str, ngram: int) -> tuple[dict, jiwer.WordOutput]:
    out = jiwer.process_words(ref, hyp, reference_transform=TO_WORDS, hypothesis_transform=TO_WORDS)
    n = out.hits + out.substitutions + out.deletions
    cer = jiwer.cer(ref, hyp, reference_transform=TO_CHARS, hypothesis_transform=TO_CHARS)
    metrics = {
        "ref_words": n,
        "hyp_words": out.hits + out.substitutions + out.insertions,
        "S": out.substitutions, "D": out.deletions, "I": out.insertions, "C": out.hits,
        "wer": round(out.wer, 4),
        "ier": round(out.insertions / n, 4) if n else 0.0,
        "cer": round(cer, 4),
        "ngram": ngram,
        "ngram_dup": ngram_dup(TO_WORDS(hyp)[0], ngram),
    }
    return metrics, out


# --- speaker metrics --------------------------------------------------------

def token_speakers(words: list[dict]) -> list[str]:
    """Speaker of each token jiwer sees (an entry may normalize to several tokens)."""
    return [w["speaker"] for w in words for _ in NORMALIZE(w["text"]).split()]


def aligned_speakers(ref_words, hyp_words, out: jiwer.WordOutput):
    """(gold speaker, hyp speaker) for every token pair jiwer aligned (hit or substitution)."""
    ref_spk, hyp_spk = token_speakers(ref_words), token_speakers(hyp_words)
    pairs = []
    for chunk in out.alignments[0]:
        if chunk.type in ("equal", "substitute"):
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                pairs.append((ref_spk[chunk.ref_start_idx + k], hyp_spk[chunk.hyp_start_idx + k]))
    return pairs


def hungarian_mapping(ref_labels, hyp_labels) -> dict[str, str]:
    """hyp label → gold label maximizing agreeing words; leftovers stay unmapped."""
    ref_ids = sorted(set(ref_labels))
    hyp_ids = sorted(set(hyp_labels))
    table = contingency_matrix(ref_labels, hyp_labels)  # rows: sorted ref, cols: sorted hyp
    rows, cols = linear_sum_assignment(table, maximize=True)
    mapping = {hyp: UNMAPPED for hyp in hyp_ids}
    mapping.update({hyp_ids[c]: ref_ids[r] for r, c in zip(rows, cols)})
    return mapping


def to_annotation(words: list[dict], merge_gap: float) -> Annotation:
    """Speaker turns: same-speaker words closer than `merge_gap` seconds become one span."""
    ann = Annotation()
    for w in words:
        if w["end"] > w["start"]:
            ann[Segment(w["start"], w["end"])] = w["speaker"]
    return ann.support(collar=merge_gap)


def speaker_metrics(ref_words, hyp_words, out, collar: float, merge_gap: float) -> dict:
    pairs = aligned_speakers(ref_words, hyp_words, out)
    ref_labels = [r for r, _ in pairs]
    hyp_labels = [h for _, h in pairs]
    mapping = hungarian_mapping(ref_labels, hyp_labels)
    mapped = [mapping[h] for h in hyp_labels]
    gold_speakers = sorted(set(ref_labels))

    p, r, f, support = precision_recall_fscore_support(
        ref_labels, mapped, labels=gold_speakers, zero_division=0)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        ref_labels, mapped, labels=gold_speakers, average="macro", zero_division=0)

    ref_ann = to_annotation(ref_words, merge_gap)
    hyp_ann = to_annotation(hyp_words, merge_gap)
    jer = JaccardErrorRate(collar=collar)(ref_ann, hyp_ann)

    return {
        "aligned_words": len(pairs),
        "gold_speakers": len(gold_speakers),
        "hyp_speakers": len(set(w["speaker"] for w in hyp_words)),
        "mapping_hyp_to_gold": mapping,
        "accuracy": round(accuracy_score(ref_labels, mapped), 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f, 4),
        "per_speaker": {
            spk: {"precision": round(p[i], 4), "recall": round(r[i], 4),
                  "f1": round(f[i], 4), "support": int(support[i])}
            for i, spk in enumerate(gold_speakers)
        },
        "jer": round(jer, 4),
    }


# --- driver -----------------------------------------------------------------

def evaluate(gold: Path, hyp: Path, ngram: int, collar: float, merge_gap: float) -> dict:
    ref_words = load_words(gold)
    hyp_words = load_words(hyp)
    text, out = text_metrics(joined(ref_words), joined(hyp_words), ngram)
    return {
        "gold": gold.name,
        "library": hyp.parent.name,
        "text": text,
        "speaker": speaker_metrics(ref_words, hyp_words, out, collar, merge_gap),
    }


SUMMARY_COLUMNS = [
    ("text", "wer"), ("text", "cer"), ("text", "ier"), ("text", "ngram_dup"),
    ("speaker", "accuracy"), ("speaker", "macro_precision"), ("speaker", "macro_recall"),
    ("speaker", "macro_f1"), ("speaker", "jer"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="library runs created on this day (YYYY-MM-DD, default today)")
    parser.add_argument("--pair", action="append", default=[], metavar="GOLD_FILE=LIBRARY_ID",
                        help="pair a gold file with a library run by hand (repeatable)")
    parser.add_argument("--min-similarity", type=float, default=50.0,
                        help="rapidfuzz ratio (0-100) needed to auto-pair a gold file")
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--collar", type=float, default=0.25, help="JER collar in seconds")
    parser.add_argument("--merge-gap", type=float, default=0.5,
                        help="join same-speaker words closer than this (s) into one turn for JER")
    parser.add_argument("-o", "--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    forced = dict(p.split("=", 1) for p in args.pair)
    runs = library_runs(args.date)
    print(f"library runs on {args.date}: {', '.join(runs) or '(none)'}")

    results = []
    for gold, (rid, score) in pair_files(runs, forced, args.min_similarity).items():
        if rid is None:
            print(f"  skip  {gold.name}  (no library run, best similarity {score})")
            continue
        print(f"  pair  {gold.name}  ↔  {rid}  (similarity {score if score is not None else 'forced'})")
        results.append(evaluate(gold, runs[rid], args.ngram, args.collar, args.merge_gap))

    if not results:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = args.date.isoformat()
    (args.out / f"{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    header = ["gold", "library"] + [key for _, key in SUMMARY_COLUMNS]
    rows = [[r["gold"], r["library"]] + [r[group][key] for group, key in SUMMARY_COLUMNS]
            for r in results]
    with open(args.out / f"{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)

    print()
    print("\t".join(header[1:]))
    for row in rows:
        print("\t".join(str(v) for v in row[1:]))
    print(f"\nsaved {args.out / stamp}.json / .csv")


if __name__ == "__main__":
    main()
