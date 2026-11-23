# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0",
#     "python-dotenv>=1.0",
#     "rapidfuzz>=3.9",
#     "supabase>=2.0",
# ]
# ///
"""Export the ready clips in Supabase and pick a sample to make gold for.

    uv run evaluation/export_clips.py                 # export + features.csv
    uv run evaluation/export_clips.py --sample 15     # ... + sample.txt

Each clip goes to data/output/clips/<id>/{transcript.json, meta.json}, the
same layout as a library run, so gold_editor.py and evaluate.py read it.
features.csv has every clip's features (clip_features.py, on the pipeline's
transcript) and `gold`: the gold file whose text matches the clip, if any.
--sample picks clips without gold spread over genre × length × speaker count.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from rapidfuzz import fuzz

from clip_features import ROOT, duration_bin, features, joined, speakers_bin

sys.path.insert(0, str(ROOT / "src"))
from web.db import SupabaseStore  # noqa: E402

OUT_DIR = ROOT / "data/output/clips"
GOLD_DIR = ROOT / "data/output/golden"
META_KEYS = ("id", "youtube_id", "title", "channel", "genre", "duration", "created_at", "timings")


def export(store: SupabaseStore, out_dir: Path) -> list[tuple[dict, dict]]:
    clips = []
    for row in store.list(limit=10_000):
        transcript = store.get_transcript(row["id"])
        if not transcript:
            continue
        meta = {k: row.get(k) for k in META_KEYS} | {
            "status": "ready",
            "source_url": f"https://www.youtube.com/watch?v={row['youtube_id']}",
        }
        run = out_dir / row["id"]
        run.mkdir(parents=True, exist_ok=True)
        (run / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
        (run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        clips.append((meta, transcript))
    return clips


def match_gold(clips: list[tuple[dict, dict]], gold_dir: Path, min_score: float) -> dict[str, str]:
    """Clip id → gold file name, by an id in the name or by transcript text."""
    golds = {p.name: joined(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(gold_dir.glob("*.json"))}
    found = {}
    for meta, transcript in clips:
        named = [g for g in golds if f"[{meta['id']}]" in g]
        if named:
            found[meta["id"]] = named[0]
            continue
        text = joined(transcript)
        best = max(golds, key=lambda g: fuzz.ratio(text, golds[g]), default=None)
        if best and fuzz.ratio(text, golds[best]) >= min_score:
            found[meta["id"]] = best
    return found


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over genre × length × speaker-count strata, so every
    stratum gives one clip before any gives a second."""
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["genre"], duration_bin(r["duration"]), speakers_bin(r["speakers"]))
        strata.setdefault(key, []).append(r)
    for members in strata.values():
        rng.shuffle(members)
    order = sorted(strata)
    rng.shuffle(order)
    picked = []
    while len(picked) < n and any(strata[k] for k in order):
        for k in order:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", type=Path, default=OUT_DIR)
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    parser.add_argument("--min-similarity", type=float, default=50.0,
                        help="rapidfuzz ratio (0-100) for a gold file to count as this clip's")
    parser.add_argument("--sample", type=int, default=0, help="clips without gold to pick for review")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    clips = export(SupabaseStore(), args.out)
    gold = match_gold(clips, args.gold_dir, args.min_similarity)

    rows = [{"id": m["id"], "title": m["title"], "gold": gold.get(m["id"], "")} | features(t, m)
            for m, t in clips]
    with open(args.out / "features.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported {len(rows)} clips to {args.out} ({len(gold)} with gold)")
    for cid, name in sorted(gold.items()):
        print(f"  gold  {cid}  {name}")

    if args.sample:
        picked = stratified([r for r in rows if not r["gold"]], args.sample, args.seed)
        lines = [f"{r['id']}\t{r['genre']}\t{duration_bin(r['duration'])}\t{speakers_bin(r['speakers'])}\t{r['title']}"
                 for r in picked]
        (args.out / "sample.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nsample ({len(picked)}) → {args.out / 'sample.txt'}")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
