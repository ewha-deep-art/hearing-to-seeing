# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0",
#     "python-dotenv>=1.0",
#     "scipy>=1.13",
# ]
# ///
"""Break the scores of evaluate.py down by video features.

    uv run evaluation/evaluate.py --gold-dir data/output/golden --library-dir data/output/clips --all-runs
    uv run evaluation/by_feature.py                          # newest data/output/eval/<date>.json
    uv run evaluation/by_feature.py data/output/eval/2026-11-24.json

Video features (clip_features.py) are measured on the gold transcript, so the
pipeline's own errors (a missed speaker, a dropped line) do not leak into
them; `reviewed` and `character_confidence` describe the pipeline's run and
come from its transcript. Only clip runs (export_clips.py) have a genre.
Writes data/output/eval/by_feature_<eval stem>.md. README.md 6.
"""

import argparse
import json
from pathlib import Path

from scipy.stats import spearmanr

from clip_features import ROOT, duration_bin, features, speakers_bin

EVAL_DIR = ROOT / "data/output/eval"
CLIPS_DIR = ROOT / "data/output/clips"
GOLD_DIR = ROOT / "data/output/golden"

METRICS = [("text", "wer", "WER"), ("text", "cer", "CER"), ("speaker", "accuracy", "화자 Acc"),
           ("speaker", "macro_f1", "화자 F1"), ("speaker", "jer", "JER")]
# Numeric features: correlated with each metric and cut into terciles.
NUMERIC = ["duration", "speakers", "minor_word_share", "speech_ratio", "chars_per_sec",
           "turns_per_min", "overlap_share", "whisper_share", "loud_share", "character_confidence"]
PIPELINE = ("reviewed", "character_confidence")


def rows_of(results: list[dict], clips_dir: Path, gold_dir: Path) -> list[dict]:
    rows = []
    for r in results:
        run = clips_dir / r["library"]
        if not (run / "meta.json").is_file():
            print(f"  skip  {r['library']}  (not a clip run in {clips_dir})")
            continue
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        gold = json.loads((gold_dir / r["gold"]).read_text(encoding="utf-8"))
        hyp = json.loads((run / "transcript.json").read_text(encoding="utf-8"))
        row = {"id": r["library"], "title": meta["title"]} | features(gold, meta)
        run_features = features(hyp, meta)
        row |= {k: run_features[k] for k in PIPELINE}
        row |= {label: r[group][key] for group, key, label in METRICS}
        rows.append(row)
    return rows


def terciles(values: list[float]) -> list[float]:
    s = sorted(values)
    return [s[len(s) // 3], s[2 * len(s) // 3]]


def fmt(v) -> str:
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
    return "–" if v is None else str(v).replace("|", "∣")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def grouped(rows: list[dict], name: str, key) -> str:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    labels = [m[2] for m in METRICS]
    body = [[g, len(members)] + [sum(m[l] for m in members) / len(members) for l in labels]
            for g, members in sorted(groups.items())]
    return f"### {name}\n\n" + table([name, "n"] + labels, body)


def report(rows: list[dict]) -> str:
    labels = [m[2] for m in METRICS]
    out = [f"# 영상 특징별 성능 (클립 {len(rows)}개)", "",
           "영상 특징은 gold 기준, `검토`(영상 리뷰가 화자를 정함)·`캐릭터 conf`는 파이프라인 결과 기준이다.", ""]

    out += ["## 클립별", ""]
    cols = ["genre", "duration", "speakers", "minor_word_share", "chars_per_sec", "turns_per_min",
            "overlap_share", "whisper_share", "reviewed"]
    out += [table(["클립", *cols, *labels],
                  [[r["title"][:30], *(r[c] for c in cols), *(r[l] for l in labels)] for r in rows]), ""]

    out += ["## 구간별 평균", ""]
    out += [grouped(rows, "장르", lambda r: r["genre"] or "–"), ""]
    out += [grouped(rows, "길이", lambda r: duration_bin(r["duration"])), ""]
    out += [grouped(rows, "화자 수", lambda r: speakers_bin(r["speakers"])), ""]
    out += [grouped(rows, "영상 리뷰", lambda r: fmt(r["reviewed"])), ""]
    for f in NUMERIC:
        vals = [r[f] for r in rows if r[f] is not None]
        if f in ("duration", "speakers") or len(set(vals)) < 3:
            continue
        lo, hi = terciles(vals)
        if lo == hi:  # mostly one value: terciles say nothing
            continue
        out += [grouped([r for r in rows if r[f] is not None], f,
                        lambda r, lo=lo, hi=hi: f"하 (<{lo:g})" if r[f] < lo else f"상 (≥{hi:g})" if r[f] >= hi
                        else f"중 ({lo:g}–{hi:g})"), ""]

    out += ["## Spearman 상관 (ρ, p)", "",
            f"n = {len(rows)}. p < 0.05에 *를 붙였다. 표본이 작아 방향을 보는 용도다.", ""]
    body = []
    for f in NUMERIC:
        pts = [r for r in rows if r[f] is not None]
        cells = []
        for l in labels:
            if len(pts) < 4 or len({r[f] for r in pts}) < 2:
                cells.append("–")
                continue
            rho, p = spearmanr([r[f] for r in pts], [r[l] for r in pts])
            cells.append(f"{rho:+.2f}{'*' if p < 0.05 else ''} ({p:.2f})")
        body.append([f, *cells])
    out += [table(["특징", *labels], body), ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval", type=Path, nargs="?", help="evaluate.py's <date>.json (default: the newest)")
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    args = parser.parse_args()

    path = args.eval or max(EVAL_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    rows = rows_of(json.loads(path.read_text(encoding="utf-8")), args.clips_dir, args.gold_dir)
    if not rows:
        print("no clip runs in", path)
        return
    text = report(rows)
    out = path.with_name(f"by_feature_{path.stem}.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
