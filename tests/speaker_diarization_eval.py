"""
speaker_diarization_eval.py

두 개의 화자분리 결과(JSON, WordEntry/Transcript 포맷)를 비교하여
speaker 필드에 대한 precision / recall / F1 score를 계산합니다.

사용법:
    python speaker_diarization_eval.py ref.json hyp.json -o result.json

- ref.json : 정답(reference) 화자분리 결과
- hyp.json : 평가 대상(hypothesis) 화자분리 결과
- 단어 정렬은 시간 구간(start~end)의 겹침(overlap)을 기준으로 수행합니다.
  (같은 오디오에 대한 결과라고 가정)
- 화자 라벨(SPEAKER_00 등)은 두 결과에서 서로 다른 의미를 가질 수 있으므로,
  겹침 개수가 가장 큰 라벨 매핑을 찾아(itertools.permutations) 정답 라벨과
  1:1 매칭시킨 뒤 지표를 계산합니다. (화자 수가 적다는 전제, 외부 라이브러리 불필요)
"""

import json
import argparse
from dataclasses import dataclass, field
from itertools import permutations
from collections import defaultdict


@dataclass
class WordEntry:
    text: str
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    volume: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


@dataclass
class Transcript:
    words: list = field(default_factory=list)
    language: str = "ko"

    def speakers(self) -> list:
        return sorted({w.speaker for w in self.words})

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        words = [
            WordEntry(
                text=w["text"],
                start=w["start"],
                end=w["end"],
                speaker=w.get("speaker", "SPEAKER_00"),
                volume=w.get("volume", 0.0),
            )
            for w in data.get("words", [])
        ]
        return cls(words=words, language=data.get("language", "ko"))


def load_transcript(path: str) -> Transcript:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Transcript.from_dict(data)


def overlap(a: WordEntry, b: WordEntry) -> float:
    """두 단어의 시간 구간이 겹치는 길이(초)를 반환."""
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def align_words(ref: Transcript, hyp: Transcript, min_overlap: float = 0.01):
    """
    ref, hyp 두 단어 리스트를 시간 overlap 기준으로 정렬(매칭)한다.
    같은 오디오에 대한 결과라고 가정하고, start 시간 순으로 정렬한 뒤
    투 포인터 방식으로 겹치는 단어끼리 짝짓는다.

    반환값: [(ref_word, hyp_word), ...]
    """
    ref_words = sorted(ref.words, key=lambda w: w.start)
    hyp_words = sorted(hyp.words, key=lambda w: w.start)

    pairs = []
    i, j = 0, 0
    while i < len(ref_words) and j < len(hyp_words):
        r, h = ref_words[i], hyp_words[j]
        ov = overlap(r, h)

        if ov >= min_overlap:
            pairs.append((r, h))
            i += 1
            j += 1
        elif r.end <= h.start:
            # ref 단어가 hyp 단어보다 먼저 끝남 -> ref 포인터 전진
            i += 1
        else:
            # hyp 단어가 ref 단어보다 먼저 끝남 -> hyp 포인터 전진
            j += 1

    return pairs


def best_speaker_mapping(pairs):
    """
    혼동행렬(contingency table)을 만들고, 가능한 순열(permutation) 중
    ref-hyp 라벨이 가장 많이 일치하는 매핑을 찾는다.

    반환값: hyp_label -> ref_label 매핑 dict
    """
    ref_speakers = sorted({r.speaker for r, h in pairs})
    hyp_speakers = sorted({h.speaker for r, h in pairs})

    contingency = defaultdict(int)
    for r, h in pairs:
        contingency[(r.speaker, h.speaker)] += 1

    best_mapping = {}
    best_score = -1

    if len(hyp_speakers) <= len(ref_speakers):
        for perm in permutations(ref_speakers, len(hyp_speakers)):
            mapping = dict(zip(hyp_speakers, perm))
            score = sum(contingency[(mapping[h], h)] for h in hyp_speakers)
            if score > best_score:
                best_score = score
                best_mapping = mapping
    else:
        for perm in permutations(hyp_speakers, len(ref_speakers)):
            mapping = dict(zip(perm, ref_speakers))
            score = sum(contingency[(mapping[h], h)] for h in perm)
            if score > best_score:
                best_score = score
                best_mapping = mapping
        # 매핑에 포함되지 못한 hyp 화자는 매칭 없음(None) 처리
        for h in hyp_speakers:
            if h not in best_mapping:
                best_mapping[h] = None

    return best_mapping


def compute_metrics(pairs, mapping):
    """
    mapping 을 이용해 hyp.speaker -> ref 라벨 공간으로 변환한 뒤
    ref 화자별 precision / recall / f1 및 accuracy, macro 평균을 계산한다.
    """
    ref_speakers = sorted({r.speaker for r, h in pairs})

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    for r, h in pairs:
        mapped = mapping.get(h.speaker)
        support[r.speaker] += 1
        if mapped == r.speaker:
            tp[r.speaker] += 1
        else:
            fn[r.speaker] += 1
            if mapped is not None:
                fp[mapped] += 1

    per_speaker = {}
    for spk in ref_speakers:
        precision = tp[spk] / (tp[spk] + fp[spk]) if (tp[spk] + fp[spk]) > 0 else 0.0
        recall = tp[spk] / (tp[spk] + fn[spk]) if (tp[spk] + fn[spk]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_speaker[spk] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support[spk],
        }

    total_tp = sum(tp.values())
    total = sum(support.values())
    accuracy = total_tp / total if total > 0 else 0.0

    n = len(per_speaker) or 1
    macro_precision = sum(v["precision"] for v in per_speaker.values()) / n
    macro_recall = sum(v["recall"] for v in per_speaker.values()) / n
    macro_f1 = sum(v["f1"] for v in per_speaker.values()) / n

    return {
        "num_aligned_words": total,
        "accuracy": round(accuracy, 4),
        "per_speaker": per_speaker,
        "macro_avg": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
    }


def evaluate(ref_path: str, hyp_path: str, min_overlap: float = 0.01) -> dict:
    ref = load_transcript(ref_path)
    hyp = load_transcript(hyp_path)

    pairs = align_words(ref, hyp, min_overlap=min_overlap)
    mapping = best_speaker_mapping(pairs)
    metrics = compute_metrics(pairs, mapping)

    result = {
        "ref_file": ref_path,
        "hyp_file": hyp_path,
        "ref_speakers": ref.speakers(),
        "hyp_speakers": hyp.speakers(),
        "speaker_mapping_hyp_to_ref": mapping,
        "num_ref_words": len(ref.words),
        "num_hyp_words": len(hyp.words),
        "num_aligned_words": metrics["num_aligned_words"],
        "metrics": metrics,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Speaker diarization accuracy evaluator")
    parser.add_argument("ref", help="정답(reference) JSON 파일 경로")
    parser.add_argument("hyp", help="평가 대상(hypothesis) JSON 파일 경로")
    parser.add_argument("-o", "--output", default="speaker_eval_result.json", help="결과 저장 경로")
    parser.add_argument("--min-overlap", type=float, default=0.01, help="단어 정렬 시 최소 overlap(초)")
    args = parser.parse_args()

    result = evaluate(args.ref, args.hyp, min_overlap=args.min_overlap)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"결과 저장 완료: {args.output}")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
