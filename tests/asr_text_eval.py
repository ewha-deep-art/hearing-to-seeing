"""
asr_text_eval.py

두 개의 ASR 결과(JSON, WordEntry/Transcript 포맷)를 비교하여
WER(단어 오류율) / IER(삽입 오류율) / 5-Dup(5-gram 중복) 을 계산합니다.

사용법:
    python asr_text_eval.py ref.json hyp.json -o result.json

- ref.json : 정답(reference) 전사 결과
- hyp.json : 평가 대상(hypothesis) 전사 결과

지표 정의:
    WER = (S + D + I) / N            (S=치환, D=삭제, I=삽입, N=정답 단어 수)
    IER = I / N                      (WER 중 삽입 성분만)
    5-Dup = 5-gram이 2번 이상 등장한 경우, 초과 등장 횟수의 합
            (예: "안녕 하세요 저는 학생 입니다" 가 2번 나오면 dup +1)
"""

import json
import re
import argparse
from dataclasses import dataclass, field
from collections import Counter


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


# ---------------------------------------------------------------------------
# 텍스트 정규화
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[.,!?;:\"'()\[\]{}~`」「『』·…]")


def normalize_text(text: str) -> str:
    """공백/문장부호 제거. 필요시 자체 규칙으로 교체 가능."""
    t = _PUNCT_RE.sub("", text)
    return t.strip()


def get_tokens(transcript: Transcript, normalize: bool = True) -> list:
    tokens = []
    for w in transcript.words:
        t = normalize_text(w.text) if normalize else w.text
        if t:  # 정규화 후 빈 문자열이면 제외
            tokens.append(t)
    return tokens


# ---------------------------------------------------------------------------
# WER / IER (word-level edit distance, S/D/I 구분)
# ---------------------------------------------------------------------------

def wer_components(ref_tokens: list, hyp_tokens: list) -> dict:
    """
    표준 word-level edit distance (Levenshtein) DP + backtrace로
    substitution(S) / deletion(D) / insertion(I) / correct(C) 개수를 구한다.
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                )

    # backtrace
    i, j = n, m
    S = D = I = C = 0
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and ref_tokens[i - 1] == hyp_tokens[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            C += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1
            j -= 1
        else:
            break

    return {"S": S, "D": D, "I": I, "C": C, "N": n}


# ---------------------------------------------------------------------------
# 5-Dup (n-gram 중복)
# ---------------------------------------------------------------------------

def ngram_duplicate_count(tokens: list, n: int = 5) -> int:
    """
    길이 n짜리 연속 토큰 시퀀스(n-gram) 중 2회 이상 등장한 것에 대해
    (등장 횟수 - 1)의 합을 반환한다. 즉 '초과 등장' 총 횟수.
    """
    if len(tokens) < n:
        return 0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    return sum(c - 1 for c in counts.values() if c > 1)


# ---------------------------------------------------------------------------
# 종합 평가
# ---------------------------------------------------------------------------

def evaluate(
    ref_path: str,
    hyp_path: str,
    normalize: bool = True,
    ngram_n: int = 5,
) -> dict:
    ref = load_transcript(ref_path)
    hyp = load_transcript(hyp_path)

    ref_tokens = get_tokens(ref, normalize=normalize)
    hyp_tokens = get_tokens(hyp, normalize=normalize)

    comp = wer_components(ref_tokens, hyp_tokens)
    n = comp["N"] or 1  # 0으로 나누기 방지
    wer = (comp["S"] + comp["D"] + comp["I"]) / n
    ier = comp["I"] / n

    dup = ngram_duplicate_count(hyp_tokens, n=ngram_n)

    result = {
        "ref_file": ref_path,
        "hyp_file": hyp_path,
        "num_ref_words": len(ref_tokens),
        "num_hyp_words": len(hyp_tokens),
        "edit_components": comp,  # S, D, I, C, N
        "wer": round(wer, 4),
        "wer_percent": round(wer * 100, 2),
        "ier": round(ier, 4),
        "ier_percent": round(ier * 100, 2),
        f"{ngram_n}_dup": dup,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="ASR text metrics evaluator (WER/IER/N-Dup)")
    parser.add_argument("ref", help="정답(reference) JSON 파일 경로")
    parser.add_argument("hyp", help="평가 대상(hypothesis) JSON 파일 경로")
    parser.add_argument("-o", "--output", default="asr_text_eval_result.json", help="결과 저장 경로")
    parser.add_argument("--ngram", type=int, default=5, help="N-Dup 계산에 사용할 n-gram 길이 (기본 5)")
    parser.add_argument("--no-normalize", action="store_true", help="문장부호 등 정규화를 끄려면 지정")
    args = parser.parse_args()

    result = evaluate(
        args.ref,
        args.hyp,
        normalize=not args.no_normalize,
        ngram_n=args.ngram,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"결과 저장 완료: {args.output}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
