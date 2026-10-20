"""Checks the merged transcript against the clip itself, in one model call.

The model sees the clip and the transcript cut into utterances, with every
spot the passes disagree on marked, and picks a speaker per utterance and a
reading per spot — never new text, so every word keeps its timing. Its names
replace the diarization labels. Without a key, for a clip too long, or on any
failure, the caller keeps the plain vote (None). See knowledge/README.md ④.
"""

import itertools
import mimetypes
import re
import sys
from dataclasses import dataclass

from hearing_to_seeing import stt
from hearing_to_seeing.knowledge import llm, prompts
from hearing_to_seeing.schema import Transcript, WordEntry

# A daily quota of its own, apart from the cast lookup's `llm.MODEL`
# (evaluation/README.md 5.5).
REVIEW_MODEL = "gemini-3.7-flash"
# Over quota or overloaded: Flash models only, each on its own quota.
REVIEW_FALLBACKS = ("gemini-3.8-flash", "gemini-3.5-flash")
# Word picks need full thinking (evaluation/README.md 5.1); without a cast or
# past TEXT_MAX_SECONDS, only speakers are asked, at low thinking. Over the
# four-pass vote the picks took 114 s on a 280 s clip and raised its WER
# (0.196 → 0.200), so every clip is asked about speakers only — which lets
# the review start on the first pass while the others decode (`pipeline.run`).
SPEAKERS_THINKING = "low"
TEXT_MAX_SECONDS = 0.0
MAX_TOKENS = 16_000
# One try, long enough for a ~10-minute clip at low thinking (~260 s).
TIMEOUT_MS = 480_000

# Utterances are cut wherever the label changes, at a pause this long, and
# after a sentence ends. Cut that finely, labelling every utterance right
# gets clip 1 to 100 % speaker accuracy (speaker change alone: 94 %).
UTTERANCE_PAUSE = 0.5
SENTENCE_END = re.compile(r"[.?!]$")
# One answered utterance: "u12 이름" (a stray ":" or "=" after the id is let through).
UTTERANCE_ANSWER = re.compile(r"\s*u?(\d+)\s*[:=]?\s*(\S.*)")
# Longer clips than the longest measured (12.3 min) are not sent
# (evaluation/README.md 5.1).
MAX_SECONDS = 900.0


@dataclass
class Review:
    transcript: Transcript
    names: dict[str, str]  # speaker label → the name the model gave it


def _utterances(words: list[dict]) -> list[list[dict]]:
    utterances: list[list[dict]] = []
    for w in words:
        prev = utterances[-1][-1] if utterances else None
        if (prev is None or prev["speaker"] != w["speaker"]
                or w["start"] - prev["end"] > UTTERANCE_PAUSE or SENTENCE_END.search(prev["text"])):
            utterances.append([])
        utterances[-1].append(w)
    return utterances


def _marker(n: int, options: list[stt.Option], at: float) -> str:
    return f"⟨d{n} @{at:.1f}s: " + " | ".join(
        f"{i}) {o.text or '(없음)'} ×{o.votes}" for i, o in enumerate(options, 1)
    ) + "⟩"


def _prompt_lines(results: list[dict], mark: bool = True):
    """The merged transcript as numbered utterance lines with its disputed
    spots marked in place, the spots (options with the vote's pick first),
    and each utterance's time span. Unmarked, the lines are the plain vote
    and there are no spots."""
    spots: list[tuple[list[stt.Option], float]] = []
    numbers: dict[int, int] = {}  # id of the options list a word carries → spot number

    def record(options: list[stt.Option], pick: str, at: float) -> str:
        numbers[id(options)] = len(spots)
        spots.append((sorted(options, key=lambda o: (o.key != pick, -o.votes)), at))
        return pick

    def marker(options: list[stt.Option]) -> str:
        n = numbers[id(options)]
        return _marker(n, *spots[n])

    utterances = _utterances(stt.merged_words(results, record if mark else None))
    lines = []
    for u, words in enumerate(utterances, 1):
        parts, covered = [], 0
        for w in words:
            parts += [marker(o) for o in w.get("spot_before", [])]
            if covered:  # a word of the reading a marker already stands for
                covered -= 1
            elif w.get("spot"):
                parts.append(marker(w["spot"]))
                covered = len(spots[numbers[id(w["spot"])]][0][0].text.split()) - 1
            else:
                parts.append(w["text"])
            parts += [marker(o) for o in w.get("spot_after", [])]
        lines.append(f"u{u} [{words[0]['start']:.1f}–{words[-1]['end']:.1f}s] "
                     f"{words[0]['speaker']}: " + " ".join(parts))
    return lines, spots, [(ws[0]["start"], ws[-1]["end"]) for ws in utterances]


def _number(value: str, prefix: str) -> int | None:
    digits = str(value).strip().removeprefix(prefix)
    return int(digits) if digits.isdigit() else None


def _relabel(words: list[WordEntry], names: list[str | None]) -> dict[str, str]:
    """Gives each name the diarization label it shares the most time with.

    One to one, greedily by shared time; a name with no label left gets a new
    one. Words the model did not name keep their label. Returns label → name.
    """
    shared: dict[tuple[str, str], float] = {}
    for w, name in zip(words, names):
        if name:
            shared[(name, w.speaker)] = shared.get((name, w.speaker), 0.0) + w.end - w.start
    label_of: dict[str, str] = {}
    for (name, label), _ in sorted(shared.items(), key=lambda kv: -kv[1]):
        if name not in label_of and label not in label_of.values():
            label_of[name] = label
    used = {w.speaker for w in words} | set(label_of.values())
    n = 0
    for name in dict.fromkeys(name for name in names if name):
        while name not in label_of:
            label = f"SPEAKER_{n:02d}"
            n += 1
            if label not in used:
                label_of[name] = label
    for w, name in zip(words, names):
        if name:
            w.speaker = label_of[name]
    return {label: name for name, label in label_of.items()}


def upload(media_path: str, seconds: float):
    """`media_path` put up for the review ahead of time, so the upload runs
    alongside STT; None when no review will run or the upload failed (the
    review then uploads it itself)."""
    if not llm.configured() or seconds > MAX_SECONDS:
        return None
    try:
        return llm.upload(media_path)
    except Exception as exc:
        print(f"… 검토용 미리 올리기 실패: {exc}", file=sys.stderr, flush=True)
        return None


@dataclass
class Answer:
    """What the model said about one set of passes."""

    spans: list[tuple[float, float]]  # each utterance's time span
    named: dict[int, str]  # utterance index → the name it gave
    chosen: dict[int, str]  # spot number → the option key it picked


def ask_review(
    results: list[dict],
    characters: list[dict] | None,
    media_path: str,
    uploaded=None,
    words: bool = True,
) -> Answer | None:
    """The model's answer about `results`, checked against `media_path`
    (video or audio); None when the review cannot run. `uploaded` is
    `media_path` as `upload` already put it up. `words=False` asks about
    speakers only."""
    # Without a cast, or for a long clip, speakers only (see SPEAKERS_THINKING).
    seconds = max((w.get("end") or 0.0 for r in results
                   for seg in r.get("segments", []) for w in seg.get("words", [])), default=0.0)
    text_task = words and bool(characters) and seconds <= TEXT_MAX_SECONDS
    lines, spots, spans = _prompt_lines(results, mark=text_task)
    if not llm.configured() or not spans or spans[-1][1] > MAX_SECONDS:
        if uploaded:
            llm.discard(uploaded)
        return None

    video = not (mimetypes.guess_type(media_path)[0] or "").startswith("audio/")
    prompt = prompts.REVIEW_PROMPT.format(
        how=prompts.REVIEW_HOW["video" if video else "audio"],
        text_task=prompts.REVIEW_TEXT_TASK if spots else "",
        cast=", ".join(c["name"] for c in characters or []) or "(모름)",
        example=(characters or [{"name": "이름"}])[0]["name"],
        utterances="\n".join(lines),
    )
    print(f"… {'영상' if video else '음성'} 검토 중: 발화 {len(spans)}개, 엇갈린 자리 {len(spots)}개",
          file=sys.stderr, flush=True)
    try:
        answer = llm.ask(prompts.REVIEW_SYSTEM, prompt, prompts.REVIEW_SCHEMA, model=REVIEW_MODEL,
                         media_path=media_path, media=uploaded, fallback=REVIEW_FALLBACKS, temperature=0,
                         max_tokens=MAX_TOKENS, timeout_ms=TIMEOUT_MS,
                         thinking_level=None if text_task else SPEAKERS_THINKING)
    except Exception as exc:  # network, quota, bad key — anything
        print(f"! 검토 실패, 다수결 결과 사용: {exc}", file=sys.stderr, flush=True)
        return None

    chosen: dict[int, str] = {}
    for c in answer.get("corrections") or []:
        n = _number(c.get("id", ""), "d")
        option = c.get("option")
        if n is not None and n < len(spots) and isinstance(option, int) and 1 <= option <= len(spots[n][0]):
            chosen[n] = spots[n][0][option - 1].key
    named: dict[int, str] = {}
    for u in answer.get("utterances") or []:
        match = UTTERANCE_ANSWER.match(str(u))
        n = int(match[1]) if match else None
        if n is not None and 1 <= n <= len(spans):
            named[n - 1] = match[2].strip()
    return Answer(spans, named, chosen)


def apply_review(answer: Answer, results: list[dict], language: str | None) -> Review:
    """`results` merged with the readings `answer` chose, and named by its
    utterances' time spans. With no readings chosen, `results` may hold more
    passes than the model was asked about (a speakers-only review of the
    first pass, asked while the others decoded)."""
    asked = itertools.count()
    transcript = stt.parse_result(
        results, language,
        decide=(lambda options, pick, at: answer.chosen.get(next(asked), pick)) if answer.chosen else None,
    )
    spans = answer.spans

    def utterance(w: WordEntry) -> int:
        mid = (w.start + w.end) / 2
        return min(range(len(spans)), key=lambda k: max(spans[k][0] - mid, mid - spans[k][1], 0.0))

    names = _relabel(transcript.words, [answer.named.get(utterance(w)) for w in transcript.words])
    print(f"… 검토 반영: 화자 {len(names)}명, 단어 {len(answer.chosen)}곳 수정", file=sys.stderr, flush=True)
    return Review(transcript, names)


def review_transcript(
    results: list[dict],
    language: str | None,
    characters: list[dict] | None,
    media_path: str,
    uploaded=None,
) -> Review | None:
    """The transcript `stt.parse_result(results)` would give, checked against
    `media_path`; None when the review cannot run (see `ask_review`)."""
    answer = ask_review(results, characters, media_path, uploaded)
    return apply_review(answer, results, language) if answer else None
