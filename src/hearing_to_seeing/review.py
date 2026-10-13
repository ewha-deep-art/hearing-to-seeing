"""Checking the merged transcript against the clip itself, in one model call.

STT merges several Whisper passes by majority vote (`stt.merge_passes`), and
the server's diarization labels the speakers. Both fail where the clip itself
settles the question: diarization cannot tell apart two fish the screen shows
plainly, and a word most passes misheard is often read right by a few.

So the model is shown the clip (the video, or the audio when there is none)
and the merged transcript cut into utterances, with every spot the passes
disagree on marked with its readings, and asked
  - who speaks each utterance, by whose mouth moves on screen, and
  - where a reading other than the vote's is clearly what is said.
It only chooses — a name per utterance, an option per spot, never new text —
so it cannot put words in anyone's mouth, and every word keeps the timing the
passes gave it. Measured in docs/STT_TUNING.md (4차).

The answer's names replace the diarization labels: each name takes over the
label it shares the most speaking time with, one to one, and a name left over
gets a label of its own. `assign_speaker_colors` then gets the label → name
map instead of guessing it from the dialogue (`identify_speakers`), so a
reviewed clip costs no extra model call. Without a key, with a clip too long,
or on any failure, the caller keeps the plain vote (None).
"""

import mimetypes
import os
import re
import sys
from dataclasses import dataclass

from hearing_to_seeing import stt
from hearing_to_seeing.knowledge import llm, prompts
from hearing_to_seeing.schema import Transcript, WordEntry

# The free tier allows each model about 20 requests a day, and the work and
# cast lookups already spend two of `llm.MODEL`'s per clip; this model's quota
# is its own. It also scored best on the human-checked clip, though it
# overrules the vote far more often than `llm.MODEL` (docs/STT_TUNING.md, 4차).
# H2S_REVIEW_MODEL overrides it.
REVIEW_MODEL = os.environ.get("H2S_REVIEW_MODEL", "gemini-3.5-flash")
# Picking among readings needs the model's full thinking: at the low level it
# overruled the vote 52 times on clip 1 and CER went 2.5 → 4.1 %. But with no
# cast list to go by, full (and medium) thinking ran past 30,000 tokens, cut
# the answer short and once ran over ten minutes. Speakers come out as well at
# the low level (99 %), so a clip with no cast is asked about speakers only.
NO_CAST_THINKING = "low"
MAX_TOKENS = 16_000

# Utterances are cut wherever the label changes, at a pause this long, and
# after a sentence ends. Cut that finely, labelling every utterance right
# gets clip 1 to 100 % speaker accuracy (speaker change alone: 94 %).
UTTERANCE_PAUSE = 0.5
SENTENCE_END = re.compile(r"[.?!]$")
# Measured on clips up to 4.6 min (≈ 115 input tokens a second at low
# resolution); twice that is still far inside the context, but the answer
# grows with the utterance count, so longer clips are not sent.
MAX_SECONDS = 600.0


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


def review_transcript(
    results: list[dict],
    language: str | None,
    characters: list[dict] | None,
    media_path: str,
    uploaded=None,
) -> Review | None:
    """The transcript `stt.parse_result(results)` would give, checked against
    `media_path` (video or audio); None when the review cannot run.
    `uploaded` is `media_path` as `upload` already put it up."""
    # Without a cast, speakers only (see NO_CAST_THINKING).
    lines, spots, spans = _prompt_lines(results, mark=bool(characters))
    if not llm.configured() or not spans or spans[-1][1] > MAX_SECONDS:
        if uploaded:
            llm.discard(uploaded)
        return None

    video = not (mimetypes.guess_type(media_path)[0] or "").startswith("audio/")
    prompt = prompts.REVIEW_PROMPT.format(
        how=prompts.REVIEW_HOW["video" if video else "audio"],
        text_task=prompts.REVIEW_TEXT_TASK if spots else "",
        cast=", ".join(c["name"] for c in characters or []) or "(모름)",
        utterances="\n".join(lines),
    )
    print(f"… {'영상' if video else '음성'} 검토 중: 발화 {len(spans)}개, 엇갈린 자리 {len(spots)}개",
          file=sys.stderr, flush=True)
    try:
        answer = llm.ask(prompts.REVIEW_SYSTEM, prompt, prompts.REVIEW_SCHEMA, model=REVIEW_MODEL,
                         media_path=media_path, media=uploaded, fallback=False, temperature=0,
                         max_tokens=MAX_TOKENS,
                         thinking_level=None if characters else NO_CAST_THINKING)
    except Exception as exc:  # network, quota, bad key — anything
        print(f"! 검토 실패, 다수결 결과 사용: {exc}", file=sys.stderr, flush=True)
        return None

    chosen: dict[int, str] = {}
    for c in answer.get("corrections") or []:
        n = _number(c.get("id", ""), "d")
        option = c.get("option")
        if n is not None and n < len(spots) and isinstance(option, int) and 1 <= option <= len(spots[n][0]):
            chosen[n] = spots[n][0][option - 1].key
    asked = iter(range(len(spots)))
    transcript = stt.parse_result(
        results, language,
        decide=(lambda options, pick, at: chosen.get(next(asked), pick)) if chosen else None,
    )

    named: dict[int, str] = {}
    for u in answer.get("utterances") or []:
        n = _number(u.get("id", ""), "u")
        if n is not None and 1 <= n <= len(spans) and (u.get("speaker") or "").strip():
            named[n - 1] = u["speaker"].strip()

    def utterance(w: WordEntry) -> int:
        mid = (w.start + w.end) / 2
        return min(range(len(spans)), key=lambda k: max(spans[k][0] - mid, mid - spans[k][1], 0.0))

    names = _relabel(transcript.words, [named.get(utterance(w)) for w in transcript.words])
    print(f"… 검토 반영: 화자 {len(names)}명, 단어 {len(chosen)}곳 수정", file=sys.stderr, flush=True)
    return Review(transcript, names)
