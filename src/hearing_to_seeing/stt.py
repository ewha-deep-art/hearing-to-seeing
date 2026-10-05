"""STT, forced alignment and speaker diarization via the remote WhisperX API.

The server runs WhisperX and returns its `assign_word_speakers` output as-is:
segments carrying per-word `start`/`end`/`speaker`. Set WHISPERX_API_KEY and
WHISPERX_API_URL in the environment or `.env`.

The server's batched decoder has no temperature fallback or compression-ratio
check, so the raw output carries Whisper's known failure modes; `parse_result`
cleans them up before anything downstream sees the words.
"""

import os
import re
import statistics
import time
from typing import Callable, NamedTuple
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

from hearing_to_seeing.schema import Transcript, WordEntry

# The server processes roughly in real time (45 s of audio took ~50 s), so a
# long input needs a generous read timeout.
REQUEST_TIMEOUT = 1800
# The server now and then answers 503 while busy; a pass is retried this many
# times, this many seconds apart, before the whole transcription fails.
REQUEST_RETRIES = 3
RETRY_WAIT = 10
# The health check answers instantly when idle; a busy server is waited on anyway.
HEALTH_TIMEOUT = 5

# Forced-alignment model for word timings (the server's `align_model_name`).
# WhisperX's Korean default (kresnik/wav2vec2-large-xlsr-korean) stretched
# words over laughter and music; this one put 95% of word starts within 0.2 s
# of the reference instead of 86% (docs/STT_TUNING.md).
ALIGN_MODEL = "w11wo/wav2vec2-xls-r-300m-korean"
# The aligner depends on the language, so an unspecified one is detected
# first with the fastest model.
DETECTION_MODEL = "large-v3-turbo"

# The audio is decoded once per (Whisper model, window length) pair — the
# server's `model` and `chunk_size` (seconds) — and the passes merged region by
# region (`merge_passes`). Long windows give Whisper context; short ones keep a
# decoding loop from eating the lines after it and time words best; the models
# mishear different words. Measured on dubbed-cartoon clips (docs/STT_TUNING.md).
# The first pass is the primary one: it anchors speaker labels and wins ties.
# It is also the only one diarized: every pass diarizes the same audio the
# same way, so the others borrow its labels by time (`_borrow_speakers`) —
# 97–100 % of words keep the speaker they had with every pass diarized, at
# 6 s less per pass. The passes are sent grouped by model, starting with the
# one the server holds (`_pass_order`): switching models costs 2–4 s.
PASSES = (
    ("large-v3", 30), ("large-v2", 30), ("large-v2", 20),
    ("large-v3", 5), ("large-v3-turbo", 5),
)

# A syllable repeated past this inside one word ("하하하하하…") is a decoding
# loop, not speech; the run is cut down to this many.
MAX_CHAR_REPEAT = 4
# Same for one word said again and again ("스폰지밥! 스폰지밥! …").
MAX_WORD_REPEAT = 4

# Alignment stretches a phrase's last word over the pause (or laugh, or music)
# that follows it, up to the next word. Nobody holds a syllable this long, so
# a word's end is clamped to this many seconds per syllable, at least
# MIN_WORD_CAP. Starts are reliable and left alone.
SECONDS_PER_SYLLABLE = 0.5
MIN_WORD_CAP = 1.0

# Passes are compared region by region; a region ends where every pass is
# silent for at least this long.
REGION_GAP = 0.3
# The first pass (the longest window) wins a region unless its disagreement
# with the others exceeds the least disagreeing pass's by more than this much
# per character it holds.
PRIMARY_BIAS = 0.05
# A spot where only one pass lacks a word all the others read (or reads one
# none of the others do), with at least this many on the other side, is no
# dispute: one pass dropping a line or hearing laughter is a decoding failure,
# not a reading. Shown to the reviewer, a line one pass dropped came out as
# nothing but such spots, and it picked "nothing" for every word.
SETTLED_VOTES = 4

# Whisper's training data was full of broadcast credits, so silence and music
# come back as these lines. They never occur as dialogue in practice.
HALLUCINATIONS = re.compile(
    r"MBC\s*뉴스|KBS\s*뉴스|SBS\s*뉴스|뉴스\s*\S+입니다"
    r"|시청\s*해\s*주셔서|구독\s*(과|와)?\s*좋아요|좋아요\s*(와|과)?\s*구독"
    r"|자막\s*(제공|by)|다음\s*영상에서\s*만나요"
)


class Option(NamedTuple):
    """One reading of a spot the passes disagree on (see `_vote`)."""

    key: str  # normalised text; "" for no word at all
    text: str
    votes: int


# (options, the vote's pick, seconds into the audio) -> the key to keep
Decide = Callable[[list[Option], str, float], str]


class STTRequestError(RuntimeError):
    pass


def _request(
    wav: bytes,
    language: str | None,
    model: str | None = None,
    chunk_size: int | None = None,
    num_speakers: int | None = None,
    diarize: bool = True,
) -> dict:
    # min/max_speakers hints changed nothing on the test clips, but an exact
    # count that is at least the real one splits voices diarization merged
    # (docs/STT_TUNING.md); without one, diarization picks the count itself.
    data = {"align": "true", "diarize": "true" if diarize else "false"}
    if language:
        data["language"] = language
    if language == "ko":
        data["align_model_name"] = ALIGN_MODEL
    if model:
        data["model"] = model
    if chunk_size:
        data["chunk_size"] = str(chunk_size)
    if num_speakers and diarize:
        data["num_speakers"] = str(num_speakers)
    return _post(wav, data)


def _loaded_model() -> str | None:
    """The Whisper model the server holds now (it keeps one), if it says."""
    load_dotenv()
    try:
        response = requests.get(urljoin(os.environ["WHISPERX_API_URL"], "health"), timeout=HEALTH_TIMEOUT)
        return response.json()["loaded_model"]["model"]
    except Exception:  # only an ordering hint
        return None


def _pass_order(loaded: str | None) -> list[int]:
    """Indices into PASSES, grouped by model, the `loaded` one's first."""
    models = sorted(dict.fromkeys(m for m, _ in PASSES), key=lambda m: m != loaded)
    return [i for m in models for i, (model, _) in enumerate(PASSES) if model == m]


def _detect_language(wav: bytes) -> str | None:
    """The spoken language, from a quick decode with nothing aligned or diarized
    (a few seconds for five minutes of audio)."""
    return _post(wav, {"model": DETECTION_MODEL}).get("language")


def _post(wav: bytes, data: dict) -> dict:
    load_dotenv()
    api_key = os.environ.get("WHISPERX_API_KEY")
    if not api_key:
        raise STTRequestError("WHISPERX_API_KEY is not set (see .env.example).")
    url = os.environ.get("WHISPERX_API_URL")
    if not url:
        raise STTRequestError("WHISPERX_API_URL is not set (see .env.example).")

    for attempt in range(REQUEST_RETRIES + 1):
        retry = attempt < REQUEST_RETRIES
        try:
            response = requests.post(
                url,
                headers={"X-API-Key": api_key},
                # The server accepts only parts typed audio/*.
                files={"file": ("audio.wav", wav, "audio/wav")},
                data=data,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.ConnectionError as exc:
            if retry:
                time.sleep(RETRY_WAIT)
                continue
            raise STTRequestError(f"WhisperX API request failed: {exc}") from exc
        except requests.RequestException as exc:
            raise STTRequestError(f"WhisperX API request failed: {exc}") from exc
        if response.status_code in (502, 503, 504) and retry:
            time.sleep(RETRY_WAIT)
            continue
        break

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise STTRequestError(f"WhisperX API returned {response.status_code}: {detail}")
    return response.json()


def _collapse_char_loop(text: str) -> str:
    """Cuts any 1–3 character unit repeated past MAX_CHAR_REPEAT down to it."""
    return re.sub(
        r"(.{1,3}?)\1{%d,}" % MAX_CHAR_REPEAT,
        lambda m: m.group(1) * MAX_CHAR_REPEAT,
        text,
    )


def _word_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", text)


def _raw_words(result: dict) -> list[dict]:
    """Flattens segments into words, each carrying its own speaker.

    WhisperX labels every word, and one segment often spans a whole exchange,
    so the segment's (majority) speaker is only the last resort.
    """
    words = []
    for segment in result["segments"]:
        if HALLUCINATIONS.search(segment.get("text", "")):
            continue
        for w in segment.get("words", []):
            text = _collapse_char_loop(w.get("word", "").strip())
            if not _word_key(text):
                continue
            words.append({
                "text": text,
                "start": w.get("start"),
                "end": w.get("end"),
                "speaker": w.get("speaker"),
                "segment_speaker": segment.get("speaker"),
            })
    return words


def _drop_word_loops(words: list[dict]) -> list[dict]:
    kept, run = [], 0
    for w in words:
        same = bool(kept) and _word_key(kept[-1]["text"]) == _word_key(w["text"])
        run = run + 1 if same else 1
        if run <= MAX_WORD_REPEAT:
            kept.append(w)
    return kept


def _fill_timing(words: list[dict]) -> None:
    """Gives untimed words (alignment skips digits, symbols…) the gap they sit in.

    A run of untimed words shares the span between the timed neighbours
    evenly; the app needs every word on the timeline to draw it.
    """
    i = 0
    while i < len(words):
        if words[i]["start"] is not None and words[i]["end"] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and (words[j]["start"] is None or words[j]["end"] is None):
            j += 1
        lo = words[i - 1]["end"] if i > 0 else None
        hi = words[j]["start"] if j < len(words) else None
        if lo is None and hi is None:
            return
        if lo is None:
            lo = max(0.0, hi - 0.3 * (j - i))
        if hi is None or hi < lo:
            hi = lo + 0.3 * (j - i)
        step = (hi - lo) / (j - i)
        for k in range(i, j):
            words[k]["start"] = lo + step * (k - i)
            words[k]["end"] = lo + step * (k - i + 1)
        i = j


def _fill_speakers(words: list[dict]) -> None:
    """Words diarization left unlabelled take the nearest labelled neighbour's speaker."""
    labelled = [i for i, w in enumerate(words) if w["speaker"]]
    if not labelled:
        for w in words:
            w["speaker"] = w["segment_speaker"] or "SPEAKER_00"
        return
    for i, w in enumerate(words):
        if not w["speaker"]:
            nearest = min(labelled, key=lambda k: abs(words[k]["start"] - w["start"]))
            w["speaker"] = words[nearest]["speaker"]


def _borrow_speakers(words: list[dict], anchor: list[dict]) -> None:
    """Labels an undiarized pass's words from the diarized `anchor` pass: the
    speaker of the anchor word overlapping each most, else the nearest one."""
    if not anchor:
        return _fill_speakers(words)
    for w in words:
        def shared(a: dict) -> float:
            return min(w["end"], a["end"]) - max(w["start"], a["start"])

        best = max(anchor, key=shared)
        if shared(best) <= 0:
            mid = (w["start"] + w["end"]) / 2
            best = min(anchor, key=lambda a: abs((a["start"] + a["end"]) / 2 - mid))
        w["speaker"] = best["speaker"]


def _clean_words(result: dict, anchor: list[dict] | None = None) -> list[dict]:
    """A response's words, cleaned; a pass not diarized takes `anchor`'s labels."""
    words = _drop_word_loops(_raw_words(result))
    _fill_timing(words)
    if anchor is not None and not any(w["speaker"] for w in words):
        _borrow_speakers(words, anchor)
    else:
        _fill_speakers(words)
    return words


def _clamp_durations(words: list[dict]) -> None:
    for w in words:
        cap = max(MIN_WORD_CAP, SECONDS_PER_SYLLABLE * len(_word_key(w["text"])))
        w["end"] = min(w["end"], w["start"] + cap)


def _keep_order(words: list[dict]) -> None:
    """Merged words come from different passes, whose timings differ slightly;
    keep them in order and from overlapping, as the karaoke fill expects."""
    for prev, w in zip(words, words[1:]):
        w["start"] = max(w["start"], prev["start"])
        w["end"] = max(w["end"], w["start"])
        prev["end"] = min(prev["end"], w["start"])


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _regions(passes: list[list[dict]]) -> list[tuple[float, float]]:
    """Stretches of the timeline separated by a gap no pass puts a word in."""
    spans = sorted((w["start"], w["end"]) for words in passes for w in words)
    regions: list[list[float]] = []
    for start, end in spans:
        if regions and start <= regions[-1][1] + REGION_GAP:
            regions[-1][1] = max(regions[-1][1], end)
        else:
            regions.append([start, end])
    return [(s, e) for s, e in regions]


def _align_speakers(anchor: list[dict], words: list[dict]) -> None:
    """Renames `words`' speakers to the `anchor` pass's labels they overlap most.

    Passes over differently prepared audio are diarized separately, so their
    labels need not match. Pairs are matched greedily by shared speaking time,
    one to one; a speaker with no counterpart keeps a label of its own.
    """
    overlap: dict[tuple[str, str], float] = {}
    for w in words:
        for a in anchor:
            shared = min(w["end"], a["end"]) - max(w["start"], a["start"])
            if shared > 0:
                key = (w["speaker"], a["speaker"])
                overlap[key] = overlap.get(key, 0.0) + shared
    mapping: dict[str, str] = {}
    for (mine, theirs), _ in sorted(overlap.items(), key=lambda kv: -kv[1]):
        if mine not in mapping and theirs not in mapping.values():
            mapping[mine] = theirs
    for w in words:
        if w["speaker"] not in mapping:
            mapping[w["speaker"]] = f"{w['speaker']}_x"
        w["speaker"] = mapping[w["speaker"]]


def _align_words(pivot: list[str], other: list[str]) -> list[tuple[int | None, int | None]]:
    """Word-level edit alignment; a pair is (pivot index, other index), None for a gap.

    Substituting one word for another costs their normalised character
    distance, so 너/넌 pair up while 핑핑아/정말 rather count as a gap each.
    """
    n, m = len(pivot), len(other)

    def sub(a: str, b: str) -> float:
        return _edit_distance(a, b) / max(len(a), len(b), 1)

    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1,
                             cost[i - 1][j - 1] + 2 * sub(pivot[i - 1], other[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + 2 * sub(pivot[i - 1], other[j - 1]):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


def _vote(
    candidates: list[list[dict]],
    pivot: int,
    decide: Decide | None = None,
    empty: list[list[Option]] | None = None,
) -> list[dict]:
    """Word-level majority vote of `candidates` around the `pivot` candidate (ROVER).

    Every other candidate is aligned to the pivot. Each pivot word stays
    unless a strict majority of all candidates puts the same other word, or
    nothing, in its place; a word the pivot lacks is added only when a strict
    majority has it in the same gap. Ties keep the pivot's reading. Every
    word comes out timed and labelled by all the candidates that read it
    (`_consensus_timing`).

    Where the candidates disagree, `decide` (if given) may overrule the vote:
    it gets the options — each reading with its votes, "" for nothing — and
    the vote's pick (and the spot's time, by every reading's median start),
    and returns the key of the option to keep. Spots `_settled` by all but
    one pass are not asked. It is called in timeline order, so the same
    passes always ask the same questions.
    The words a disputed spot puts out carry its options (`spot`, on the
    first of them), so a reviewer can be shown each one in place. Spots left
    empty collect in `empty` until the next word puts out, which carries them
    as `spot_before`; the list may span regions (see `merge_passes`).
    """
    base = candidates[pivot]
    keys = [_word_key(w["text"]) for w in base]
    slots: list[dict[str, list[dict | None]]] = [{} for _ in base]
    gaps: list[dict[str, list[list[dict]]]] = [{} for _ in range(len(base) + 1)]
    others = [c for k, c in enumerate(candidates) if k != pivot]
    for i, w in enumerate(base):
        slots[i].setdefault(keys[i], []).append(w)
    for words in others:
        pending: list[dict] = []
        for i, j in _align_words(keys, [_word_key(w["text"]) for w in words]):
            if i is None:
                pending.append(words[j])
                continue
            gap_key = " ".join(_word_key(w["text"]) for w in pending)
            gaps[i].setdefault(gap_key, []).append(pending)
            pending = []
            slots[i].setdefault("" if j is None else _word_key(words[j]["text"]), []).append(
                None if j is None else words[j]
            )
        gap_key = " ".join(_word_key(w["text"]) for w in pending)
        gaps[len(base)].setdefault(gap_key, []).append(pending)

    majority = len(candidates) // 2 + 1
    out: list[dict] = []
    empty = [] if empty is None else empty  # spots left empty since the last word

    def put(words: list[dict], options: list[Option] | None) -> None:
        if words and options:
            words[0]["spot"] = options
        if words and empty:
            words[0]["spot_before"] = empty[:]
            empty.clear()
        out.extend(words)

    for i in range(len(base) + 1):
        if gaps[i]:
            key, runs = max(gaps[i].items(), key=lambda kv: len(kv[1]))
            # The pivot itself (and any candidate not reaching this gap) votes "nothing".
            pick = key if key and len(runs) >= majority else ""
            readings = {k: r for k, r in gaps[i].items() if k}
            options = None
            if decide and readings:
                nothing = len(candidates) - sum(len(r) for r in readings.values())
                options = [Option("", "", nothing)] + [
                    Option(k, " ".join(w["text"] for w in r[0]), len(r)) for k, r in readings.items()
                ]
                if _settled(options):
                    options = None
                else:
                    at = _spot_time([r[0] for runs in readings.values() for r in runs])
                    pick = _decided(decide, options, pick, at)
                    if not pick:
                        empty.append(options)
            if pick:
                put([_consensus_timing(list(same)) for same in zip(*gaps[i][pick])], options)
        if i < len(base):
            key, votes = max(slots[i].items(), key=lambda kv: (len(kv[1]), kv[0] == keys[i]))
            pick = keys[i] if len(votes) < majority or key == keys[i] else key
            options = None
            if decide and len(slots[i]) > 1:
                options = [Option(k, v[0]["text"] if v[0] else "", len(v)) for k, v in slots[i].items()]
                if _settled(options):
                    options = None
                else:
                    at = _spot_time([w for v in slots[i].values() for w in v if w])
                    pick = _decided(decide, options, pick, at)
                    if not pick:
                        empty.append(options)
            if pick:
                put([_consensus_timing([w for w in slots[i][pick] if w])], options)
    return out


def _decided(decide: Decide, options: list[Option], pick: str, at: float) -> str:
    """`decide`'s answer, or the vote's `pick` when the answer is not an option."""
    choice = decide(options, pick, at)
    return choice if any(o.key == choice for o in options) else pick


def _settled(options: list[Option]) -> bool:
    """A word against no word, with one pass alone on its side (see SETTLED_VOTES)."""
    voted = sorted((o for o in options if o.votes), key=lambda o: o.votes)
    return (len(voted) == 2 and any(not o.key for o in voted)
            and voted[0].votes == 1 and voted[1].votes >= SETTLED_VOTES)


def _spot_time(words: list[dict]) -> float:
    """Where a spot is, by the median start of every reading of it.

    The pivot pass alone can be seconds off — a long window stretched over
    laughter put a line 4 s early — and the reviewer looks at that moment.
    The median is also how the words come out timed (`_consensus_timing`).
    """
    return statistics.median(w["start"] for w in words)


def _consensus_timing(same: list[dict]) -> dict:
    """One word as several passes read it: timed at their median, spoken by their majority.

    Alignment sometimes fails for a whole stretch in one pass, squeezing its
    words into a moment seconds away, and a word's speaker follows its timing;
    the other passes still place it. Ties keep the first word's speaker.
    """
    word = dict(same[0])
    word["start"] = statistics.median(w["start"] for w in same)
    word["end"] = max(word["start"], statistics.median(w["end"] for w in same))
    counts: dict[str, int] = {}
    for w in same:
        counts[w["speaker"]] = counts.get(w["speaker"], 0) + 1
    word["speaker"] = max(counts, key=lambda k: (counts[k], k == same[0]["speaker"]))
    return word


def merge_passes(passes: list[list[dict]], decide: Decide | None = None) -> list[dict]:
    """Merges several decodings of the same audio, region by region.

    Passes fail in different places: a decoding loop or a dropped line shows
    up in one pass and is outvoted by the rest. Per region the pass that
    agrees most with the others is picked (minimum Bayes risk, the first pass
    favoured by PRIMARY_BIAS), then its words are put to a word-level vote
    (`_vote`, which hands disputed spots to `decide`). Speaker labels are
    mapped onto the first pass's beforehand.
    """
    if len(passes) == 1:
        return passes[0]
    for words in passes[1:]:
        _align_speakers(passes[0], words)
    merged: list[dict] = []
    empty: list[list[Option]] = []
    for lo, hi in _regions(passes):
        candidates = [
            [w for w in words if lo <= (w["start"] + w["end"]) / 2 <= hi] for words in passes
        ]
        texts = ["".join(_word_key(w["text"]) for w in c) for c in candidates]
        risks = [sum(_edit_distance(t, u) for u in texts) for t in texts]
        risks[0] -= PRIMARY_BIAS * len(texts[0]) * (len(texts) - 1)
        pivot = risks.index(min(risks))
        merged += _vote(candidates, pivot, decide, empty) if len(passes) >= 3 else candidates[pivot]
    if empty and merged:
        merged[-1]["spot_after"] = empty
    return merged


def merged_words(results: list[dict], decide: Decide | None = None) -> list[dict]:
    """The words of several server responses, cleaned, merged and kept in order."""
    first = _clean_words(results[0])
    words = merge_passes([first] + [_clean_words(r, first) for r in results[1:]], decide)
    _clamp_durations(words)
    _keep_order(words)
    return words


def parse_result(
    result: dict | list[dict], language: str | None = None, decide: Decide | None = None
) -> Transcript:
    """Builds the transcript from one server response, or merges several passes."""
    results = result if isinstance(result, list) else [result]
    words = merged_words(results, decide)
    return Transcript(
        words=[
            WordEntry(
                text=w["text"],
                start=round(w["start"], 3),
                end=round(w["end"], 3),
                speaker=w["speaker"],
            )
            for w in words
        ],
        language=language or results[0].get("language"),
    )


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def transcribe(
    wav_path: str, language: str | None = None, num_speakers: int | None = None
) -> Transcript:
    """`num_speakers`, when known, is passed to diarization as the exact count."""
    return parse_result(*decode(wav_path, language, num_speakers))


def decode(
    wav_path: str, language: str | None = None, num_speakers: int | None = None
) -> tuple[list[dict], str | None]:
    """Every pass's server response, and the language they were decoded in.

    `parse_result` merges them; `review.review_transcript` merges them again
    with the clip's say on the disputed spots.
    """
    # Requests go one after another: the server handles one at a time anyway.
    wav = _read(wav_path)
    loaded = None if language else DETECTION_MODEL
    language = language or _detect_language(wav)
    results: list[dict] = [{}] * len(PASSES)
    for i in _pass_order(loaded or _loaded_model()):
        results[i] = _request(wav, language, *PASSES[i], num_speakers, diarize=i == 0)
    return results, language


# --- character names --------------------------------------------------------------

_HANGUL_BASE, _JUNG, _JONG = 0xAC00, 21, 28
# Particles and vocative endings a name can carry ("징징아", "스폰지밥한테").
_NAME_SUFFIXES = sorted(
    ["", "아", "야", "이", "가", "은", "는", "을", "를", "도", "의", "와", "과", "랑", "이랑",
     "한테", "에게", "하고", "님", "이가", "이는", "이도", "이를", "이야", "이의", "이한테", "만", "까지"],
    key=len, reverse=True,
)
# A name must keep at least this many syllables to be matched: two-syllable
# names ("뚱이", "진주") sit one jamo away from everyday words.
MIN_NAME_SYLLABLES = 3


def _jamo(text: str) -> str:
    """Hangul syllables split into their letters, so 스펀지밥 and 스폰지밥 differ by one."""
    out = []
    for ch in text:
        code = ord(ch) - _HANGUL_BASE
        if 0 <= code < 11172:
            out += [chr(0x1100 + code // (_JUNG * _JONG)), chr(0x1161 + code % (_JUNG * _JONG) // _JONG)]
            if code % _JONG:
                out.append(chr(0x11A7 + code % _JONG))
        else:
            out.append(ch)
    return "".join(out)


def _name_for(stem: str, names: list[tuple[str, str]]) -> str | None:
    """The name `stem` is a near miss of: same syllable count, one letter off (two for long names)."""
    jamo = _jamo(stem)
    best, best_dist = None, None
    for name, name_jamo in names:
        if len(stem) != len(name):
            continue
        dist = _edit_distance(jamo, name_jamo)
        if 0 < dist <= (1 if len(name) <= 3 else 2) and (best_dist is None or dist < best_dist):
            best, best_dist = name, dist
    return best


def correct_names(transcript: Transcript, names: list[str]) -> int:
    """Respells words that are a near miss of a character's name; returns how many.

    Whisper hears unfamiliar names as similar-sounding ones (스폰지밥 → 스펀지밥,
    징징이 → 진진이). The cast lookup already knows the right spellings, so a
    word one or two letters off a name — after peeling a particle — takes it.
    """
    known = [n for n in dict.fromkeys(n.replace(" ", "") for n in names) if len(n) >= MIN_NAME_SYLLABLES]
    if not known:
        return 0
    # A name ending in 이 drops it when called out: 징징이 → 징징아.
    spelled = set(known) | {n[:-1] + v for n in known if n.endswith("이") for v in ("아", "야")}
    targets = [(n, _jamo(n)) for n in sorted(spelled)]
    fixed = 0
    for w in transcript.words:
        lead, body, trail = re.match(r"^(\W*)(.*?)(\W*)$", w.text).groups()
        splits = [(body[: len(body) - len(x)], x) for x in _NAME_SUFFIXES if body.endswith(x)]
        if any(stem in spelled for stem, _ in splits):
            continue
        for stem, suffix in splits:
            if len(stem) < MIN_NAME_SYLLABLES:
                continue
            name = _name_for(stem, targets)
            if name:
                w.text = f"{lead}{name}{suffix}{trail}"
                fixed += 1
                break
    return fixed
