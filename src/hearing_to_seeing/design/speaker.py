"""Deciding which colour each speaker's subtitles are drawn in.

In pipeline order:

  1. `lookup_characters(title)` — which work the clip title names, what
     Namu wiki and Wikipedia say about its cast, and a colour per character
     (costume if the documents give one, else a symbolic colour). Needs no
     transcript, so the pipeline runs it alongside STT.
  2. `assign_speaker_colors(transcript, characters)` — which speaker is which
     character, from the dialogue; then a hue for every speaker, placed as
     near its character's colour as it can get while staying `MIN_HUE_GAP`
     from everyone else.

Colours are OKLCH at one fixed lightness, so every hue reads equally well
against video and only the hue is free. Speakers matched to a character get
full chroma; the rest get a muted colour of their own, which reads as "not
one of the named characters". Without an API key, or when any model call
fails, every speaker simply gets an evenly spaced hue.
"""

import math
import re
import sys
from collections import Counter
from functools import lru_cache

from hearing_to_seeing.knowledge import llm, prompts
from hearing_to_seeing.knowledge.retrieve import retrieve
from hearing_to_seeing.schema import Transcript

# --- colour model --------------------------------------------------------------

# One lightness for every speaker colour: legibility is settled once, here.
# 0.75 sits in the band of the Wong palette the project used before, and still
# leaves every hue at least 0.128 chroma, so none comes out washed grey.
# TODO: L·채도 값은 실제 영상 위에서 확인 후 확정 필요 — 밝은 장면에서의 가독성 미검증.
LIGHTNESS = 0.75
# Chroma as a fraction of the most each hue can hold at LIGHTNESS.
MAIN_CHROMA_RATIO = 0.95
MINOR_CHROMA_RATIO = 0.32
# How far apart two speakers' hues must stay, loosened automatically when
# there are more speakers than the circle can space that widely.
MIN_HUE_GAP = 40.0
# Below this OKLCH chroma a proposed colour is grey: no grounds for a hue.
ACHROMATIC_CHROMA = 0.06

# Colour every word starts out in, before it is spoken. No speaker colour can
# collide with it: they all carry chroma.
BASE_COLOUR = "&H00FFFFFF"  # white

# How sure the speaker → character answer must be to be used. A wrong colour
# is worse than an arbitrary one: a viewer who cannot hear has no way to catch it.
# TODO: 임계값 0.7은 경험값 — 매핑 정확도를 측정해 재조정 필요.
MIN_CONFIDENCE = 0.7

# Who keeps their preferred hue when two collide: leads over bit parts, and a
# costume (checkable on screen) over a symbolic reading.
IMPORTANCE_WEIGHT = {"main": 1.0, "supporting": 0.6}
BASIS_WEIGHT = {"costume": 1.0, "symbolic": 0.7}

# --- dialogue handling ---------------------------------------------------------

# Names shorter than this are not looked for: a single syllable matches half
# the words in a scene.
MIN_NAME_LENGTH = 2
# A reply this soon after a name was spoken is taken to be from the person named.
REPLY_WINDOW = 4.0
# How much dialogue the model sees; the opening is kept whole, the rest sampled.
MAX_DIALOGUE_CHARS = 9000
# Speakers with fewer words than this are not asked about.
MIN_WORDS = 3


def _warn(step: str, exc: BaseException) -> None:
    # The lookup is an enhancement: when it fails the subtitles still come out,
    # in evenly spaced colours, so the error is reported rather than raised.
    print(f"… {step} 실패, 기본 색상으로 진행: {exc}", file=sys.stderr, flush=True)


def _fold(name: str) -> str:
    return "".join(name.split()).casefold()


# --- OKLCH ↔ sRGB (Björn Ottosson's matrices) ------------------------------------

def _to_linear(v: float) -> float:
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _from_linear(v: float) -> float:
    return 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def _oklch_to_linear_rgb(L: float, C: float, hue: float) -> tuple[float, float, float]:
    a, b = C * math.cos(math.radians(hue)), C * math.sin(math.radians(hue))
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )


def _rgb_to_oklch(r: int, g: int, b: int) -> tuple[float, float, float]:
    """8-bit sRGB to (L, C, hue in degrees)."""
    r, g, b = (_to_linear(v / 255) for v in (r, g, b))
    l, m, s = (
        math.copysign(abs(v) ** (1 / 3), v)
        for v in (
            0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
            0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
            0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
        )
    )
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    A = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    B = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, math.hypot(A, B), math.degrees(math.atan2(B, A)) % 360


@lru_cache(maxsize=4096)
def _max_chroma(L: float, hue: float) -> float:
    """The most saturated `hue` can be at lightness `L` and stay in sRGB.

    The gamut boundary has no closed form in OKLCH, so binary search.
    """
    low, high = 0.0, 0.5
    for _ in range(24):
        mid = (low + high) / 2
        if all(-1e-6 <= v <= 1 + 1e-6 for v in _oklch_to_linear_rgb(L, mid, hue)):
            low = mid
        else:
            high = mid
    return low


def hue_to_ass(hue: float, chroma_ratio: float = MAIN_CHROMA_RATIO) -> str:
    """The ASS colour (&H00BBGGRR) for `hue` at LIGHTNESS."""
    hue %= 360
    chroma = _max_chroma(LIGHTNESS, hue) * chroma_ratio
    r, g, b = (
        round(255 * _from_linear(min(1.0, max(0.0, v))))
        for v in _oklch_to_linear_rgb(LIGHTNESS, chroma, hue)
    )
    return f"&H00{b:02X}{g:02X}{r:02X}"


def hex_hue(value: str | None) -> float | None:
    """The OKLCH hue of an sRGB hex colour, or None if it is too grey to have one."""
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", (value or "").strip())
    if not match:
        return None
    digits = match.group(1)
    _, chroma, hue = _rgb_to_oklch(*(int(digits[i:i + 2], 16) for i in (0, 2, 4)))
    return hue if chroma >= ACHROMATIC_CHROMA else None


# --- hue assignment ------------------------------------------------------------

def hue_distance(a: float, b: float) -> float:
    """The shorter way round the circle between two hues, in degrees."""
    apart = abs(a - b) % 360
    return min(apart, 360 - apart)


def _nearest_free_hue(preferred: float, placed: list[float], gap: float) -> float:
    """The hue closest to `preferred` that clears `gap` from everything placed.

    When the preference is blocked, the nearest legal spot is always flush
    against whatever blocks it, so only those spots are tested.
    """
    def free(hue: float) -> bool:
        return all(hue_distance(hue, other) >= gap - 1e-9 for other in placed)

    candidates = [preferred % 360] + [(p + d) % 360 for p in placed for d in (gap, -gap)]
    return min(
        (h for h in candidates if free(h)),
        key=lambda h: (hue_distance(h, preferred), h),
        default=_widest_gap_midpoint(placed),
    )


def _widest_gap_midpoint(placed: list[float]) -> float:
    """The emptiest spot on the circle — where a speaker with no wish goes."""
    if not placed:
        return 0.0
    ordered = sorted(placed)
    sizes = [((ordered[(i + 1) % len(ordered)] - h) % 360 or 360, h) for i, h in enumerate(ordered)]
    size, start = max(sizes)
    return (start + size / 2) % 360


def assign_hues(labels: list[str], preferred: dict[str, float] | None = None) -> dict[str, float]:
    """Places every label on the hue circle.

    `preferred` is taken in its own order — most important first — each as
    close to its wish as the ones already placed allow; labels without a wish
    drop into the widest remaining gap. With no wishes at all this is an even
    spread, which keeps an unidentified cast distinguishable anyway.
    """
    labels = sorted(set(labels))
    preferred = {label: hue for label, hue in (preferred or {}).items() if label in labels}
    if not preferred:
        return {label: i * 360 / len(labels) for i, label in enumerate(labels)}

    # Dividing by one more than the count leaves slack: placements land flush
    # against each other, and an exactly-tight circle can strand the last one.
    gap = min(MIN_HUE_GAP, 360 / (len(labels) + 1))
    hues: dict[str, float] = {}
    for label in list(preferred) + [l for l in labels if l not in preferred]:
        target = preferred.get(label, _widest_gap_midpoint(list(hues.values())))
        hues[label] = _nearest_free_hue(target, list(hues.values()), gap)
    return hues


# --- 1. lookup: title → cast with colours (no transcript needed) ----------------

def lookup_characters(title: str | None) -> list[dict] | None:
    """The cast of the work `title` names, each with a preferred `hue`.

    Returns the model's character dicts (see `prompts.CAST_SCHEMA`) with a
    `hue` key added — None when the colour it proposed is too grey. Returns
    None without an API key, without a recognisable work, or on any failure.
    """
    if not title or not llm.configured():
        return None
    try:
        work = llm.ask(prompts.TITLE_SYSTEM, prompts.TITLE_PROMPT.format(title=title), prompts.TITLE_SCHEMA)
        work_title = (work.get("title") or "").strip()
        if not work_title:
            return None
        print(f"… 작품 검색 중: {work_title!r}", file=sys.stderr, flush=True)
        documents = retrieve(work_title) or prompts.NO_DOCUMENTS
        cast = llm.ask(
            prompts.CAST_SYSTEM,
            prompts.CAST_PROMPT.format(title=work_title, documents=documents),
            prompts.CAST_SCHEMA,
        )
    except Exception as exc:  # network, quota, bad key — anything
        _warn("작품·인물 검색", exc)
        return None

    characters = [c for c in cast.get("characters") or [] if (c.get("name") or "").strip()]
    for c in characters:
        c["hue"] = hex_hue(c.get("color"))
    print(f"… 인물 {len(characters)}명 확보", file=sys.stderr, flush=True)
    return characters


# --- 2. speaker → character ------------------------------------------------------

def _turns(transcript: Transcript) -> list[tuple[str, float, float, str]]:
    """Consecutive same-speaker words joined into (speaker, start, end, text)."""
    turns: list[tuple[str, float, float, str]] = []
    for w in transcript.words:
        if turns and turns[-1][0] == w.speaker:
            speaker, start, _, text = turns[-1]
            turns[-1] = (speaker, start, w.end, f"{text} {w.text}")
        else:
            turns.append((w.speaker, w.start, w.end, w.text))
    return turns


def _anchors(turns: list[tuple[str, float, float, str]], characters: list[dict]) -> str:
    """Who named whom, counted by rule — constraints for the model.

    A speaker who says a character's name is almost never that character, and
    whoever answers right after very often is. Korean attaches particles to
    names ("기택아"), so a token matches by prefix.
    """
    said: Counter = Counter()
    answered: Counter = Counter()
    for i, (speaker, _, end, text) in enumerate(turns):
        tokens = text.split()
        for c in characters:
            names = {n.strip() for n in [c["name"], *c.get("aliases", [])]}
            names = [n for n in names if len(n) >= MIN_NAME_LENGTH]
            if not any(t.startswith(n) for t in tokens for n in names):
                continue
            said[(speaker, c["name"])] += 1
            if i + 1 < len(turns):
                replier, reply_start, _, _ = turns[i + 1]
                if replier != speaker and reply_start - end <= REPLY_WINDOW:
                    answered[(replier, c["name"])] += 1

    lines = [
        f"- {s} 이(가) '{c}' 을(를) {n}번 부름 → {s} 은(는) {c} 이(가) 아닐 가능성 높음"
        for (s, c), n in said.most_common()
    ] + [
        f"- '{c}' 이(가) 불린 직후 {s} 이(가) {n}번 대답함 → {s} 이(가) {c} 일 가능성 높음"
        for (s, c), n in answered.most_common()
    ]
    return "\n".join(lines) or "(없음)"


def _dialogue(turns: list[tuple[str, float, float, str]]) -> str:
    """The dialogue as labelled turns: the opening whole, the rest sampled evenly.

    The opening is where people are introduced and addressed by name.
    """
    lines = [f"[{start:06.1f}] {speaker}: {text}" for speaker, start, _, text in turns]
    if len("\n".join(lines)) <= MAX_DIALOGUE_CHARS:
        return "\n".join(lines)

    head, used = [], 0
    for line in lines:
        if used + len(line) > MAX_DIALOGUE_CHARS // 2:
            break
        head.append(line)
        used += len(line) + 1
    rest = lines[len(head):]
    average = max(1, sum(len(line) + 1 for line in rest) // len(rest))
    keep = max(1, (MAX_DIALOGUE_CHARS - used) // average)
    return "\n".join(head + ["…"] + rest[:: max(1, len(rest) // keep)][:keep])


def identify_speakers(transcript: Transcript, characters: list[dict]) -> dict[str, dict]:
    """Label → {"character", "confidence", "reason"} for the speakers the model named.

    Returns {} on any failure. Confidence is not filtered here.
    """
    counts = Counter(w.speaker for w in transcript.words)
    labels = sorted(label for label, n in counts.items() if n >= MIN_WORDS)
    if not labels or not characters:
        return {}

    turns = _turns(transcript)
    prompt = prompts.SPEAKERS_PROMPT.format(
        characters="\n".join(
            prompts.CHARACTER_LINE.format(
                name=c["name"],
                aliases=f" (호칭: {', '.join(c['aliases'])})" if c.get("aliases") else "",
                importance="주연" if c.get("importance") == "main" else "조연",
                role=c.get("role") or "-",
                personality=c.get("personality") or "-",
                relationships=c.get("relationships") or "-",
            )
            for c in characters
        ),
        anchors=_anchors(turns, characters),
        dialogue=_dialogue(turns),
        labels=", ".join(labels),
    )
    print(f"… 화자 → 인물 추론 중: {', '.join(labels)}", file=sys.stderr, flush=True)
    try:
        answer = llm.ask(prompts.SPEAKERS_SYSTEM, prompt, prompts.SPEAKERS_SCHEMA)
    except Exception as exc:
        _warn("화자 → 인물 추론", exc)
        return {}

    by_name = {}
    for c in characters:
        for name in [*c.get("aliases", []), c["name"]]:  # the real name wins a clash
            by_name[_fold(name)] = c
    matches: dict[str, dict] = {}
    for entry in answer.get("speakers") or []:
        label = entry.get("label")
        character = by_name.get(_fold(entry.get("character") or ""))
        if label in labels and character is not None and label not in matches:
            matches[label] = {
                "character": character,
                "confidence": min(max(float(entry.get("confidence") or 0.0), 0.0), 1.0),
                "reason": (entry.get("reasoning") or "").strip(),
            }
    return matches


# --- resolution ------------------------------------------------------------------

def assign_speaker_colors(
    transcript: Transcript, characters: list[dict] | None = None,
) -> dict[str, dict]:
    """One profile per speaker: {"color", "name", "confidence", "reason"}.

    With `characters` (from `lookup_characters`), speakers the model matches
    with at least MIN_CONFIDENCE take their character's hue, vividly; the rest
    are muted. Without, every speaker gets an evenly spaced vivid hue.
    """
    labels = transcript.speakers()
    matches = {
        label: match
        for label, match in (identify_speakers(transcript, characters) if characters else {}).items()
        if match["confidence"] >= MIN_CONFIDENCE
    }

    def weight(match: dict) -> float:
        c = match["character"]
        return (
            IMPORTANCE_WEIGHT.get(c.get("importance"), 0.6)
            * BASIS_WEIGHT.get(c.get("color_basis"), 0.7)
            * float(c.get("color_confidence") or 0.0)
        )

    preferred = {
        label: match["character"]["hue"]
        for label, match in sorted(matches.items(), key=lambda kv: (-weight(kv[1]), kv[0]))
        if match["character"].get("hue") is not None
    }
    hues = assign_hues(labels, preferred)

    profiles = {}
    for label in labels:
        match = matches.get(label)
        if match is None:
            # Muted only means something when someone else was named.
            ratio = MINOR_CHROMA_RATIO if matches else MAIN_CHROMA_RATIO
            profiles[label] = {"color": hue_to_ass(hues[label], ratio), "name": None,
                               "confidence": 0.0, "reason": None}
            continue
        c = match["character"]
        profiles[label] = {
            "color": hue_to_ass(hues[label]),
            "name": c["name"],
            "confidence": match["confidence"],
            "reason": f"{match['reason']} / 색({c.get('color_basis')}): {c.get('color_reason') or '-'}",
        }
    return profiles


def speaker_colors(transcript: Transcript) -> dict[str, str]:
    """Label → ASS colour, as the pipeline decided it.

    Read from `speaker_profiles` so the ASS converter and the web legend show
    the same colours. A transcript that never went through the pipeline gets
    the evenly spaced fallback.
    """
    profiles = transcript.speaker_profiles or assign_speaker_colors(transcript)
    return {label: profile.get("color") or BASE_COLOUR for label, profile in profiles.items()}
