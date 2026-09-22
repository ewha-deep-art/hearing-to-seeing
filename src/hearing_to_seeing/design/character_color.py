"""Deciding which hue each *character* should prefer.

The speaker step (`speaker.py`) places hues on the circle but has no opinion
about which hue suits whom; this module supplies that opinion, from what the
lookup step found out about the cast.

The grounds for a colour, in order:

  1. **costume** — the colour the character actually wears. Preferred because
     the screen and the subtitle then explain each other: a viewer who sees a
     red robe and red subtitles needs no legend.
  2. **symbolic** — a colour argued from role and temperament, used only when
     the costume is unknown or has no colour to speak of. Most live-action
     casts wear black, grey and navy, so this is the common case, not the
     exception.
  3. none — nothing to argue from. The character carries no preference and
     the speaker step spreads them out like any other.

The model proposes an sRGB colour for each ground; it never sees OKLCH. The
tiering — is this colour vivid enough to count, which ground wins — is done
here, in code, on the OKLCH chroma of what it proposed, so the rule is the
same for every run and can be tested without a model.

Output is a hue and a weight per character, written onto the `Character`
itself. Nothing here touches a speaker: joining hues to diarization labels is
`with_character_hues()`, which does it through whatever identity strategy is
in force.
"""

import re

from hearing_to_seeing.design.oklch import rgb_to_oklch
from hearing_to_seeing.design.speaker import SpeakerCandidate, SpeakerStrategy
from hearing_to_seeing.knowledge.llm import LLM
from hearing_to_seeing.schema import Character, Transcript, WorkInfo

# Below this OKLCH chroma a colour is grey for our purposes: its hue is
# numerically defined but not something a viewer would recognise on screen, so
# it is no grounds for a preference. Black, white and most "navy" suits land
# under it (a muted navy suit measures about 0.05); a genuinely coloured garment
# clears it by a wide margin.
ACHROMATIC_CHROMA = 0.06

# How much weight each ground carries when two characters want the same hue.
# Costume is checkable against the screen; a symbolic colour is a reading.
BASIS_WEIGHT = {"costume": 1.0, "symbolic": 0.7}
# Leads keep their wish over bit parts.
IMPORTANCE_WEIGHT = {"main": 1.0, "supporting": 0.6}

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")

SYSTEM = """\
당신은 자막 디자인을 돕는 색채 전문가다. 영화·드라마 등장인물마다 자막 색의 근거가 될
색을 제안한다. 두 가지 근거를 각각 따로 답한다.

1. costume: 인물이 극중에서 실제로 입는 대표 의상의 색. 자료에 의상 색이 없거나,
   검정·회색·흰색·짙은 남색처럼 무채색에 가까우면 null.
2. symbolic: 인물의 성격·역할·관계가 연상시키는 상징색. 항상 하나를 제안한다.
   같은 작품 안의 인물들이 서로 구분되도록, 인물마다 다른 색상 계열을 고른다.

색은 sRGB 6자리 hex 로 답한다. 밝기는 신경 쓰지 않아도 된다 — 자막 시스템이 밝기를
고정하고 색상(hue)만 가져다 쓴다. 각 제안에는 한 줄 근거를 붙인다."""

PROMPT = """\
작품: {title}
{summary}
등장인물:
{characters}

인물마다 costume 색과 symbolic 색을 제안하라."""

CHARACTER_LINE = """\
- {name} ({importance}){actor}
  역할: {role}
  성격: {personality}
  관계: {relationships}
  외형·의상: {appearance}"""

COLOUR_SCHEMA = {
    "type": "object",
    "properties": {
        "hex": {"type": ["string", "null"], "description": "sRGB 6자리 hex, 예: #C0392B"},
        "evidence": {"type": ["string", "null"], "description": "한 줄 근거"},
    },
    "required": ["hex", "evidence"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "입력과 동일한 극중 이름"},
                    "costume": COLOUR_SCHEMA,
                    "symbolic": COLOUR_SCHEMA,
                    "confidence": {
                        "type": "number",
                        "description": "제안한 색이 이 인물에게 맞을 확률 0~1",
                    },
                },
                "required": ["name", "costume", "symbolic", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["characters"],
    "additionalProperties": False,
}


# --- colour arithmetic --------------------------------------------------------

def parse_hex(value: str | None) -> tuple[int, int, int] | None:
    match = _HEX.match((value or "").strip())
    if not match:
        return None
    digits = match.group(1)
    return tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))


def hue_of(value: str | None) -> float | None:
    """The OKLCH hue of an sRGB hex colour, or None if it is too grey to have one."""
    rgb = parse_hex(value)
    if rgb is None:
        return None
    _, chroma, hue = rgb_to_oklch(*rgb)
    return hue if chroma >= ACHROMATIC_CHROMA else None


def choose_basis(
    costume: str | None, symbolic: str | None,
) -> tuple[float | None, str]:
    """Applies the tiering: costume if it has colour, else symbolic, else none.

    Returns the hue and which ground it came from — "achromatic" when a
    costume colour was proposed but too grey to use and nothing symbolic
    stood in, so the JSON says why the character has no preference.
    """
    hue = hue_of(costume)
    if hue is not None:
        return hue, "costume"
    hue = hue_of(symbolic)
    if hue is not None:
        return hue, "symbolic"
    return None, "achromatic" if parse_hex(costume) else "none"


def hue_weight(character: Character) -> float:
    """How hard this character's preference pushes back when hues collide."""
    if character.hue is None:
        return 0.0
    return (
        BASIS_WEIGHT.get(character.hue_basis or "", 0.5)
        * IMPORTANCE_WEIGHT.get(character.importance, 0.6)
        * max(character.hue_confidence, 0.0)
    )


# --- the step ----------------------------------------------------------------

def apply_proposals(work: WorkInfo, proposals: list[dict]) -> None:
    """Writes the model's colour proposals onto `work`'s characters."""
    by_name = {_fold(p.get("name") or ""): p for p in proposals}
    for character in work.characters:
        proposal = by_name.get(_fold(character.name))
        if proposal is None:
            # Asked about but not answered: no preference, and say so.
            character.hue, character.hue_basis = None, "none"
            character.hue_evidence, character.hue_confidence = None, 0.0
            continue

        costume = proposal.get("costume") or {}
        symbolic = proposal.get("symbolic") or {}
        hue, basis = choose_basis(costume.get("hex"), symbolic.get("hex"))
        evidence = {"costume": costume, "symbolic": symbolic}.get(basis, {}).get("evidence")

        character.hue = round(hue, 1) if hue is not None else None
        character.hue_basis = basis
        character.hue_evidence = (evidence or "").strip() or None
        character.hue_confidence = (
            float(proposal.get("confidence") or 0.0) if hue is not None else 0.0
        )


def suggest_hues(work: WorkInfo, llm: LLM, *, refresh: bool = False) -> WorkInfo:
    """Gives every character in `work` a preferred hue, or a reason for none.

    One request for the whole cast rather than one per character, so the
    model can keep the symbolic colours apart from each other — the same
    reason a designer picks a palette rather than one colour at a time.
    Skipped when every character has already been decided, unless `refresh`.
    """
    undecided = [c for c in work.characters if c.hue_basis is None]
    if not work.characters or (not undecided and not refresh):
        return work

    prompt = PROMPT.format(
        title=work.title or "(제목 미상)",
        summary=f"요약: {work.summary}\n" if work.summary else "",
        characters="\n".join(
            CHARACTER_LINE.format(
                name=c.name,
                importance="주연" if c.importance == "main" else "조연",
                actor=f", 배우 {c.actor}" if c.actor else "",
                role=c.role or "-",
                personality=c.personality or "-",
                relationships=c.relationships or "-",
                appearance=c.appearance or "-",
            )
            for c in work.characters
        ),
    )
    answer = llm.structured(SYSTEM, prompt, SCHEMA)
    apply_proposals(work, answer.get("characters") or [])
    return work


# --- joining hues to speakers ------------------------------------------------

def with_character_hues(
    strategy: SpeakerStrategy | None, work: WorkInfo | None,
) -> SpeakerStrategy | None:
    """Wraps an identity strategy so its named speakers inherit their character's hue.

    The strategy says *who* a speaker is; the character sheet says what
    colour that person should lean towards. Keeping the two apart is what
    lets a manual `--speaker-map` — which knows names and nothing about
    colour — still come out in colours that mean something.

    A candidate that already states a hue keeps it: a strategy with its own
    colour evidence (a frame sample, say) outranks a description in a wiki.
    """
    if strategy is None or work is None or not work.characters:
        return strategy

    def resolve(transcript: Transcript, media_path: str) -> dict[str, SpeakerCandidate]:
        candidates = strategy(transcript, media_path)
        for candidate in candidates.values():
            if candidate.name is None or candidate.preferred_hue is not None:
                continue
            character = work.find(candidate.name)
            if character is None or character.hue is None:
                continue
            candidate.preferred_hue = character.hue
            candidate.hue_weight = hue_weight(character)
            reason = f"색상 근거({character.hue_basis}): {character.hue_evidence or '-'}"
            candidate.note = f"{candidate.note}; {reason}" if candidate.note else reason
        return candidates

    return resolve


def _fold(name: str) -> str:
    return "".join(name.split()).casefold()
