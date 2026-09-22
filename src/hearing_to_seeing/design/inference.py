"""Working out who each speaker is from what they say.

The text path of speaker → character mapping. It has two layers, and the
order matters:

  1. **Anchors, by rule.** Characters get named in dialogue — "아버지",
     "사장님", "기택아" — and a name tells two things for certain enough to
     act on: the speaker who says it is almost never that character, and the
     speaker who answers next very often is. These are counted, not judged,
     so they can be checked by hand and cannot be argued away.
  2. **Inference, by model.** With the anchors as constraints, the model reads
     the dialogue against the cast sheet — role, temperament, relationships,
     who speaks formally to whom — and names a character per speaker, with a
     confidence. It is told to leave a speaker blank rather than guess: the
     colour step drops anything under its threshold, and a wrong name is
     worse than none because a viewer cannot hear that it is wrong.

The result is a `SpeakerStrategy`, the same shape as the manual mapping, so
`speaker.resolve_speakers()` treats the two identically. Nothing here decides
a colour; `character_color.with_character_hues()` attaches those afterwards.

Diarization sometimes splits one voice into two labels. When the model says
two labels are the same character, both keep the name — merging labels is
the confirmation UI's job, not this step's — and the note says they collide.
"""

from collections import Counter
from dataclasses import dataclass, field

from hearing_to_seeing.design.speaker import SpeakerCandidate, SpeakerStrategy
from hearing_to_seeing.knowledge import progress
from hearing_to_seeing.knowledge.llm import LLM
from hearing_to_seeing.schema import Character, Transcript, WorkInfo

SOURCE = "text"

# A name has to be at least this long to be looked for. Korean given names are
# two syllables and titles ("사장님") longer; a single syllable matches half
# the words in a scene.
MIN_NAME_LENGTH = 2

# A reply this soon after a name was spoken is taken to be from the person
# named. Longer than that and the floor may have passed to anyone.
REPLY_WINDOW = 4.0

# How much of the dialogue the model sees. A twenty-minute clip runs to a few
# thousand words; the opening turns carry the introductions and most of the
# names, so they are kept whole and the rest is sampled.
MAX_DIALOGUE_CHARS = 9000
# Speakers with fewer words than this are not asked about — a grunt or a
# single misattributed word is not evidence of anything.
MIN_WORDS = 3

SYSTEM = """\
당신은 영화·드라마 클립의 음성 인식 결과에서 각 화자(SPEAKER_xx)가 어느 등장인물인지 추론하는
도구다. 근거는 대사 내용, 말투(존댓말·반말의 방향), 호칭(가족·직함), 인물 간 관계, 그리고
규칙으로 미리 찾아 둔 '호명 앵커'다.

원칙:
- 호명 앵커는 강한 제약이다. 어떤 화자가 어떤 인물의 이름·호칭을 부르면, 그 화자는 그 인물이
  아닐 가능성이 매우 높고, 바로 다음에 대답한 화자가 그 인물일 가능성이 높다.
- 확신이 없으면 character 를 null 로 두고 confidence 를 낮게 적는다. 틀린 이름은 이름이 없는 것보다
  나쁘다 — 자막을 보는 사람은 소리를 듣지 못해 오류를 알아챌 수 없다.
- 한 인물은 보통 한 화자다. 두 화자가 같은 인물로 보이면(화자 분리 오류) 둘 다 그 인물로 적고
  reasoning 에 그렇게 판단한 이유를 밝힌다.
- 음성 인식 오류가 있을 수 있다. 발음이 비슷한 이름은 같은 것으로 본다.
- character 에는 등장인물 목록에 있는 극중 이름을 그대로 쓴다."""

PROMPT = """\
작품: {title}
{scene}
등장인물 (장면 등장 인물이 먼저):
{characters}

호명 앵커 (규칙으로 탐지):
{anchors}

대사 (시간 순서):
{dialogue}

화자마다 어느 인물인지 답하라. 대상 화자: {labels}"""

CHARACTER_LINE = """\
- {name}{aliases} ({importance})
  역할: {role} / 성격: {personality} / 관계: {relationships}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "SPEAKER_xx"},
                    "character": {
                        "type": ["string", "null"],
                        "description": "등장인물 목록의 극중 이름, 모르면 null",
                    },
                    "confidence": {"type": "number", "description": "0~1"},
                    "reasoning": {"type": "string", "description": "한두 문장의 근거"},
                },
                "required": ["label", "character", "confidence", "reasoning"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["speakers"],
    "additionalProperties": False,
}


@dataclass
class Anchors:
    """Who named whom, counted from the transcript.

    `said[(speaker, character)]` — how often `speaker` spoke a name of
    `character`. `answered[(speaker, character)]` — how often `speaker` took
    the next turn within `REPLY_WINDOW` of that name being spoken by someone
    else.
    """

    said: Counter = field(default_factory=Counter)
    answered: Counter = field(default_factory=Counter)

    def is_empty(self) -> bool:
        return not self.said and not self.answered

    def describe(self) -> str:
        """The anchors as lines for the prompt, strongest first."""
        if self.is_empty():
            return "(없음)"
        lines = [
            f"- {speaker} 이(가) '{character}' 을(를) {count}번 부름 → {speaker} 은(는) {character} 이(가) 아닐 가능성 높음"
            for (speaker, character), count in self.said.most_common()
        ]
        lines += [
            f"- '{character}' 이(가) 불린 직후 {speaker} 이(가) {count}번 대답함 → {speaker} 이(가) {character} 일 가능성 높음"
            for (speaker, character), count in self.answered.most_common()
        ]
        return "\n".join(lines)


def _names_of(character: Character) -> list[str]:
    names = [character.name, *character.aliases]
    return [n for n in dict.fromkeys(n.strip() for n in names) if len(n) >= MIN_NAME_LENGTH]


def _mentioned(text: str, names: list[str]) -> bool:
    # Korean attaches particles to the word ("기택아", "사장님은"), so a
    # prefix match on the token is the right test, not equality.
    return any(text.startswith(name) for name in names)


def detect_anchors(transcript: Transcript, work: WorkInfo) -> Anchors:
    """Counts who spoke each character's name, and who answered to it."""
    anchors = Anchors()
    lookup = [(c.name, _names_of(c)) for c in work.characters]
    lookup = [(name, names) for name, names in lookup if names]
    if not lookup:
        return anchors

    turns = transcript.turns()
    for index, (speaker, _, end, text) in enumerate(turns):
        for character, names in lookup:
            if not any(_mentioned(token, names) for token in text.split()):
                continue
            anchors.said[(speaker, character)] += 1
            if index + 1 < len(turns):
                replier, reply_start, _, _ = turns[index + 1]
                if replier != speaker and reply_start - end <= REPLY_WINDOW:
                    anchors.answered[(replier, character)] += 1
    return anchors


def dialogue_text(transcript: Transcript, limit: int = MAX_DIALOGUE_CHARS) -> str:
    """The dialogue as labelled turns, whole at the start and sampled after.

    The opening is where people are introduced and addressed by name, so it
    is kept in full; past the budget, turns are taken evenly from the rest so
    the model still sees every stretch of the clip.
    """
    turns = [f"[{start:06.1f}] {speaker}: {text}" for speaker, start, _, text in transcript.turns()]
    text = "\n".join(turns)
    if len(text) <= limit:
        return text

    head_budget = limit // 2
    head: list[str] = []
    used = 0
    for line in turns:
        if used + len(line) > head_budget:
            break
        head.append(line)
        used += len(line) + 1

    rest = turns[len(head):]
    tail_budget = limit - used
    if not rest or tail_budget <= 0:
        return "\n".join(head)
    average = max(1, sum(len(line) + 1 for line in rest) // len(rest))
    keep = max(1, tail_budget // average)
    stride = max(1, len(rest) // keep)
    sampled = rest[::stride][:keep]
    return "\n".join(head + ["…"] + sampled)


def infer_from_text(work: WorkInfo, llm: LLM) -> SpeakerStrategy:
    """Builds the strategy: anchors by rule, then the model, per speaker."""

    def resolve(transcript: Transcript, media_path: str) -> dict[str, SpeakerCandidate]:
        counts = Counter(word.speaker for word in transcript.words)
        labels = sorted(label for label, n in counts.items() if n >= MIN_WORDS)
        if not labels or not work.characters:
            return {}

        anchors = detect_anchors(transcript, work)
        prompt = PROMPT.format(
            title=work.title or "(제목 미상)",
            scene=f"장면: {work.scene}\n" if work.scene else "",
            characters="\n".join(
                CHARACTER_LINE.format(
                    name=c.name,
                    aliases=f" (호칭: {', '.join(c.aliases)})" if c.aliases else "",
                    importance="주연" if c.importance == "main" else "조연",
                    role=c.role or "-",
                    personality=c.personality or "-",
                    relationships=c.relationships or "-",
                )
                for c in work.characters
            ),
            anchors=anchors.describe(),
            dialogue=dialogue_text(transcript),
            labels=", ".join(labels),
        )
        progress(f"화자 → 인물 추론 중: {', '.join(labels)}")
        answer = llm.structured(SYSTEM, prompt, SCHEMA)
        return _candidates(answer.get("speakers") or [], labels, work, anchors)

    return resolve


def _candidates(
    answers: list[dict], labels: list[str], work: WorkInfo, anchors: Anchors,
) -> dict[str, SpeakerCandidate]:
    candidates: dict[str, SpeakerCandidate] = {}
    for entry in answers:
        label = (entry.get("label") or "").strip()
        if label not in labels or label in candidates:
            continue
        character = work.find((entry.get("character") or "").strip())
        reasoning = (entry.get("reasoning") or "").strip()
        if character is None:
            # Left blank, or a name that is not in the cast: either way there
            # is no character to attach — recorded so the JSON shows the model
            # was asked and declined.
            candidates[label] = SpeakerCandidate(
                label=label, name=None, confidence=0.0, source=SOURCE,
                note=reasoning or "인물 미특정",
            )
            continue
        candidates[label] = SpeakerCandidate(
            label=label,
            name=character.name,
            confidence=min(max(float(entry.get("confidence") or 0.0), 0.0), 1.0),
            source=SOURCE,
            note=reasoning or None,
        )

    # Two labels on one character: both keep it (diarization may have split a
    # voice) but each is told about the other, so the collision is visible in
    # the JSON and to whoever merges labels later.
    by_name: dict[str, list[str]] = {}
    for label, candidate in candidates.items():
        if candidate.name:
            by_name.setdefault(candidate.name, []).append(label)
    for name, shared in by_name.items():
        if len(shared) < 2:
            continue
        for label in shared:
            others = ", ".join(other for other in shared if other != label)
            collision = f"{others} 도 {name} 으로 추정 — 화자 병합 검토"
            candidate = candidates[label]
            candidate.note = f"{candidate.note}; {collision}" if candidate.note else collision

    # The strongest anchor is recorded alongside the model's reasoning, so a
    # reader can see whether the two agree.
    for label, candidate in candidates.items():
        if candidate.name is None:
            continue
        replies = anchors.answered.get((label, candidate.name), 0)
        if replies:
            evidence = f"'{candidate.name}' 호명 뒤 {replies}회 응답"
            candidate.note = f"{candidate.note}; {evidence}" if candidate.note else evidence
    return candidates
