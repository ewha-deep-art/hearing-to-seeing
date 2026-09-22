"""Finding out who is in the work — the generation half of the lookup step.

Given a work (and, when known, a scene) and the documents `retrieve.py`
fetched about it, this asks the model to read them and write the cast as
*characters*: names and the things they are called, who plays them, what
they are like, how they relate to each other, and how they dress. The last
two are what the colour step argues from; the aliases and relationships are
what the mapping step argues from.

The model is told to use the documents and nothing else, and to leave a field
null rather than fill it from memory: the documents are on record (their
URLs become `sources`) and a remembered fact is not. When retrieval found
nothing at all the model is still asked, from memory, and says so in its
confidence — a cast from memory beats no cast, and the confidence tells the
speaker step how far to trust it.

The prompt asks for the characters that appear in the scene first: a clip
holds one scene, and a scene holds three characters where the film holds
twenty, so a cast cut down to the scene is the single biggest simplification
the whole identification problem gets.
"""

from collections.abc import Sequence

from hearing_to_seeing.knowledge.llm import LLM
from hearing_to_seeing.knowledge.retrieve import Document
from hearing_to_seeing.knowledge.title import WorkQuery
from hearing_to_seeing.schema import Character, WorkInfo

# A transcript excerpt this long is enough to recognise a scene by and short
# enough not to dominate the prompt.
EXCERPT_CHARS = 1200

SYSTEM = """\
당신은 영화·드라마 등장인물 조사 도구다. 배우가 아니라 **극중 인물**을 정리한다.
주어진 문서에 있는 내용만 쓰고, 문서에 없는 항목은 null 로 둔다. 인물을 지어내지 않는다.
문서가 하나도 없을 때만 알고 있는 지식으로 답하되, confidence 를 그만큼 낮게 적는다.
성격·역할·관계·의상(색) 은 문서에 있는 만큼 자세히 적는다."""

PROMPT = """\
작품 정보:
- 제목: {title}{year}
- 장면 힌트: {scene_hint}

{documents}
{excerpt}
위 문서를 바탕으로 이 작품의 등장인물을 정리하라. 인물마다:
1. 극중 이름 (자막에 표시될 이름). 배우 이름은 actor 에.
2. 대사에서 불리는 다른 호칭 (예: 아버지, 사장님, 전하, 촌장, 나으리) — aliases 에.
3. 주연/조연 (importance).
4. 역할 한 줄 (role), 성격 (personality), 다른 인물과의 관계 (relationships).
5. 외형·의상 (appearance). 대표 의상 색이나 상징색이 문서에 있으면 반드시 적는다.
6. 장면 힌트나 대사가 가리키는 장면에 등장하는지 (in_scene).

장면 힌트가 있으면 그 장면의 인물을 먼저 적는다. summary 는 작품 요약 한 단락,
scene 은 장면 힌트가 가리키는 장면의 설명 (특정할 수 없으면 null)."""

DOCUMENTS_BLOCK = """\
참고 문서 ({count}건):
{documents}"""

DOCUMENT_BLOCK = """\
<document source="{source}" title="{title}" url="{url}">
{text}
</document>"""

NO_DOCUMENTS = "참고 문서: 없음 (검색 실패). 알고 있는 범위에서 답하고 confidence 를 낮게 적는다."

EXCERPT_BLOCK = """
아래는 이 클립의 음성 인식 결과 일부다. 장면을 특정하는 데 참고하되, 인식 오류가 있을 수 있다.
<transcript>
{excerpt}
</transcript>
"""

CHARACTER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "극중 이름"},
        "aliases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "대사에서 불리는 다른 호칭들",
        },
        "actor": {"type": ["string", "null"]},
        "importance": {"type": "string", "enum": ["main", "supporting"]},
        "role": {"type": ["string", "null"], "description": "극중 역할 한 줄"},
        "personality": {"type": ["string", "null"]},
        "relationships": {"type": ["string", "null"]},
        "appearance": {
            "type": ["string", "null"],
            "description": "외형·의상 묘사. 대표 의상 색이 있으면 명시",
        },
        "in_scene": {
            "type": "boolean",
            "description": "장면 힌트/대사가 가리키는 장면에 등장하는지",
        },
    },
    "required": [
        "name", "aliases", "actor", "importance", "role", "personality",
        "relationships", "appearance", "in_scene",
    ],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"], "description": "확인된 정식 제목"},
        "year": {"type": ["integer", "null"]},
        "summary": {"type": ["string", "null"], "description": "작품 요약 한 단락"},
        "scene": {
            "type": ["string", "null"],
            "description": "장면 힌트가 가리키는 장면 설명. 특정 못하면 null",
        },
        "confidence": {
            "type": "number",
            "description": "이 작품이 맞고 인물 목록이 정확할 확률 0~1",
        },
        "characters": {"type": "array", "items": CHARACTER_SCHEMA},
    },
    "required": ["title", "year", "summary", "scene", "confidence", "characters"],
    "additionalProperties": False,
}


def retrieve_characters(
    query: WorkQuery,
    llm: LLM,
    documents: Sequence[Document] = (),
    excerpt: str | None = None,
) -> WorkInfo:
    """Reads `documents` and returns the work's cast, scene characters first."""
    if documents:
        documents_text = DOCUMENTS_BLOCK.format(
            count=len(documents),
            documents="\n\n".join(
                DOCUMENT_BLOCK.format(source=d.source, title=d.title, url=d.url, text=d.text)
                for d in documents
            ),
        )
    else:
        documents_text = NO_DOCUMENTS

    prompt = PROMPT.format(
        title=query.title,
        year=f" ({query.year})" if query.year else "",
        scene_hint=query.scene_hint or "(없음)",
        documents=documents_text,
        excerpt=EXCERPT_BLOCK.format(excerpt=excerpt[:EXCERPT_CHARS]) if excerpt else "",
    )
    answer = llm.structured(SYSTEM, prompt, SCHEMA)

    characters = [
        _character(entry)
        for entry in answer.get("characters") or []
        if (entry.get("name") or "").strip()
    ]
    # Scene characters first, then leads — the order the mapping step should
    # try candidates in, and the order a person reading the JSON wants.
    characters.sort(key=lambda c: (not c[1], c[0].importance != "main"))

    return WorkInfo(
        title=(answer.get("title") or "").strip() or query.title,
        year=answer.get("year") or query.year,
        scene_hint=query.scene_hint,
        scene=(answer.get("scene") or "").strip() or None,
        summary=(answer.get("summary") or "").strip() or None,
        # What the model was shown, not what it says it read.
        sources=[d.url for d in documents],
        confidence=float(answer.get("confidence") or 0.0),
        characters=[c for c, _ in characters],
    )


def _character(entry: dict) -> tuple[Character, bool]:
    aliases = [a.strip() for a in entry.get("aliases") or [] if a and a.strip()]
    importance = entry.get("importance") if entry.get("importance") in ("main", "supporting") else "supporting"
    character = Character(
        name=entry["name"].strip(),
        aliases=aliases,
        actor=_clean(entry.get("actor")),
        importance=importance,
        role=_clean(entry.get("role")),
        personality=_clean(entry.get("personality")),
        relationships=_clean(entry.get("relationships")),
        appearance=_clean(entry.get("appearance")),
    )
    return character, bool(entry.get("in_scene"))


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None
