"""Turning a YouTube clip title into something that can be searched for.

Clip titles are written for the recommendation feed, not for lookup:

    "[4K] 기생충 명장면 모음 | 아들아 너는 계획이 다 있구나 #shorts"

Inside that is a work — 기생충 — and a scene hint — the line being quoted —
and the two need separating, because the work is what the character search
runs on and the scene is what narrows twenty characters down to three.

Splitting them is one model call. The rule-based `clean_title()` runs first
and exists on its own for the no-key case: it strips the obvious furniture
(tags in brackets, hashtags, resolution badges) and hands back what is left as
the best available guess, without pretending to know where the work ends and
the hint begins.
"""

import re
from dataclasses import dataclass

from hearing_to_seeing.knowledge.llm import LLM
from hearing_to_seeing.schema import MediaInfo

# Things that decorate a clip title without being part of any title.
_BRACKETED = re.compile(r"[\[\(【《〈「『][^\]\)】》〉」』]*[\]\)】》〉」』]")
_BRACKET_CHARS = re.compile(r"[\[\]\(\)【】《》〈〉「」『』]")
_HASHTAG = re.compile(r"#\S+")
_BADGE = re.compile(
    r"\b(4k|8k|1080p|720p|hd|uhd|fhd|full\s*hd|60fps|hdr|shorts?|official|"
    r"trailer|teaser|mv|ost)\b",
    re.IGNORECASE,
)
_KO_BADGE = re.compile(r"(공식\s*예고편|예고편|티저|명장면\s*모음|명장면|하이라이트|결말\s*포함|리뷰|요약|몰아보기)")
_SEPARATORS = re.compile(r"\s*[|｜/／‖•·]+\s*")
_SPACES = re.compile(r"\s+")

# Below this the parsed title is not believed and the raw one is searched.
MIN_CONFIDENCE = 0.5

SYSTEM = """\
당신은 한국어 유튜브 영상 제목을 분석하는 도구다. 클립 제목에서 원작 작품(영화·드라마·
애니메이션)의 정확한 제목과, 제목이 가리키는 특정 장면의 힌트를 분리한다.
결과는 요청된 JSON 스키마로만 답한다. 작품을 특정할 수 없으면 title 을 null 로 둔다.
추측으로 없는 작품명을 만들어내지 않는다."""

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": ["string", "null"],
            "description": "원작 작품의 정식 제목 (한국어 개봉/방영 제목). 모르면 null",
        },
        "year": {
            "type": ["integer", "null"],
            "description": "개봉/방영 연도. 제목만으로 알 수 없으면 null",
        },
        "scene_hint": {
            "type": ["string", "null"],
            "description": "제목이 가리키는 장면·대사·상황 힌트. 없으면 null",
        },
        "confidence": {
            "type": "number",
            "description": "title 이 맞을 확률 0~1",
        },
    },
    "required": ["title", "year", "scene_hint", "confidence"],
    "additionalProperties": False,
}


@dataclass
class WorkQuery:
    """What to look the clip up as: a work, and where in it the clip sits."""

    title: str
    year: int | None = None
    scene_hint: str | None = None
    confidence: float = 0.0
    source: str = "raw"       # raw | rules | llm


def clean_title(raw: str) -> str:
    """Strips clip-title furniture, leaving the words that could name a work.

    Brackets usually hold a tag — "[4K]", "(ENG SUB)" — and go first. But a
    Korean channel just as often brackets the work itself, "[왕과 사는 남자]
    공식 예고편", so when nothing survives without them the bracketed words
    are kept and only the brackets go.
    """
    text = _strip_furniture(_BRACKETED.sub(" ", raw))
    if not text:
        text = _strip_furniture(_BRACKET_CHARS.sub(" ", raw))
    return text or raw.strip()


def _strip_furniture(text: str) -> str:
    text = _HASHTAG.sub(" ", text)
    text = _BADGE.sub(" ", text)
    text = _KO_BADGE.sub(" ", text)
    text = _SEPARATORS.sub(" ", text)
    return _SPACES.sub(" ", text).strip(" -–—:,.|")


def parse_title(media: MediaInfo, llm: LLM | None = None) -> WorkQuery | None:
    """Decides what to search for, from the title (and URL) the user gave.

    Returns None only when there is no title at all — a URL by itself is not
    fetched, so it cannot name the work. Without a model the cleaned title
    stands in for the work; with one, the work and the scene hint are split
    apart and the split is trusted only above `MIN_CONFIDENCE`.
    """
    raw = (media.title or "").strip()
    if not raw:
        return None

    fallback = WorkQuery(title=clean_title(raw), source="rules")
    if llm is None:
        return fallback

    prompt = f"유튜브 클립 제목: {raw}"
    if media.source_url:
        prompt += f"\n영상 URL: {media.source_url}"
    answer = llm.structured(SYSTEM, prompt, SCHEMA)

    title = (answer.get("title") or "").strip()
    confidence = float(answer.get("confidence") or 0.0)
    if not title or confidence < MIN_CONFIDENCE:
        return fallback

    hint = (answer.get("scene_hint") or "").strip() or None
    return WorkQuery(
        title=title,
        year=answer.get("year"),
        scene_hint=hint,
        confidence=confidence,
        source="llm",
    )
