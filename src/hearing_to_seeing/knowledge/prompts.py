"""The questions the pipeline asks the model.

Each is a (SYSTEM, PROMPT, SCHEMA) triple for `llm.ask()`; PROMPT is filled in
with `str.format`. The answers are plain dicts shaped by SCHEMA and are read
directly by `pipeline.py` (1–3) and `review.py` (4).
"""

# --- 1. clip title → work -----------------------------------------------------
# Clip titles are written for the feed ("[4K] 기생충 명장면 모음 #shorts"), so
# the work's own title has to be pulled out before anything can be searched.

TITLE_SYSTEM = """\
당신은 한국어 유튜브 영상 제목을 분석하는 도구다. 클립 제목에서 원작 작품(영화·드라마·
애니메이션)의 정확한 제목을 뽑아낸다. 작품을 특정할 수 없으면 title 을 null 로 둔다.
추측으로 없는 작품명을 만들어내지 않는다."""

TITLE_PROMPT = "유튜브 클립 제목: {title}"

TITLE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": ["string", "null"],
            "description": "원작 작품의 정식 제목 (한국어 개봉/방영 제목). 모르면 null",
        },
    },
    "required": ["title"],
    "additionalProperties": False,
}

# --- 2. documents → cast, with a subtitle colour per character ----------------
# One request for the whole cast, so the model can keep the colours apart the
# way a designer picks a palette rather than one colour at a time. The model
# never sees OKLCH: it proposes sRGB and pipeline.py keeps only the hue.

CAST_SYSTEM = """\
당신은 영화·드라마 등장인물 조사 도구이자 자막 색채 전문가다. 배우가 아니라 **극중 인물**을
정리하고, 인물마다 자막 색을 제안한다.
- 주어진 문서에 있는 내용만 쓰고, 문서에 없는 항목은 null 로 둔다. 인물을 지어내지 않는다.
  문서가 없을 때만 알고 있는 지식으로 답하되, color_confidence 를 그만큼 낮게 적는다.
- 색은 sRGB 6자리 hex 로 답한다. 밝기는 신경 쓰지 않는다 — 자막 시스템이 밝기를 고정하고
  색상(hue)만 쓴다.
- 인물이 극중에서 실제로 입는 대표 의상의 색이 문서에 있고 무채색(검정·회색·흰색·짙은 남색)이
  아니면 그 색을 쓴다 (color_basis=costume). 아니면 성격·역할·관계가 연상시키는 상징색을
  쓴다 (color_basis=symbolic). 같은 작품의 인물끼리는 서로 다른 색상 계열을 고른다."""

CAST_PROMPT = """\
작품: {title}

{documents}

위 문서를 바탕으로 이 작품의 등장인물을 주연부터 정리하라. 인물마다:
1. 극중 이름 (자막에 표시될 이름)
2. 대사에서 불리는 다른 호칭 (예: 아버지, 사장님, 전하) — aliases
3. 주연/조연 (importance), 역할 한 줄, 성격, 다른 인물과의 관계
4. 자막 색 (color), 그 근거 (color_basis, color_reason), 이 색이 인물에게 맞을 확률 (color_confidence)"""

NO_DOCUMENTS = "참고 문서: 없음 (검색 실패). 알고 있는 범위에서 답하고 color_confidence 를 낮게 적는다."

CAST_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "극중 이름"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "string", "enum": ["main", "supporting"]},
                    "role": {"type": ["string", "null"]},
                    "personality": {"type": ["string", "null"]},
                    "relationships": {"type": ["string", "null"]},
                    "color": {"type": "string", "description": "sRGB 6자리 hex, 예: #C0392B"},
                    "color_basis": {"type": "string", "enum": ["costume", "symbolic"]},
                    "color_reason": {"type": "string", "description": "한 줄 근거"},
                    "color_confidence": {"type": "number", "description": "0~1"},
                },
                "required": [
                    "name", "aliases", "importance", "role", "personality",
                    "relationships", "color", "color_basis", "color_reason", "color_confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["characters"],
    "additionalProperties": False,
}

# --- 3. dialogue → which speaker is which character ---------------------------

SPEAKERS_SYSTEM = """\
당신은 영화·드라마 클립의 음성 인식 결과에서 각 화자(SPEAKER_xx)가 어느 등장인물인지 추론하는
도구다. 근거는 대사 내용, 말투(존댓말·반말의 방향), 호칭(가족·직함), 인물 간 관계, 그리고
규칙으로 미리 찾아 둔 '호명 앵커'다.

원칙:
- 호명 앵커는 강한 제약이다. 어떤 화자가 어떤 인물의 이름·호칭을 부르면, 그 화자는 그 인물이
  아닐 가능성이 매우 높고, 바로 다음에 대답한 화자가 그 인물일 가능성이 높다.
- 확신이 없으면 character 를 null 로 두고 confidence 를 낮게 적는다. 틀린 이름은 이름이 없는 것보다
  나쁘다 — 자막을 보는 사람은 소리를 듣지 못해 오류를 알아챌 수 없다.
- 음성 인식 오류가 있을 수 있다. 발음이 비슷한 이름은 같은 것으로 본다.
- character 에는 등장인물 목록에 있는 극중 이름을 그대로 쓴다."""

SPEAKERS_PROMPT = """\
등장인물:
{characters}

호명 앵커 (규칙으로 탐지):
{anchors}

대사 (시간 순서):
{dialogue}

화자마다 어느 인물인지 답하라. 대상 화자: {labels}"""

CHARACTER_LINE = """\
- {name}{aliases} ({importance})
  역할: {role} / 성격: {personality} / 관계: {relationships}"""

SPEAKERS_SCHEMA = {
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

# --- 4. the clip itself → who speaks each utterance, which disputed reading ----
# Asked with the video (or the audio) attached. The model only chooses: a name
# per utterance and an option per disputed spot, never new text. Measured on
# the test clips (docs/STT_TUNING.md, 4차): flash-lite answers every spot with
# a junk option, so this question has no fallback model; and in English the
# model leaves more spots alone than with the same instructions in Korean,
# which overruled the vote more often and more often wrongly.

REVIEW_SYSTEM = (
    "You check Korean subtitles of a Korean-dubbed video clip against the clip itself. "
    "The subtitles come from speech recognition. Speaker labels (SPEAKER_xx) come from "
    "automatic diarization and are often wrong."
)

REVIEW_PROMPT = """\
Task 1, speakers: for EVERY utterance id, name the character actually speaking it. {how} \
Use the cast names below; for an unnamed character use 손님1, 손님2, … and keep the same \
number for the same character throughout. Several utterances in a row may belong to \
different characters even when they share a SPEAKER label.
{text_task}
Cast: {cast}

Utterances:
{utterances}"""

REVIEW_HOW = {
    "video": "Judge by whose mouth moves on screen and by the voice, not by the story.",
    "audio": "Judge by the voice and the dialogue.",
}

REVIEW_TEXT_TASK = """
Task 2, disputed words: ⟨dN @t: 1) … | 2) …⟩ marks a spot (t seconds into the clip) where \
several recognition passes disagree. Option 1 is the current reading, ×k is how many passes \
read it, (없음) means no word there. Report a spot only when you clearly hear an option \
other than 1; skip spots where option 1 is right or you are unsure. Never answer with text \
that is not one of the options.
"""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "utterances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "u1, u2, …"},
                    "speaker": {"type": "string", "description": "the speaking character"},
                },
                "required": ["id", "speaker"],
                "additionalProperties": False,
            },
        },
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "d0, d1, …"},
                    "option": {"type": "integer", "description": "the chosen option (2 or more)"},
                },
                "required": ["id", "option"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["utterances", "corrections"],
    "additionalProperties": False,
}
