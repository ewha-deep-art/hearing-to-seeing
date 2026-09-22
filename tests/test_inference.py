import pytest

from hearing_to_seeing.design.inference import (
    MIN_WORDS,
    REPLY_WINDOW,
    detect_anchors,
    dialogue_text,
    infer_from_text,
)
from hearing_to_seeing.design.speaker import resolve_speakers
from hearing_to_seeing.schema import Character, Transcript, WordEntry, WorkInfo


class FakeLLM:
    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.prompts: list[str] = []

    def structured(self, system, prompt, schema, *, web_search=False):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def _words(*turns: tuple[str, str], gap: float = 0.5) -> Transcript:
    """`("SPEAKER_00", "아버지 밥 먹어요")` per turn, laid out end to end."""
    words: list[WordEntry] = []
    clock = 0.0
    for speaker, text in turns:
        for token in text.split():
            words.append(WordEntry(text=token, start=clock, end=clock + 0.4, speaker=speaker))
            clock += 0.5
        clock += gap
    return Transcript(words=words)


def _work() -> WorkInfo:
    return WorkInfo(
        title="기생충",
        scene="반지하 집, 저녁",
        characters=[
            Character(name="기택", aliases=["아버지", "여보"], importance="main", role="가장"),
            Character(name="충숙", aliases=["엄마", "여보"], importance="main"),
            Character(name="기우", aliases=["아들", "오빠"], importance="main"),
            Character(name="박 사장", aliases=["사장님"], importance="supporting"),
        ],
    )


def _answer(*rows: tuple[str, str | None, float, str]) -> dict:
    return {"speakers": [
        {"label": label, "character": character, "confidence": confidence, "reasoning": why}
        for label, character, confidence, why in rows
    ]}


# --- anchors -----------------------------------------------------------------

def test_detect_anchors_counts_who_named_whom_and_who_answered():
    transcript = _words(
        ("SPEAKER_00", "아버지 이거 좀 보세요"),
        ("SPEAKER_01", "뭔데 그래"),
    )
    anchors = detect_anchors(transcript, _work())

    assert anchors.said[("SPEAKER_00", "기택")] == 1
    assert anchors.answered[("SPEAKER_01", "기택")] == 1
    assert anchors.said[("SPEAKER_01", "기택")] == 0


def test_detect_anchors_matches_a_name_with_a_particle_attached():
    transcript = _words(("SPEAKER_00", "사장님은 언제 오세요"), ("SPEAKER_01", "곧"))
    anchors = detect_anchors(transcript, _work())
    assert anchors.said[("SPEAKER_00", "박 사장")] == 1


def test_detect_anchors_does_not_credit_a_late_reply():
    transcript = _words(
        ("SPEAKER_00", "아버지"),
        ("SPEAKER_01", "네"),
        gap=REPLY_WINDOW + 1.0,
    )
    anchors = detect_anchors(transcript, _work())
    assert anchors.said[("SPEAKER_00", "기택")] == 1
    assert anchors.answered == {}


def test_detect_anchors_does_not_credit_the_speaker_answering_themselves():
    transcript = _words(("SPEAKER_00", "아버지 아버지 어디 계세요"))
    anchors = detect_anchors(transcript, _work())
    # One turn, one mention (counted per turn, not per word), no reply.
    assert anchors.said[("SPEAKER_00", "기택")] == 1
    assert anchors.answered == {}


def test_detect_anchors_ignores_single_syllable_names():
    work = WorkInfo(characters=[Character(name="강", aliases=["형"])])
    transcript = _words(("SPEAKER_00", "강 형 왔어"), ("SPEAKER_01", "응"))
    assert detect_anchors(transcript, work).is_empty()


def test_anchors_describe_lists_the_rules_the_model_is_given():
    transcript = _words(("SPEAKER_00", "아버지"), ("SPEAKER_01", "응"))
    text = detect_anchors(transcript, _work()).describe()
    assert "SPEAKER_00 이(가) '기택'" in text
    assert "SPEAKER_01 이(가) 1번 대답함" in text


def test_anchors_describe_when_nothing_was_found():
    assert detect_anchors(_words(("SPEAKER_00", "밥 먹자")), _work()).describe() == "(없음)"


# --- dialogue_text -----------------------------------------------------------

def test_dialogue_text_labels_turns_in_order():
    transcript = _words(("SPEAKER_00", "안녕 하세요"), ("SPEAKER_01", "네"))
    assert dialogue_text(transcript) == "[0000.0] SPEAKER_00: 안녕 하세요\n[0001.5] SPEAKER_01: 네"


def test_dialogue_text_keeps_the_opening_whole_and_samples_the_rest():
    turns = [(f"SPEAKER_0{i % 2}", f"대사 번호 {i:03d} 입니다") for i in range(200)]
    text = dialogue_text(_words(*turns), limit=1500)

    lines = text.split("\n")
    assert len(text) <= 1500 + 50           # a little slack for the marker line
    assert lines[0].endswith("대사 번호 000 입니다")
    assert lines[1].endswith("대사 번호 001 입니다")      # opening turns are contiguous
    assert "…" in lines
    assert lines[-1] > lines[lines.index("…") + 1]       # sampled turns keep time order


# --- infer_from_text ---------------------------------------------------------

def test_infer_from_text_asks_once_with_cast_anchors_and_dialogue():
    llm = FakeLLM(_answer(
        ("SPEAKER_00", "기우", 0.85, "아버지를 부르고 반말을 들음"),
        ("SPEAKER_01", "기택", 0.9, "아버지로 불린 뒤 대답"),
    ))
    transcript = _words(("SPEAKER_00", "아버지 이거 좀 보세요"), ("SPEAKER_01", "뭔데 그래 인마"))
    candidates = infer_from_text(_work(), llm)(transcript, "clip.mp4")

    prompt = llm.prompts[0]
    assert "기생충" in prompt and "반지하 집" in prompt
    assert "박 사장 (호칭: 사장님)" in prompt
    assert "호명 앵커" in prompt and "SPEAKER_01 이(가) 1번 대답함" in prompt
    assert "SPEAKER_00: 아버지 이거 좀 보세요" in prompt

    assert candidates["SPEAKER_01"].name == "기택"
    assert candidates["SPEAKER_01"].confidence == 0.9
    assert candidates["SPEAKER_01"].source == "text"
    assert "아버지로 불린 뒤 대답" in candidates["SPEAKER_01"].note
    # The anchor that agrees with the model is recorded next to its reasoning.
    assert "'기택' 호명 뒤 1회 응답" in candidates["SPEAKER_01"].note
    assert candidates["SPEAKER_00"].name == "기우"
    assert candidates["SPEAKER_00"].preferred_hue is None      # colour is not this step's


def test_infer_from_text_resolves_an_alias_to_the_character_name():
    llm = FakeLLM(_answer(("SPEAKER_00", "사장님", 0.8, "직함")))
    transcript = _words(("SPEAKER_00", "일단 계획대로 갑시다"))
    candidates = infer_from_text(_work(), llm)(transcript, "clip.mp4")
    assert candidates["SPEAKER_00"].name == "박 사장"


def test_infer_from_text_keeps_a_blank_or_unknown_answer_as_unnamed():
    llm = FakeLLM(_answer(
        ("SPEAKER_00", None, 0.2, "근거 부족"),
        ("SPEAKER_01", "없는사람", 0.9, "환각"),
    ))
    transcript = _words(("SPEAKER_00", "그냥 그런 말 있잖아"), ("SPEAKER_01", "어 그래 알았어"))
    candidates = infer_from_text(_work(), llm)(transcript, "clip.mp4")

    assert candidates["SPEAKER_00"].name is None
    assert candidates["SPEAKER_00"].confidence == 0.0
    assert candidates["SPEAKER_00"].note == "근거 부족"
    assert candidates["SPEAKER_01"].name is None       # a name not in the cast is no name


def test_infer_from_text_flags_two_labels_on_one_character():
    llm = FakeLLM(_answer(
        ("SPEAKER_00", "기택", 0.8, "a"),
        ("SPEAKER_01", "기택", 0.75, "b"),
    ))
    transcript = _words(("SPEAKER_00", "밥 먹자 얘들아"), ("SPEAKER_01", "다들 이리 와 봐"))
    candidates = infer_from_text(_work(), llm)(transcript, "clip.mp4")

    assert candidates["SPEAKER_00"].name == candidates["SPEAKER_01"].name == "기택"
    assert "SPEAKER_01 도 기택" in candidates["SPEAKER_00"].note
    assert "화자 병합 검토" in candidates["SPEAKER_01"].note


def test_infer_from_text_ignores_labels_it_did_not_ask_about_and_clamps_confidence():
    llm = FakeLLM(_answer(
        ("SPEAKER_00", "기택", 1.7, "확신"),
        ("SPEAKER_99", "충숙", 0.9, "없는 화자"),
    ))
    transcript = _words(("SPEAKER_00", "밥 먹자 얘들아"))
    candidates = infer_from_text(_work(), llm)(transcript, "clip.mp4")
    assert set(candidates) == {"SPEAKER_00"}
    assert candidates["SPEAKER_00"].confidence == 1.0


def test_infer_from_text_skips_speakers_with_too_few_words():
    llm = FakeLLM(_answer(("SPEAKER_00", "기택", 0.9, "x")))
    transcript = _words(("SPEAKER_00", "밥 먹자 얘들아"), ("SPEAKER_01", "응"))
    infer_from_text(_work(), llm)(transcript, "clip.mp4")
    assert "대상 화자: SPEAKER_00" in llm.prompts[0]
    assert "SPEAKER_01" not in llm.prompts[0].split("대상 화자:")[1]


def test_infer_from_text_asks_nothing_without_a_cast_or_speakers():
    llm = FakeLLM()
    assert infer_from_text(WorkInfo(title="x"), llm)(_words(("SPEAKER_00", "a b c d")), "m") == {}
    assert infer_from_text(_work(), llm)(Transcript(), "m") == {}
    assert infer_from_text(_work(), llm)(_words(("SPEAKER_00", "응")), "m") == {}
    assert llm.prompts == []


def test_end_to_end_a_sure_answer_names_the_speaker_and_an_unsure_one_does_not():
    llm = FakeLLM(_answer(
        ("SPEAKER_00", "기택", 0.9, "아버지로 불림"),
        ("SPEAKER_01", "기우", 0.4, "애매"),
    ))
    transcript = _words(("SPEAKER_00", "밥 먹자 얘들아 어서"), ("SPEAKER_01", "네 아버지 갈게요"))
    profiles = resolve_speakers(transcript, infer_from_text(_work(), llm)(transcript, "m"))

    assert profiles["SPEAKER_00"].name == "기택"
    assert profiles["SPEAKER_00"].source == "text"
    assert profiles["SPEAKER_01"].name is None
    assert "discarded 기우" in profiles["SPEAKER_01"].note
