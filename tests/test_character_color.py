import pytest

from hearing_to_seeing.design.character_color import (
    ACHROMATIC_CHROMA,
    apply_proposals,
    choose_basis,
    hue_of,
    hue_weight,
    parse_hex,
    suggest_hues,
    with_character_hues,
)
from hearing_to_seeing.design.identity import manual_mapping
from hearing_to_seeing.design.oklch import from_ass, rgb_to_oklch
from hearing_to_seeing.design.speaker import SpeakerCandidate, resolve_speakers
from hearing_to_seeing.schema import Character, Transcript, WordEntry, WorkInfo

RED = "#C0392B"
BLUE = "#1F4E9C"
BLACK = "#000000"
CHARCOAL = "#36454F"


class FakeLLM:
    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.calls: list[str] = []

    def structured(self, system, prompt, schema, *, web_search=False):
        self.calls.append(prompt)
        return self.answers.pop(0)


def _transcript(*labels: str) -> Transcript:
    return Transcript(
        words=[
            WordEntry(text="w", start=float(i), end=float(i) + 0.5, speaker=label)
            for i, label in enumerate(labels)
        ]
    )


def _proposal(name, costume=None, symbolic=None, confidence=0.9, **evidence) -> dict:
    return {
        "name": name,
        "costume": {"hex": costume, "evidence": evidence.get("costume_evidence")},
        "symbolic": {"hex": symbolic, "evidence": evidence.get("symbolic_evidence")},
        "confidence": confidence,
    }


# --- colour arithmetic -------------------------------------------------------

@pytest.mark.parametrize("value, rgb", [
    ("#C0392B", (0xC0, 0x39, 0x2B)),
    ("c0392b", (0xC0, 0x39, 0x2B)),
    (" #C0392B ", (0xC0, 0x39, 0x2B)),
])
def test_parse_hex_accepts_common_spellings(value, rgb):
    assert parse_hex(value) == rgb


@pytest.mark.parametrize("value", [None, "", "red", "#FFF", "#GGGGGG", "#C0392B00"])
def test_parse_hex_rejects_anything_else(value):
    assert parse_hex(value) is None


def test_hue_of_a_coloured_garment_matches_oklch():
    expected = rgb_to_oklch(0xC0, 0x39, 0x2B)[2]
    assert hue_of(RED) == pytest.approx(expected)


@pytest.mark.parametrize("value", [BLACK, "#FFFFFF", "#808080", CHARCOAL])
def test_hue_of_a_grey_is_none(value):
    # Grey has a hue on paper and none on screen; it is no grounds for a colour.
    assert rgb_to_oklch(*parse_hex(value))[1] < ACHROMATIC_CHROMA
    assert hue_of(value) is None


def test_choose_basis_prefers_costume_over_symbolic():
    hue, basis = choose_basis(costume=RED, symbolic=BLUE)
    assert basis == "costume"
    assert hue == pytest.approx(hue_of(RED))


def test_choose_basis_falls_through_to_symbolic_when_costume_is_grey():
    hue, basis = choose_basis(costume=BLACK, symbolic=BLUE)
    assert basis == "symbolic"
    assert hue == pytest.approx(hue_of(BLUE))


def test_choose_basis_falls_through_to_symbolic_when_costume_is_unknown():
    assert choose_basis(costume=None, symbolic=BLUE)[1] == "symbolic"


def test_choose_basis_explains_why_there_is_no_hue():
    # A grey costume that nothing symbolic rescued is recorded as such, so the
    # JSON says why this character carries no preference.
    assert choose_basis(costume=BLACK, symbolic=CHARCOAL) == (None, "achromatic")
    assert choose_basis(costume=None, symbolic=None) == (None, "none")


def test_hue_weight_ranks_costume_over_symbolic_and_leads_over_bit_parts():
    lead_costume = Character(name="a", importance="main", hue=1.0, hue_basis="costume", hue_confidence=0.9)
    lead_symbolic = Character(name="b", importance="main", hue=1.0, hue_basis="symbolic", hue_confidence=0.9)
    minor_costume = Character(name="c", importance="supporting", hue=1.0, hue_basis="costume", hue_confidence=0.9)
    assert hue_weight(lead_costume) > hue_weight(lead_symbolic)
    assert hue_weight(lead_costume) > hue_weight(minor_costume)
    assert hue_weight(Character(name="d")) == 0.0


# --- apply_proposals / suggest_hues -----------------------------------------

def test_apply_proposals_writes_hue_basis_and_evidence_onto_characters():
    work = WorkInfo(characters=[Character(name="기택"), Character(name="박 사장")])
    apply_proposals(work, [
        _proposal("기택", costume=BLACK, symbolic="#A0522D", confidence=0.8,
                  costume_evidence="검은 셔츠", symbolic_evidence="흙빛, 반지하"),
        # Name spelt without the space the sheet has: still the same person.
        _proposal("박사장", costume=BLUE, confidence=0.95, costume_evidence="푸른 정장"),
    ])

    기택, 박사장 = work.characters
    assert 기택.hue_basis == "symbolic"
    assert 기택.hue_evidence == "흙빛, 반지하"     # the evidence for the basis chosen
    assert 기택.hue_confidence == 0.8
    assert 박사장.hue_basis == "costume"
    assert 박사장.hue == pytest.approx(hue_of(BLUE), abs=0.1)
    assert 박사장.hue_evidence == "푸른 정장"


def test_apply_proposals_marks_an_unanswered_character_as_decided_with_no_hue():
    work = WorkInfo(characters=[Character(name="문광")])
    apply_proposals(work, [])
    assert work.characters[0].hue is None
    assert work.characters[0].hue_basis == "none"
    assert work.characters[0].hue_confidence == 0.0


def test_suggest_hues_asks_once_for_the_whole_cast():
    llm = FakeLLM({"characters": [_proposal("기택", costume=RED), _proposal("충숙", symbolic=BLUE)]})
    work = WorkInfo(
        title="기생충",
        characters=[
            Character(name="기택", role="가장", appearance="붉은 셔츠"),
            Character(name="충숙", personality="강단 있음"),
        ],
    )
    suggest_hues(work, llm)

    assert len(llm.calls) == 1
    assert "붉은 셔츠" in llm.calls[0] and "강단 있음" in llm.calls[0]
    assert work.characters[0].hue_basis == "costume"
    assert work.characters[1].hue_basis == "symbolic"


def test_suggest_hues_skips_a_cast_that_is_already_decided():
    llm = FakeLLM()
    work = WorkInfo(characters=[Character(name="기택", hue=None, hue_basis="achromatic")])
    suggest_hues(work, llm)
    assert llm.calls == []


def test_suggest_hues_with_refresh_asks_again():
    llm = FakeLLM({"characters": [_proposal("기택", costume=RED)]})
    work = WorkInfo(characters=[Character(name="기택", hue=200.0, hue_basis="symbolic")])
    suggest_hues(work, llm, refresh=True)
    assert work.characters[0].hue_basis == "costume"


def test_suggest_hues_with_no_characters_asks_nothing():
    llm = FakeLLM()
    suggest_hues(WorkInfo(title="기생충"), llm)
    assert llm.calls == []


# --- with_character_hues -----------------------------------------------------

def _work() -> WorkInfo:
    return WorkInfo(
        title="기생충",
        characters=[
            Character(name="기택", aliases=["아버지"], importance="main",
                      hue=40.0, hue_basis="costume", hue_evidence="갈색 점퍼", hue_confidence=0.9),
            Character(name="문광", importance="supporting", hue=None, hue_basis="achromatic"),
        ],
    )


def test_with_character_hues_gives_a_manual_mapping_its_characters_colour():
    strategy = with_character_hues(manual_mapping({"SPEAKER_00": "기택"}), _work())
    candidate = strategy(_transcript("SPEAKER_00"), "clip.mp4")["SPEAKER_00"]

    assert candidate.name == "기택"
    assert candidate.confidence == 1.0           # identity is still the mapping's
    assert candidate.source == "manual"
    assert candidate.preferred_hue == 40.0
    assert candidate.hue_weight == pytest.approx(hue_weight(_work().characters[0]))
    assert "갈색 점퍼" in candidate.note


def test_with_character_hues_matches_an_alias():
    strategy = with_character_hues(manual_mapping({"SPEAKER_00": "아버지"}), _work())
    candidate = strategy(_transcript("SPEAKER_00"), "clip.mp4")["SPEAKER_00"]
    assert candidate.preferred_hue == 40.0


def test_with_character_hues_leaves_a_hueless_or_unknown_character_alone():
    strategy = with_character_hues(
        manual_mapping({"SPEAKER_00": "문광", "SPEAKER_01": "없는사람"}), _work(),
    )
    candidates = strategy(_transcript("SPEAKER_00", "SPEAKER_01"), "clip.mp4")
    assert candidates["SPEAKER_00"].preferred_hue is None
    assert candidates["SPEAKER_01"].preferred_hue is None
    assert candidates["SPEAKER_00"].note == "사용자 지정"


def test_with_character_hues_does_not_override_a_strategy_with_its_own_hue():
    def strategy(transcript, media_path):
        return {"SPEAKER_00": SpeakerCandidate(
            label="SPEAKER_00", name="기택", confidence=0.9, preferred_hue=200.0,
        )}

    wrapped = with_character_hues(strategy, _work())
    assert wrapped(_transcript("SPEAKER_00"), "clip.mp4")["SPEAKER_00"].preferred_hue == 200.0


def test_with_character_hues_passes_through_when_there_is_nothing_to_add():
    strategy = manual_mapping({"SPEAKER_00": "기택"})
    assert with_character_hues(None, _work()) is None
    assert with_character_hues(strategy, None) is strategy
    assert with_character_hues(strategy, WorkInfo(title="x")) is strategy


def test_end_to_end_the_subtitle_colour_lands_on_the_characters_hue():
    strategy = with_character_hues(manual_mapping({"SPEAKER_00": "기택"}), _work())
    transcript = _transcript("SPEAKER_00", "SPEAKER_01")
    profiles = resolve_speakers(transcript, strategy(transcript, "clip.mp4"))

    hue = rgb_to_oklch(*from_ass(profiles["SPEAKER_00"].color))[2]
    assert hue == pytest.approx(40.0, abs=1.0)
    assert profiles["SPEAKER_00"].name == "기택"
    assert "갈색 점퍼" in profiles["SPEAKER_00"].note
