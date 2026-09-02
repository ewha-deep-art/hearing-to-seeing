from hearing_to_seeing.schema import Transcript

# TODO: 색상 배정 방식 최종 미확정 — 단순 팔레트 순서 배정(현재) vs RAG 기반
#       (인물 정보 → LLM 색상 결정). 기획서 §다음 논의 필요 사항 참조.
# TODO: ASS BGR hex 값이 Wong 팔레트와 일치하는지 검수 필요. ASS 형식은 &HAABBGGRR이므로
#       Orange(RGB #E69F00)는 &H00009FE6이어야 하며, &H00E69F00은 잘못된 값일 수 있음.
# Wong colorblind-friendly palette in ASS BGR hex format (&HAABBGGRR&)
_PALETTE = [
    "&H00E69F00",  # Orange
    "&H00E9B456",  # Sky Blue
    "&H0073E600",  # Bluish Green
    "&H0042E4F0",  # Yellow
    "&H00B20072",  # Blue
    "&H000055D5",  # Vermillion
    "&H00A779CC",  # Reddish Purple
]


def assign_speaker_colors(speakers: list[str]) -> dict[str, str]:
    # TODO: 화자가 알파벳순(SPEAKER_00, SPEAKER_01, …)으로 정렬되어 색상이
    #       WhisperX 라벨 순서에 의존함. 대화 첫 등장 순서 기반 배정 방식 검토 필요.
    return {
        speaker: _PALETTE[i % len(_PALETTE)]
        for i, speaker in enumerate(sorted(set(speakers)))
    }


def annotate_speakers(transcript: Transcript) -> dict[str, str]:
    color_map = assign_speaker_colors(transcript.speakers())
    return color_map
