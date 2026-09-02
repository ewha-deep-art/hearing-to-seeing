# TODO: 색상 배정 방식 최종 미확정 — 단순 팔레트 순서 배정(현재) vs RAG 기반
#       (인물 정보 → LLM 색상 결정). 기획서 §다음 논의 필요 사항 참조.
# Wong colorblind-friendly palette in ASS BGR hex format (&HAABBGGRR&).
# ASS stores colours byte-reversed relative to RGB, so #E69F00 → &H00009FE6.
_PALETTE = [
    "&H00009FE6",  # Orange        #E69F00
    "&H00E9B456",  # Sky Blue      #56B4E9
    "&H00739E00",  # Bluish Green  #009E73
    "&H0042E4F0",  # Yellow        #F0E442
    "&H00B27200",  # Blue          #0072B2
    "&H00005ED5",  # Vermillion    #D55E00
    "&H00A779CC",  # Reddish Purple #CC79A7
]

# Colour every word starts out in, before the karaoke fill reaches it.
BASE_COLOUR = "&H00FFFFFF"  # white


def assign_speaker_colors(speakers: list[str]) -> dict[str, str]:
    # TODO: 화자가 알파벳순(SPEAKER_00, SPEAKER_01, …)으로 정렬되어 색상이
    #       WhisperX 라벨 순서에 의존함. 대화 첫 등장 순서 기반 배정 방식 검토 필요.
    return {
        speaker: _PALETTE[i % len(_PALETTE)]
        for i, speaker in enumerate(sorted(set(speakers)))
    }
