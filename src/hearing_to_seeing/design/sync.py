from hearing_to_seeing.schema import WordEntry


def seconds_to_centiseconds(seconds: float) -> int:
    return max(1, round(seconds * 100))


def karaoke_duration(word: WordEntry) -> int:
    return seconds_to_centiseconds(word.duration)


def build_karaoke_text(words: list[WordEntry]) -> str:
    """Returns ASS karaoke-tagged text for a sequence of words with the same speaker."""
    # TODO: \k는 좌→우 색상 와이프 효과를 적용함. 기획서는 "색이 채워지는" fill 효과를
    #       요구하므로 \kf(카라오케 fill)로 전환 검토 필요.
    # TODO: 기획서에서 요구하는 "글자가 살짝 위로 뜨는" 효과 미구현 —
    #       ASS 출력에서 단어별 \move 또는 \pos + \an 애니메이션으로 구현 필요.
    parts = []
    for word in words:
        cs = karaoke_duration(word)
        parts.append(f"{{\\k{cs}}}{word.text}")
    return " ".join(parts)
