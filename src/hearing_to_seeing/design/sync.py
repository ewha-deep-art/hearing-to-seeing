from hearing_to_seeing.schema import WordEntry
from hearing_to_seeing.design.volume import compute_font_size

POP_SCALE = 150
GROW_RATIO = 0.4   # 발화 시간 중 몇 %를 "커지는 데" 쓸지 (나머지는 축소)

def seconds_to_centiseconds(seconds: float) -> int:
    return max(1, round(seconds * 100))


def build_fill_text(words: list[WordEntry], color: str) -> str:
    """평상시 상태: 말하기 전엔 흰색으로 보이고, '말하는 동안'엔 숨겨짐
    (그 순간엔 Pop 레이어가 대신 보임), 말이 끝나면 채워진 색으로 고정."""
    if not words:
        return ""

    chunk_start = words[0].start
    parts = []
    for word in words:
        fs = compute_font_size(word.volume)
        t0 = round((word.start - chunk_start) * 1000)
        t1 = round((word.end - chunk_start) * 1000)
        parts.append(
            f"{{\\fs{fs}\\c&H00FFFFFF&\\alpha&H00&"
            f"\\t({t0},{t0 + 1},\\alpha&HFF&)"
            f"\\t({t1},{t1 + 1},\\alpha&H00&\\c{color})}}{word.text} "
        )
    return "".join(parts).rstrip()


def build_pop_event_text(word: WordEntry, color: str, cx: float, cy: float, fs: int) -> str:
    """말하는 동안에만 존재하는 단어. 시작=word.start, 끝=word.end로 정확히 고정.
    그 구간 안에서 GROW_RATIO 비율만큼 커졌다가 나머지 시간 동안 원래 크기로 축소."""
    duration_ms = max(1, round(word.duration * 1000))
    grow_ms = max(1, round(duration_ms * GROW_RATIO))
    cs = seconds_to_centiseconds(word.duration)

    return (
        f"{{\\an5\\pos({cx:.0f},{cy:.0f})\\fs{fs}\\c{color}"
        f"\\t(0,{grow_ms},\\fscx{POP_SCALE}\\fscy{POP_SCALE})"
        f"\\t({grow_ms},{duration_ms},\\fscx100\\fscy100)"
        f"\\kf{cs}}}{word.text}"
    )