from hearing_to_seeing.schema import WordEntry
from hearing_to_seeing.design.volume import BASE_FONT_SIZE

LOUD_SCALE = 150
LOUD_GROW_RATIO = 0.4

WHISPER_SCALE = 65
WHISPER_SHRINK_RATIO = 0.4

WAVE_RISE_PX = 18
WAVE_RISE_MS = 70
WAVE_FALL_MS = 90


def build_fill_text(words: list[WordEntry], color: str) -> str:
    """평상시: 말하기 전엔 흰색, 말하는 동안엔 숨김(다른 애니메이션이 대신 보임),
    말한 후엔 채워진 색. 모든 단어가 같은 고정 크기(BASE_FONT_SIZE)."""
    if not words:
        return ""

    chunk_start = words[0].start
    parts = []
    for word in words:
        t0 = round((word.start - chunk_start) * 1000)
        t1 = round((word.end - chunk_start) * 1000)
        parts.append(
            f"{{\\fs{BASE_FONT_SIZE}\\c&H00FFFFFF&\\alpha&H00&"
            f"\\t({t0},{t0 + 1},\\alpha&HFF&)"
            f"\\t({t1},{t1 + 1},\\alpha&H00&\\c{color})}}{word.text} "
        )
    return "".join(parts).rstrip()


def build_loud_event_text(word: WordEntry, color: str, cx: float, cy: float) -> str:
    """소리지르는 단어: 발화 구간 동안 확 커졌다가 원래 크기로 돌아옴."""
    duration_ms = max(1, round(word.duration * 1000))
    grow_ms = max(1, round(duration_ms * LOUD_GROW_RATIO))
    return (
        f"{{\\an5\\pos({cx:.0f},{cy:.0f})\\fs{BASE_FONT_SIZE}\\c{color}"
        f"\\t(0,{grow_ms},\\fscx{LOUD_SCALE}\\fscy{LOUD_SCALE})"
        f"\\t({grow_ms},{duration_ms},\\fscx100\\fscy100)}}{word.text}"
    )


def build_whisper_event_text(word: WordEntry, color: str, cx: float, cy: float) -> str:
    """속삭이는 단어: 발화 구간 동안 작아졌다가 원래 크기로 돌아옴."""
    duration_ms = max(1, round(word.duration * 1000))
    shrink_ms = max(1, round(duration_ms * WHISPER_SHRINK_RATIO))
    return (
        f"{{\\an5\\pos({cx:.0f},{cy:.0f})\\fs{BASE_FONT_SIZE}\\c{color}"
        f"\\t(0,{shrink_ms},\\fscx{WHISPER_SCALE}\\fscy{WHISPER_SCALE})"
        f"\\t({shrink_ms},{duration_ms},\\fscx100\\fscy100)}}{word.text}"
    )


def build_wave_events(
    word: WordEntry, color: str, char_positions: list[tuple[str, float, float]]
) -> list[tuple[float, float, str]]:
    """보통 발화 단어: 글자를 순서대로 하나씩 살짝 띄웠다 내려놓아 파도타듯 보이게 함.
    ASS는 한 이벤트당 위치 이동(\\move)을 한 번만 지원해서, 글자당
    '띄우기'와 '내려놓기' 두 이벤트로 나눠서 이어붙임."""
    n = len(char_positions)
    if n == 0:
        return []

    slot = word.duration / n
    events = []
    for i, (ch, cx, cy) in enumerate(char_positions):
        rise_start = word.start + i * slot
        rise_dur = max(1, min(WAVE_RISE_MS, round(slot * 1000)))
        fall_dur = max(1, min(WAVE_FALL_MS, round(slot * 1000)))
        rise_end = rise_start + rise_dur / 1000

        rise_text = (
            f"{{\\an5\\move({cx:.0f},{cy:.0f},{cx:.0f},{cy - WAVE_RISE_PX:.0f},0,{rise_dur})"
            f"\\fs{BASE_FONT_SIZE}\\c{color}}}{ch}"
        )
        events.append((rise_start, rise_end, rise_text))

        fall_text = (
            f"{{\\an5\\move({cx:.0f},{cy - WAVE_RISE_PX:.0f},{cx:.0f},{cy:.0f},0,{fall_dur})"
            f"\\fs{BASE_FONT_SIZE}\\c{color}}}{ch}"
        )
        events.append((rise_end, word.end, fall_text))

    return events