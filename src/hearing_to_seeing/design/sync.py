from hearing_to_seeing.schema import WordEntry

# The last word of a line has no successor to bound it, so its own duration
# decides how long the fill runs. Forced aligners routinely stretch the final
# character of an utterance across the silence that follows it, which would
# otherwise leave one word creeping across the screen for tens of seconds.
MAX_TRAILING_HOLD = 2.0

# --- "위로 뜨는" 효과 ---------------------------------------------------------
# 기획서 §4는 발화 시점에 해당 단어가 살짝 위로 뜨는 효과를 요구한다. ASS에서
# 위치 이동 태그(\pos, \move)는 Dialogue 이벤트 전체에만 걸리므로 한 줄 안의 한
# 단어만 실제로 옮기려면 단어마다 이벤트를 쪼개고 글꼴 메트릭으로 x 좌표를 직접
# 계산해야 한다. 대신 한 이벤트 안에서 처리 가능한 두 신호로 부양감을 만든다:
#   * \fscy — 글자가 베이스라인 위로 자란다. 세로만 늘리므로 뒤따르는 단어가
#             가로로 밀리지 않는다 (\fs나 \fscx를 쓰면 줄 전체가 흔들린다).
#   * \shad — 그림자가 글자에서 멀어지며 떠오른 높이를 암시한다.
# TODO: 기획서가 말하는 "위치 이동"은 아직 근사치 — 실제 좌표 이동은 단어별
#       Dialogue 이벤트 분할과 글꼴 메트릭 기반 x 좌표 계산이 필요함. 현재 연출로
#       충분한지 사용자 검증 후 결정.
LIFT_SCALE = 110        # \fscy at the peak, percent
LIFT_SHADOW = 4         # \shad at the peak, pixels
BASE_SHADOW = 1         # \shad in the Default style — where the fall lands
LIFT_RISE = 0.12        # seconds to reach the peak
LIFT_FALL = 0.20        # seconds to settle back


def seconds_to_centiseconds(seconds: float) -> int:
    return max(1, round(seconds * 100))


def karaoke_durations(words: list[WordEntry]) -> list[int]:
    """Per-word karaoke durations (centiseconds) that span the whole line.

    A word's \\k value must cover the silence that follows it, not just its own
    utterance — ASS advances the fill by the sum of the \\k values, so dropping
    the inter-word gaps makes the animation race ahead of the audio. Each word
    therefore holds until the next word begins; the last word uses its own
    duration.
    """
    durations = []
    for i, word in enumerate(words):
        if i + 1 < len(words):
            span = words[i + 1].start - word.start
        else:
            span = min(word.duration, MAX_TRAILING_HOLD)
        durations.append(seconds_to_centiseconds(span))
    return durations


def lift_tags(durations: list[int]) -> list[str]:
    """Per-word \\t transforms that lift each word as the karaoke fill reaches it.

    `durations` is what `karaoke_durations()` returned for the same line, so the
    lift runs off the karaoke clock itself and cannot drift away from the \\kf
    fill that triggers it: \\t offsets are milliseconds from the Dialogue event
    start, which is exactly where that clock starts.

    A word too short for the full rise-and-fall gets a proportionally quicker
    animation, so it is back down before the next word takes off instead of
    leaving a run of words hanging at once.
    """
    tags: list[str] = []
    offset_ms = 0
    for cs in durations:
        span_ms = cs * 10
        rise = min(round(LIFT_RISE * 1000), max(1, span_ms // 3))
        fall = min(round(LIFT_FALL * 1000), span_ms - rise)
        peak = offset_ms + rise
        tags.append(
            f"\\t({offset_ms},{peak},\\fscy{LIFT_SCALE}\\shad{LIFT_SHADOW})"
            f"\\t({peak},{peak + fall},\\fscy100\\shad{BASE_SHADOW})"
        )
        offset_ms += span_ms
    return tags
