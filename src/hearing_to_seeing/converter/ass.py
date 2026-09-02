import unicodedata

from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.design.speaker import BASE_COLOUR, assign_speaker_colors
from hearing_to_seeing.design.sync import karaoke_durations, lift_tags
from hearing_to_seeing.design.volume import compute_font_size

# --- Subtitle line segmentation ------------------------------------------------
# A single Dialogue event is one on-screen subtitle. Without these limits the
# whole transcript renders as one event that sits on screen for its full
# duration, which is what "all the text at once" looks like.
ROW_WIDTH = 42          # display cells per on-screen row
MAX_WIDTH = 2 * ROW_WIDTH  # a subtitle may occupy at most two rows
MIN_WIDTH_FOR_BREAK = 20  # don't split after punctuation on a near-empty line
MAX_DURATION = 6.0      # seconds a single subtitle may stay on screen
MAX_GAP = 0.7           # a silence this long ends the current subtitle
HOLD = 0.3              # extra seconds the finished line lingers

_TERMINATORS = (".", "?", "!", "…")

# The coordinate space every position, margin and font size in the script is
# expressed in; the preview renderer sizes its canvas to match (render.py).
# TODO: 파라미터로 받거나 ffprobe로 입력 영상 해상도를 자동 감지하도록 개선 필요.
PLAY_RES = (1920, 1080)

# TODO: Default 스타일 폰트가 Arial이라 한글 자모를 그리지 못함 — 렌더 시
#       force_style로 우회 중. 한글 지원 폰트를 기본값으로 확정 필요.
_HEADER = f"""\
[Script Info]
Title: Hearing to Seeing
ScriptType: v4.00+
PlayResX: {PLAY_RES[0]}
PlayResY: {PLAY_RES[1]}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,36,&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,0,0,0,0,100,100,0,0,1,3,1,2,160,160,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _display_width(text: str) -> int:
    """Width of `text` in terminal-style display cells.

    Hangul, kana and CJK ideographs render at roughly twice the advance of a
    Latin glyph, so counting characters would let a Korean line run about twice
    as wide as the same budget allows in English.
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _split_into_lines(words: list[WordEntry]) -> list[list[WordEntry]]:
    """Splits the word stream into one group per on-screen subtitle."""
    lines: list[list[WordEntry]] = []
    current: list[WordEntry] = []
    width = 0  # display width of `current` once joined by spaces

    for word in words:
        word_width = _display_width(word.text)
        if current:
            prev = current[-1]
            should_break = (
                word.speaker != prev.speaker
                or word.start - prev.end > MAX_GAP
                or width + 1 + word_width > MAX_WIDTH
                or word.end - current[0].start > MAX_DURATION
                or (prev.text.endswith(_TERMINATORS) and width >= MIN_WIDTH_FOR_BREAK)
            )
            if should_break:
                lines.append(current)
                current = []
                width = 0
        width += word_width + 1 if current else word_width
        current.append(word)

    if current:
        lines.append(current)
    return lines

def _row_break_index(words: list[WordEntry]) -> int | None:
    """Index of the word that should start a second row, or None to keep one row.

    WrapStyle 0 only wraps once the text outgrows the frame, and at these font
    sizes a full-width row holds far more than is comfortable to read — so the
    break is placed here instead, at whichever word splits the line most evenly.
    """
    widths = [_display_width(w.text) for w in words]
    total = sum(widths) + len(words) - 1
    if total <= ROW_WIDTH or len(words) < 2:
        return None

    best_index, best_delta = 1, None
    running = widths[0]
    for i in range(1, len(words)):
        delta = abs(running - (total - running - 1))
        if best_delta is None or delta < best_delta:
            best_index, best_delta = i, delta
        running += widths[i] + 1
    return best_index


def _render_line(words: list[WordEntry], color: str, durations: list[int]) -> str:
    # \kf fills each word left-to-right in the speaker colour; until the fill
    # reaches a word it is drawn in SecondaryColour (\2c), the neutral base.
    # The \t transforms from sync.lift_tags() ride the same clock, so each word
    # rises at the moment its fill arrives.
    break_at = _row_break_index(words)
    lifts = lift_tags(durations)
    chunks = [f"{{\\1c{color}\\2c{BASE_COLOUR}}}"]
    for i, (word, cs, lift) in enumerate(zip(words, durations, lifts)):
        if i:
            chunks.append("\\N" if i == break_at else " ")
        fs = compute_font_size(word.volume)
        chunks.append(f"{{\\kf{cs}\\fs{fs}{lift}}}{word.text}")
    return "".join(chunks)


def generate_ass(transcript: Transcript) -> str:
    color_map = assign_speaker_colors(transcript.speakers())
    lines = [_HEADER]

    groups = [g for g in _split_into_lines(transcript.words) if g]
    for i, group in enumerate(groups):
        speaker = group[0].speaker
        start = group[0].start
        durations = karaoke_durations(group)
        # Deriving the end from the karaoke sum keeps the two in step: the fill
        # can neither finish early nor be cut off mid-line.
        end = start + sum(durations) / 100 + HOLD
        if i + 1 < len(groups):
            end = min(end, groups[i + 1][0].start)
        end = max(end, start + sum(durations) / 100)
        color = color_map.get(speaker, BASE_COLOUR)
        text = _render_line(group, color, durations)
        lines.append(
            f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Default,{speaker},0,0,0,,{text}"
        )

    return "\n".join(lines) + "\n"


def write_ass(transcript: Transcript, output_path: str) -> None:
    content = generate_ass(transcript)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
