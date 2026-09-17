from itertools import groupby

from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.design.speaker import assign_speaker_colors

from hearing_to_seeing.design.sync import build_fill_text, build_pop_event_text
from hearing_to_seeing.design.volume import compute_font_size
from hearing_to_seeing.design.layout import text_width


PLAY_RES_X = 1920
PLAY_RES_Y = 1080
MARGIN_L = 10
MARGIN_R = 10
MARGIN_V = 30

POP_SCALE = 130          # 말하는 순간 팝업 크기 배율(%)
BOX_FONT_SIZE = 40  # 박스는 항상 이 크기로 고정 (넘치는 건 허용)

MAX_LINE_CHARS = 20
GAP_THRESHOLD = 0.7
SENTENCE_ENDINGS = (".", "!", "?", "…")

FONT_NAME = "Pretendard ExtraBold"  # 실제 설치된 이름으로 확인 후 맞춰주세요

_HEADER = f"""\
[Script Info]
Title: Hearing to Seeing
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Box,{FONT_NAME},40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,3,4,0,2,10,10,30,1
Style: Fill,{FONT_NAME},40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,10,10,30,1
Style: Pop,{FONT_NAME},40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,5,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _group_by_speaker(words: list[WordEntry]) -> list[tuple[str, list[WordEntry]]]:
    return [(speaker, list(group)) for speaker, group in groupby(words, key=lambda w: w.speaker)]


def _ends_sentence(word: WordEntry) -> bool:
    return word.text.rstrip().endswith(SENTENCE_ENDINGS)


def _split_into_chunks(
    words: list[WordEntry],
    max_chars: int = MAX_LINE_CHARS,
    gap_threshold: float = GAP_THRESHOLD,
) -> list[list[WordEntry]]:
    chunks: list[list[WordEntry]] = []
    current: list[WordEntry] = []
    current_len = 0

    for word in words:
        added_len = len(word.text) + (1 if current else 0)
        gap = word.start - current[-1].end if current else 0.0
        exceeds_chars = bool(current) and current_len + added_len > max_chars
        exceeds_gap = bool(current) and gap > gap_threshold

        if exceeds_chars or exceeds_gap:
            chunks.append(current)
            current = [word]
            current_len = len(word.text)
        else:
            current.append(word)
            current_len += added_len

        if _ends_sentence(word):
            chunks.append(current)
            current = []
            current_len = 0

    if current:
        chunks.append(current)
    return chunks

def _build_box_text(chunk: list[WordEntry]) -> str:
    """박스 모양 전용: Fill과 똑같은 실제 크기를 써서 타이트하게 맞춤."""
    parts = []
    for word in chunk:
        fs = compute_font_size(word.volume)
        parts.append(f"{{\\fs{fs}\\c&H000000&}}{word.text}")
    return " ".join(parts)


def _compute_word_positions(chunk: list[WordEntry]) -> list[tuple[float, float, int]]:
    sizes = [compute_font_size(w.volume) for w in chunk]
    widths = [text_width(w.text, fs) for w, fs in zip(chunk, sizes)]
    space_width = text_width(" ", 40)

    total_width = sum(widths) + space_width * max(len(chunk) - 1, 0)
    available_width = PLAY_RES_X - MARGIN_L - MARGIN_R
    cursor = MARGIN_L + (available_width - total_width) / 2

    positions = []
    for width, fs in zip(widths, sizes):
        center_x = cursor + width / 2
        center_y = PLAY_RES_Y - MARGIN_V - fs / 2
        positions.append((center_x, center_y, fs))
        cursor += width + space_width
    return positions


def generate_ass(transcript: Transcript) -> str:
    color_map = assign_speaker_colors(transcript.speakers())
    lines = [_HEADER]

    for speaker, group in _group_by_speaker(transcript.words):
        if not group:
            continue
        color = color_map.get(speaker, "&H00FFFFFF")

        for chunk in _split_into_chunks(group):
            start, end = chunk[0].start, chunk[-1].end

            box_text = _build_box_text(chunk)
            lines.append(
                f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Box,{speaker},0,0,0,,{box_text}"
            )

            fill_text = build_fill_text(chunk, color)
            lines.append(
                f"Dialogue: 1,{_fmt_time(start)},{_fmt_time(end)},Fill,{speaker},0,0,0,,{fill_text}"
            )

            positions = _compute_word_positions(chunk)
            for word, (cx, cy, fs) in zip(chunk, positions):
                pop_text = build_pop_event_text(word, color, cx, cy, fs)
                lines.append(
                    f"Dialogue: 2,{_fmt_time(word.start)},{_fmt_time(word.end)},Pop,{speaker},0,0,0,,{pop_text}"
                )

    return "\n".join(lines) + "\n"

def _build_pop_events(
    chunk: list[WordEntry], color: str, positions: list[tuple[float, float, int]]
) -> list[tuple[float, float, str]]:
    events = []
    for word, (cx, cy, fs) in zip(chunk, positions):
        pop_fs = round(fs * POP_SCALE / 100)
        text = f"{{\\an5\\pos({cx:.0f},{cy:.0f})\\fs{pop_fs}\\c{color}}}{word.text}"
        events.append((word.start, word.end, text))
    return events

def write_ass(transcript: Transcript, output_path: str) -> None:
    content = generate_ass(transcript)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)