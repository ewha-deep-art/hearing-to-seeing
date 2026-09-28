import unicodedata

from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.design.layout import text_width
from hearing_to_seeing.design.speaker import BASE_COLOUR, speaker_colors
from hearing_to_seeing.design.sync import (
    build_fill_text,
    build_loud_event_text,
    build_wave_events,
    build_whisper_event_text,
    karaoke_durations,
    spoken_end,
)
from hearing_to_seeing.design.volume import BASE_FONT_SIZE, classify_volume

# --- Subtitle line segmentation ------------------------------------------------
# A single Dialogue event is one on-screen subtitle. Without these limits the
# whole transcript renders as one event that sits on screen for its full
# duration, which is what "all the text at once" looks like.
#
# split_into_lines breaks the word stream into groups, in two passes:
#
# 1. Forced breaks (each checked word-to-word against the previous word):
#    - speaker changes
#    - the silence since the previous word exceeds MAX_GAP
#    - adding the word would push the line past MAX_WIDTH display cells
#    - the line's span would exceed MAX_DURATION seconds
#    - the previous word ends a sentence (_TERMINATORS) -- a line always ends
#      at a terminator, however short, so no sentence is split across the same
#      subtitle as the one after it
#
# 2. Merge pass (_merge_unterminated_lines): a line that ends without a
#    terminator only exists because one of the non-punctuation rules above cut
#    it off mid-sentence. If the very next line is the one that finally
#    reaches a terminator, the two are reattached into a single subtitle --
#    but only when doing so still respects every limit from pass 1 (same
#    speaker, gap, width, duration). A forced break whose cause would still
#    hold true for the merged line is never undone.
ROW_WIDTH = 42          # display cells per on-screen row
MAX_WIDTH = 2 * ROW_WIDTH  # a subtitle may occupy at most two rows
MAX_DURATION = 6.0      # seconds a single subtitle may stay on screen
MAX_GAP = 0.7           # a silence this long ends the current subtitle
HOLD = 0.3              # extra seconds the finished line lingers

_TERMINATORS = (".", "?", "!", "…")

# The coordinate space every position, margin and font size in the script is
# expressed in; the preview renderer sizes its canvas to match (render.py).
# TODO: 파라미터로 받거나 ffprobe로 입력 영상 해상도를 자동 감지하도록 개선 필요.
PLAY_RES = (1920, 1080)

# Horizontal and bottom margins, shared by the styles and by
# `_compute_word_layout`, which has to reproduce where libass puts each row.
MARGIN_H = 160
MARGIN_V = 60

# The face the web player renders every line in (watch.js), bundled for layout
# measurement in design/fonts/ — the render preview passes that directory to
# libass, so both rendering paths and the measured positions agree.
# TODO: 폰트 선택(Pretendard SemiBold 확정 여부)은 기획 미확정.
_DEFAULT_FONT = "Pretendard SemiBold"

# Default is the plain baseline (`generate_plain_ass`). The dynamic subtitle is
# three layers — Box, Fill, Pop — described in design/sync.py.
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
Style: Default,{_DEFAULT_FONT},{BASE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,0,0,0,0,100,100,0,0,1,3,1,2,{MARGIN_H},{MARGIN_H},{MARGIN_V},1
Style: Box,{_DEFAULT_FONT},{BASE_FONT_SIZE},&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,3,4,0,2,{MARGIN_H},{MARGIN_H},{MARGIN_V},1
Style: Fill,{_DEFAULT_FONT},{BASE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,{MARGIN_H},{MARGIN_H},{MARGIN_V},1
Style: Pop,{_DEFAULT_FONT},{BASE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,5,{MARGIN_H},{MARGIN_H},{MARGIN_V},1

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


def split_into_lines(words: list[WordEntry]) -> list[list[WordEntry]]:
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
                or prev.text.endswith(_TERMINATORS)
            )
            if should_break:
                lines.append(current)
                current = []
                width = 0
        width += word_width + 1 if current else word_width
        current.append(word)

    if current:
        lines.append(current)
    return _merge_unterminated_lines(lines)


def _merge_unterminated_lines(lines: list[list[WordEntry]]) -> list[list[WordEntry]]:
    """Reattaches a line cut off mid-sentence to the fragment that finishes it.

    A line ends up without a sentence terminator only when something other than
    punctuation forced the break (speaker change, silence, width, duration). If
    the very next line is the one that finally reaches a terminator, they are
    really one sentence split across two events -- so they're merged back into
    one, as long as doing so still respects the same speaker/gap/width/duration
    limits `split_into_lines` enforces on every other line.
    """
    merged: list[list[WordEntry]] = []
    for line in lines:
        if merged and _can_merge(merged[-1], line):
            merged[-1] = merged[-1] + line
        else:
            merged.append(line)
    return merged


def _can_merge(prev_line: list[WordEntry], next_line: list[WordEntry]) -> bool:
    if prev_line[-1].text.endswith(_TERMINATORS):
        return False  # prev line already closes its own sentence
    if not next_line[-1].text.endswith(_TERMINATORS):
        return False  # only pull in a line that closes the sentence
    if prev_line[-1].speaker != next_line[0].speaker:
        return False
    if next_line[0].start - prev_line[-1].end > MAX_GAP:
        return False
    combined_width = _display_width(" ".join(w.text for w in prev_line + next_line))
    if combined_width > MAX_WIDTH:
        return False
    if next_line[-1].end - prev_line[0].start > MAX_DURATION:
        return False
    return True

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


def _rows(words: list[WordEntry], break_at: int | None) -> list[list[WordEntry]]:
    if break_at is None:
        return [words]
    return [words[:break_at], words[break_at:]]


def _compute_word_layout(
    words: list[WordEntry], break_at: int | None,
) -> list[tuple[float, float]]:
    """(left x, centre y) of every word, where libass draws it in the Fill line.

    Each row is centred between the margins on its own width, and rows stack
    upwards from the bottom margin one font height apart — libass sizes a font
    so its ascent + descent equal the font size, which is also its line height.
    Offsets are measured on the joined prefix rather than summed per word, so
    kerning across a word boundary is counted the way libass counts it.
    """
    rows = _rows(words, break_at)
    centre_x = MARGIN_H + (PLAY_RES[0] - 2 * MARGIN_H) / 2
    bottom_y = PLAY_RES[1] - MARGIN_V - BASE_FONT_SIZE / 2

    layout = []
    for r, row in enumerate(rows):
        cy = bottom_y - (len(rows) - 1 - r) * BASE_FONT_SIZE
        texts = [w.text for w in row]
        left = centre_x - text_width(" ".join(texts), BASE_FONT_SIZE) / 2
        for i in range(len(row)):
            prefix = "".join(t + " " for t in texts[:i])
            layout.append((left + text_width(prefix, BASE_FONT_SIZE), cy))
    return layout


def _char_positions(word: WordEntry, left_x: float, cy: float) -> list[tuple[str, float, float]]:
    """(character, centre x, centre y) for each character of `word`."""
    positions = []
    for i, ch in enumerate(word.text):
        start = left_x + text_width(word.text[:i], BASE_FONT_SIZE)
        end = left_x + text_width(word.text[: i + 1], BASE_FONT_SIZE)
        positions.append((ch, (start + end) / 2, cy))
    return positions


def _box_text(words: list[WordEntry], break_at: int | None) -> str:
    return "".join(
        ("\\N" if i == break_at else " ") + w.text if i else w.text
        for i, w in enumerate(words)
    )


def _pop_events(
    word: WordEntry, color: str, left_x: float, cy: float,
) -> list[tuple[float, float, str]]:
    """Layer-2 events for one word, chosen by how loudly it was said."""
    category = classify_volume(word.volume)
    if category == "normal":
        return build_wave_events(word, color, _char_positions(word, left_x, cy))

    cx = left_x + text_width(word.text, BASE_FONT_SIZE) / 2
    build = build_loud_event_text if category == "loud" else build_whisper_event_text
    return [(word.start, spoken_end(word), build(word, color, cx, cy))]


def subtitle_events(
    transcript: Transcript,
) -> list[tuple[list[WordEntry], float, float, list[int]]]:
    """One `(words, start, end, karaoke centiseconds)` entry per on-screen subtitle."""
    groups = [g for g in split_into_lines(transcript.words) if g]
    events = []
    for i, group in enumerate(groups):
        start = group[0].start
        durations = karaoke_durations(group)
        # Deriving the end from the karaoke sum keeps the two in step: the fill
        # can neither finish early nor be cut off mid-line.
        end = start + sum(durations) / 100 + HOLD
        if i + 1 < len(groups):
            end = min(end, groups[i + 1][0].start)
        end = max(end, start + sum(durations) / 100)
        events.append((group, start, end, durations))
    return events


def _dialogue(
    start: float, end: float, speaker: str, text: str,
    style: str = "Default", layer: int = 0,
) -> str:
    return (
        f"Dialogue: {layer},{_fmt_time(start)},{_fmt_time(end)},"
        f"{style},{speaker},0,0,0,,{text}"
    )


def generate_ass(transcript: Transcript) -> str:
    color_map = speaker_colors(transcript)
    lines = [_HEADER]
    for group, start, end, _ in subtitle_events(transcript):
        speaker = group[0].speaker
        color = color_map.get(speaker, BASE_COLOUR)
        break_at = _row_break_index(group)

        lines.append(_dialogue(start, end, speaker, _box_text(group, break_at), "Box", 0))
        lines.append(
            _dialogue(start, end, speaker, build_fill_text(group, color, break_at), "Fill", 1)
        )
        for word, (left_x, cy) in zip(group, _compute_word_layout(group, break_at)):
            for w_start, w_end, text in _pop_events(word, color, left_x, cy):
                lines.append(_dialogue(w_start, w_end, speaker, text, "Pop", 2))
    return "\n".join(lines) + "\n"


def generate_plain_ass(transcript: Transcript) -> str:
    """The same subtitles as `generate_ass` in plain white at one size.

    A baseline for comparison: identical line breaks and timing, with the
    speaker colour, box, fill and volume motion left out.
    """
    lines = [_HEADER]
    for group, start, end, _ in subtitle_events(transcript):
        text = _box_text(group, _row_break_index(group))
        lines.append(_dialogue(start, end, group[0].speaker, text))
    return "\n".join(lines) + "\n"


def write_ass(transcript: Transcript, output_path: str) -> None:
    content = generate_ass(transcript)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
