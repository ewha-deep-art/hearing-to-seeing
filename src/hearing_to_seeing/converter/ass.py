import unicodedata
from functools import lru_cache

from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.design.font import (
    DEFAULT_WEIGHT,
    resolve_faces,
    speaker_weights,
)
from hearing_to_seeing.design.layout import text_width
from hearing_to_seeing.design.speaker import BASE_COLOUR, speaker_colors
from hearing_to_seeing.design.sync import (
    WaveLineContext,
    WordLineContext,
    build_fill_text,
    build_loud_event_text,
    build_wave_events,
    space_scale,
    width_scale,
    word_separator,
    build_whisper_event_text,
    whisper_scale,
    karaoke_durations,
    spoken_end,
)
from hearing_to_seeing.design.volume import BASE_FONT_SIZE, classify_volume

# Layout measures the same prefixes over and over (every word of every line, and
# repeated words across the transcript), so measure each distinct string once.
_text_width = lru_cache(maxsize=None)(text_width)

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

# The base face, bundled in design/fonts/ — the render preview passes that
# directory to libass. Per-speaker weights (design/font.py) replace it in the
# speaker styles; this is the face of the plain baseline and the base styles.
_DEFAULT_FONT = "NanumGothic"

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


def _weight_styles(faces: dict[str, tuple[str, bool]], used: list[str]) -> str:
    """Box / Fill / Pop styles for every weight in `used`, named e.g. `Fill-Bold`.

    They repeat the three base styles with only the face changed, so a speaker's
    weight is the style its events are in; no font tag is needed in the text.
    """
    tail = f"{MARGIN_H},{MARGIN_H},{MARGIN_V},1"
    lines = []
    for weight in used:
        font, bold = faces[weight]
        b = -1 if bold else 0
        lines += [
            f"Style: Box-{weight},{font},{BASE_FONT_SIZE},&H00000000,&H00000000,&H00000000,&H00000000,{b},0,0,0,100,100,0,0,3,4,0,2,{tail}",
            f"Style: Fill-{weight},{font},{BASE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,{b},0,0,0,100,100,0,0,1,0,0,2,{tail}",
            f"Style: Pop-{weight},{font},{BASE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,{b},0,0,0,100,100,0,0,1,3,0,5,{tail}",
        ]
    return "\n".join(lines)


def _header_with_weights(faces: dict[str, tuple[str, bool]], used: list[str]) -> str:
    head, events = _HEADER.split("\n\n[Events]", 1)
    return f"{head}\n{_weight_styles(faces, used)}\n\n[Events]{events}"


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


def _width_scale(word: WordEntry) -> float:
    """Horizontal scale of `word` in the Box and Fill lines (whispers are narrowed)."""
    return width_scale(word)


def _compute_word_layout(
    words: list[WordEntry], break_at: int | None,
) -> list[tuple[float, float]]:
    """(left x, centre y) of every word, where libass draws it in the Fill line.

    Each row is centred between the margins on its own width, and rows stack
    upwards from the bottom margin one font height apart — libass sizes a font
    so its ascent + descent equal the font size, which is also its line height.
    Offsets are measured on the joined prefix rather than summed per word, so
    kerning across a word boundary is counted the way libass counts it. A row
    that holds a whispered word is instead summed word by word, each word at its
    narrowed width (the Box and Fill lines scale it with \\fscx), so kerning
    across its boundaries is not counted there.
    """
    rows = _rows(words, break_at)
    centre_x = MARGIN_H + (PLAY_RES[0] - 2 * MARGIN_H) / 2
    bottom_y = PLAY_RES[1] - MARGIN_V - BASE_FONT_SIZE / 2

    layout = []
    for r, row in enumerate(rows):
        cy = bottom_y - (len(rows) - 1 - r) * BASE_FONT_SIZE
        texts = [w.text for w in row]
        scales = [_width_scale(w) for w in row]
        if all(sc == 1.0 for sc in scales):
            left = centre_x - _text_width(" ".join(texts), BASE_FONT_SIZE) / 2
            prefix = ""
            for t in texts:
                layout.append((left + _text_width(prefix, BASE_FONT_SIZE), cy))
                prefix += t + " "
            continue

        # Each space is narrowed the same way word_separator narrows it in the
        # Box/Fill lines, so Pop words land where libass puts the placeholders.
        space = _text_width(" ", BASE_FONT_SIZE)
        widths = [_text_width(t, BASE_FONT_SIZE) * sc for t, sc in zip(texts, scales)]
        gaps = [
            space * space_scale(row[j], row[j + 1]) / 100
            for j in range(len(row) - 1)
        ]
        x = centre_x - (sum(widths) + sum(gaps)) / 2
        for j, wd in enumerate(widths):
            layout.append((x, cy))
            x += wd + (gaps[j] if j < len(gaps) else 0)
    return layout


def _char_positions(word: WordEntry, left_x: float, cy: float) -> list[tuple[str, float, float]]:
    """(character, centre x, centre y) for each character of `word`."""
    text = word.text
    # Edge i is where character i starts; measuring each prefix once gives both
    # the start and the end of every character.
    edges = [left_x + _text_width(text[:i], BASE_FONT_SIZE) for i in range(len(text) + 1)]
    return [(ch, (edges[i] + edges[i + 1]) / 2, cy) for i, ch in enumerate(text)]


def _box_text(
    words: list[WordEntry], break_at: int | None, narrow_whispers: bool = False,
) -> str:
    """The line as plain text for the Box layer (or the plain baseline).

    With `narrow_whispers` a whispered word is narrowed with \\fscx, matching its
    placeholder in the Fill layer, so the box is as wide as the visible text.
    """
    def piece(w: WordEntry) -> str:
        if narrow_whispers and classify_volume(w.volume) == "whisper":
            return f"{{\\fscx{whisper_scale(w.volume)}}}{w.text}{{\\fscx100}}"
        return w.text

    def sep(i: int) -> str:
        if i == break_at:
            return "\\N"
        return word_separator(words[i - 1], words[i]) if narrow_whispers else " "

    return "".join(sep(i) + piece(w) if i else piece(w) for i, w in enumerate(words))


def _separator(words: list[WordEntry], i: int, break_at: int | None) -> str:
    """What goes before word `i`: nothing, a row break, or a (scaled) space."""
    if i == 0:
        return ""
    if i == break_at:
        return "\\N"
    return word_separator(words[i - 1], words[i])


def _invisible_pieces(words: list[WordEntry], break_at: int | None) -> list[str]:
    """Each word as it appears in the Fill layout, with the separator before it."""
    pieces = []
    for i, w in enumerate(words):
        sep = _separator(words, i, break_at)
        if classify_volume(w.volume) == "whisper":
            body = f"{{\\fscx{whisper_scale(w.volume)}}}{w.text}{{\\fscx100}}"
        else:
            body = w.text
        pieces.append(sep + body)
    return pieces


def _wave_context(
    words: list[WordEntry], break_at: int | None, index: int,
) -> WaveLineContext:
    """Per-character before/after text so libass lays a wave character out itself."""
    pieces = _invisible_pieces(words, break_at)
    sep = _separator(words, index, break_at)
    text = words[index].text
    before_words = "".join(pieces[:index]) + sep
    after_words = "".join(pieces[index + 1:])
    before = [before_words + text[:i] for i in range(len(text))]
    after = [text[i + 1:] + after_words for i in range(len(text))]
    anchor = (PLAY_RES[0] // 2, PLAY_RES[1] - MARGIN_V)
    return WaveLineContext(anchor, before, after)


def _word_context(
    words: list[WordEntry], break_at: int | None, index: int,
) -> WordLineContext:
    """Line text before/after word `index` so libass lays a whole word out itself."""
    pieces = _invisible_pieces(words, break_at)
    before = "".join(pieces[:index]) + _separator(words, index, break_at)
    after = "".join(pieces[index + 1:])
    return WordLineContext((PLAY_RES[0] // 2, PLAY_RES[1] - MARGIN_V), before, after)


def _pop_events(
    word: WordEntry, color: str, left_x: float, cy: float,
    line_start: float, line_end: float,
    context: "WaveLineContext | WordLineContext | None" = None,
) -> list[tuple[float, float, str]]:
    """Layer-2 events for one word, chosen by how loudly it was said.

    Every event lasts until `line_end` (the end of the whole subtitle). The Fill
    layer hides the word the moment it is said and never shows it again, so the
    Pop layer must keep drawing it — handing the same glyphs back and forth
    between two layers made words visibly twitch.

    A whispered word is the exception on the start side: its event begins with
    the line (`line_start`), so it is small before it is even said.
    """
    category = classify_volume(word.volume)
    if category == "normal":
        # With a line context libass places every character itself, so only the
        # characters are needed -- not their measured positions.
        chars = (
            [(ch, 0.0, 0.0) for ch in word.text]
            if context is not None
            else _char_positions(word, left_x, cy)
        )
        return build_wave_events(
            word, color, chars, hold_until=line_end, line_context=context,
        )

    cx = (
        0.0 if context is not None
        else left_x + _text_width(word.text, BASE_FONT_SIZE) * _width_scale(word) / 2
    )
    end = max(line_end, spoken_end(word))
    if category == "whisper":
        return [(
            line_start, end,
            build_whisper_event_text(word, color, cx, cy, line_start, context),
        )]
    return [(word.start, end, build_loud_event_text(word, color, cx, cy, context))]


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
    weights = speaker_weights(transcript)
    used = sorted(set(weights.values()) | {DEFAULT_WEIGHT})
    lines = [_header_with_weights(resolve_faces(used), used)]
    for group, start, end, _ in subtitle_events(transcript):
        speaker = group[0].speaker
        color = color_map.get(speaker, BASE_COLOUR)
        break_at = _row_break_index(group)
        weight = weights.get(speaker, DEFAULT_WEIGHT)

        lines.append(
            _dialogue(start, end, speaker, _box_text(group, break_at, True), f"Box-{weight}", 0)
        )
        lines.append(
            _dialogue(
                start, end, speaker, build_fill_text(group, break_at), f"Fill-{weight}", 1,
            )
        )
        for idx, word in enumerate(group):
            # Every word is laid out by libass inside the invisible line, not
            # by measured coordinates, so no word can drift from the Fill layer
            # (and no font has to be measured: `_compute_word_layout` is only
            # kept for a `_pop_events` call without a context).
            context = (
                _wave_context(group, break_at, idx)
                if classify_volume(word.volume) == "normal"
                else _word_context(group, break_at, idx)
            )
            for w_start, w_end, text in _pop_events(
                word, color, 0.0, 0.0, start, end, context,
            ):
                lines.append(_dialogue(w_start, w_end, speaker, text, f"Pop-{weight}", 2))
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
