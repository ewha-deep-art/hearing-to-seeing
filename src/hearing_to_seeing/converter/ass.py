import unicodedata

from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.design.layout import BASE_FONT_SIZE, FONT_BOLD, FONT_NAME, space_scale
from hearing_to_seeing.design.speaker import BASE_COLOUR, speaker_colors
from hearing_to_seeing.design.sync import (
    fall_move,
    hand_over_ms,
    hide_tag,
    karaoke_durations,
    rise_move,
    spoken_end,
    wave_chars,
    wave_hand_over_ms,
    wipe_tags,
)
from hearing_to_seeing.design.volume import (
    classify_volume,
    loud_tags,
    narrow_tag,
    whisper_lift,
    whisper_phases,
    whisper_return_tags,
    whisper_tags,
)

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

# Horizontal and bottom margins of every style; Pop events are anchored on the
# same bottom centre (`_ANCHOR`).
MARGIN_H = 160
MARGIN_V = 60

# Padding (script px) of the opaque box around the text. 6 rather than the 4 it
# once was: with whispered words narrowed by \\fscx the per-run boxes meet at
# fractional pixels, and at 4 a hairline gap showed between them (none at 6, at
# 1080p or when scaled down to 720p).
BOX_PAD = 6

# The one bundled face (design/layout.py) for every style.
_BOLD = -1 if FONT_BOLD else 0

# Default is the plain baseline (`generate_plain_ass`). The dynamic subtitle is
# drawn as three stacked layers of Dialogue events:
#   0. Box  — the whole line in an opaque black box, so it reads over any video.
#   1. Fill — the whole line again, white until a word is said, then hidden
#             for good (`_fill_text`).
#   2. Pop  — one event per word (two per character for a waved word) that
#             draws it from then until the line ends, carrying the effects:
#             size from volume.py, fill wipe and wave from sync.py, colour from
#             speaker.py (`_pop_events`).
# ASS moves and scales apply to a whole event, which is why layer 2 splits
# words out instead of animating inside the line.
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
Style: Default,{FONT_NAME},{BASE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,{_BOLD},0,0,0,100,100,0,0,1,3,1,2,{MARGIN_H},{MARGIN_H},{MARGIN_V},1
Style: Box,{FONT_NAME},{BASE_FONT_SIZE},&H00000000,&H00000000,&H00000000,&H00000000,{_BOLD},0,0,0,100,100,0,0,3,{BOX_PAD},0,2,{MARGIN_H},{MARGIN_H},{MARGIN_V},1
Style: Fill,{FONT_NAME},{BASE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,{_BOLD},0,0,0,100,100,0,0,1,0,0,2,{MARGIN_H},{MARGIN_H},{MARGIN_V},1
Style: Pop,{FONT_NAME},{BASE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,{_BOLD},0,0,0,100,100,0,0,1,3,0,5,{MARGIN_H},{MARGIN_H},{MARGIN_V},1

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


# The `\an2` anchor of a whole line: bottom centre, where the Box and Fill styles
# put it through their margins.
_ANCHOR = (PLAY_RES[0] // 2, PLAY_RES[1] - MARGIN_V)

# What a word looks like before it is said (the styles' SecondaryColour too).
_WHITE = "&H00FFFFFF&"


def _separator(words: list[WordEntry], i: int, break_at: int | None) -> str:
    """What goes before word `i`: nothing, a row break, or a (narrowed) space.

    \\fscx is reset right after a narrowed space, so the next word is unaffected.
    """
    if i == 0:
        return ""
    if i == break_at:
        return "\\N"
    pct = space_scale(words[i - 1], words[i])
    return " " if pct >= 100 else f"{{\\fscx{pct}}} {{\\fscx100}}"


def _narrowed(word: WordEntry) -> str:
    """`word` taking up the width volume.py leaves it (all words keep their full width)."""
    tag = narrow_tag(word.volume)
    return f"{{{tag}}}{word.text}{{\\fscx100}}" if tag else word.text


def _box_text(
    words: list[WordEntry], break_at: int | None, narrow_whispers: bool = False,
) -> str:
    """Layer 0: the line as plain text in an opaque box (or the plain baseline).

    With `narrow_whispers` every word and space takes the width it has in the
    Fill layer, so the box is as wide as the visible text.
    """
    if not narrow_whispers:
        return "".join(
            ("\\N" if i == break_at else " ") + w.text if i else w.text
            for i, w in enumerate(words)
        )
    return "".join(_separator(words, i, break_at) + _narrowed(w) for i, w in enumerate(words))


def _fill_text(words: list[WordEntry], break_at: int | None) -> str:
    """Layer 1: white before a word is said, hidden from the moment it is.

    Once hidden a word is never shown again here: the Pop layer keeps drawing
    it until the line ends, so the same glyphs are never handed back and forth
    between two layers (that is what made words twitch). A waved word is
    handed over character by character, as each one starts to rise.

    A whispered word is handed over like a shouted one, at the moment it is
    said: until then it is drawn here at full size and full width like every
    other word, and the Pop layer then shrinks it about its own centre and
    returns it. The placeholder keeps its full width, so the box does not
    change size.
    """
    if not words:
        return ""
    line_start = words[0].start
    head = f"\\fs{BASE_FONT_SIZE}\\c{_WHITE}\\alpha&H00&"
    parts = []
    for i, word in enumerate(words):
        parts.append(_separator(words, i, break_at))
        if classify_volume(word.volume) == "normal":
            for ch, at in zip(word.text, wave_hand_over_ms(word, line_start)):
                parts.append(f"{{{head}{hide_tag(at)}}}{ch}")
        else:  # loud and whisper: the whole word is handed over when it is said
            parts.append(f"{{{head}{hide_tag(hand_over_ms(word, line_start))}}}{word.text}")
    return "".join(parts)


def _invisible_pieces(words: list[WordEntry], break_at: int | None) -> list[str]:
    """Each word as it takes up room in the Fill line, with the separator before it."""
    return [_separator(words, i, break_at) + _narrowed(w) for i, w in enumerate(words)]


def _word_event_text(
    words: list[WordEntry], break_at: int | None, index: int,
    head: str, tags: str, lead: str, sweep: str,
) -> str:
    """The whole line with only `words[index]` visible, laid out by libass.

    Every other word is invisible, so libass puts the visible one at the very
    pixel the Fill layer drew it at (same font, same kerning, same rounding) —
    nothing is measured. `head` positions the line (\\pos or \\move), `tags`
    are the word's size tags, `lead`/`sweep` its fill wipe. Because the
    invisible line is re-centred as the word's width changes, a horizontal
    \\fscx growth stays centred on the word; vertically it grows from the
    baseline.
    """
    pieces = _invisible_pieces(words, break_at)
    before = "".join(pieces[:index]) + _separator(words, index, break_at)
    after = "".join(pieces[index + 1:])
    return (
        f"{{{head}\\fs{BASE_FONT_SIZE}\\alpha&HFF&}}{before}"
        f"{{\\alpha&H00&{tags}{lead}}}{{{sweep}}}{words[index].text}"
        f"{{\\alpha&HFF&\\fscx100\\fscy100}}{after}"
    )


def _wave_events(
    words: list[WordEntry], break_at: int | None, index: int, color: str, line_end: float,
) -> list[tuple[float, float, str]]:
    """Layer 2, normally spoken word: its characters lift and land in a ripple.

    Each event carries the whole line with one character visible (see
    `_word_event_text`), moved by sync.py's wave. ASS allows one \\move per
    event, so each character is two events — the rise, white, then the fall,
    in which it is wiped into `color`. A \\move stays at its end point, so the
    fall event simply lasts until the line ends.
    """
    word = words[index]
    pieces = _invisible_pieces(words, break_at)
    before_word = "".join(pieces[:index]) + _separator(words, index, break_at)
    after_word = "".join(pieces[index + 1:])
    end = spoken_end(word)

    events = []
    for i, c in enumerate(wave_chars(word)):
        before = before_word + word.text[:i]
        after = word.text[i + 1:] + after_word
        ch = word.text[i]
        events.append((
            c.rise_start, c.rise_end,
            f"{{\\an2{rise_move(_ANCHOR, c.rise_ms)}\\fs{BASE_FONT_SIZE}\\alpha&HFF&}}{before}"
            f"{{\\alpha&H00&\\c{_WHITE}}}{ch}{{\\alpha&HFF&}}{after}",
        ))
        lead, sweep = wipe_tags(c.rise_end, c.wipe_start, c.wipe_end)
        events.append((
            c.rise_end, max(c.rise_end + c.fall_ms / 1000, end, line_end),
            f"{{\\an2{fall_move(_ANCHOR, c.fall_ms)}\\fs{BASE_FONT_SIZE}\\alpha&HFF&}}{before}"
            f"{{\\alpha&H00&\\c{color}{lead}}}{{{sweep}}}{ch}{{\\alpha&HFF&}}{after}",
        ))
    return events


def _pop_events(
    words: list[WordEntry], break_at: int | None, index: int,
    color: str, line_start: float, line_end: float,
) -> list[tuple[float, float, str]]:
    """Layer-2 events for one word: how loudly it was said picks the motion.

      loud    → grows (more when louder), holds, then returns (volume.py)
      whisper → shrinks about its own centre, holds, then returns (volume.py)
      normal  → its characters ripple up and down (sync.py)

    Each is wiped into the speaker's `color` while it is said (sync.py). Every
    event lasts until `line_end`: the Fill layer hides the word the moment it
    is said and never shows it again.
    """
    word = words[index]
    kind = classify_volume(word.volume)
    if kind == "normal":
        return _wave_events(words, break_at, index, color, line_end)

    end = max(line_end, spoken_end(word))
    if kind == "whisper":
        x, y = _ANCHOR
        spoken = spoken_end(word) - word.start
        in_ms, back_ms, out_ms = whisper_phases(spoken)
        back_at = word.start + back_ms / 1000
        # The line is bottom-anchored, so a shrinking word would sink towards
        # the baseline; the whole (otherwise invisible) line is raised by the
        # same linear ramp as the shrink, so the word scales about its centre.
        # ASS allows one \\move per event, so the word is two events back to
        # back: shrink + hold (lifted), then return (lowered again).
        lift = whisper_lift(word.volume, BASE_FONT_SIZE)
        head_in = f"\\an2\\move({x},{y},{x},{y - lift},0,{in_ms})"
        head_out = f"\\an2\\move({x},{y - lift},{x},{y},0,{out_ms})"
        # The wipe has to be finished when the second event takes over (it
        # cannot be resumed half way), so it runs until the return starts.
        lead, sweep = wipe_tags(word.start, word.start, back_at)
        shrink = _word_event_text(
            words, break_at, index, head_in,
            f"\\c{color}{whisper_tags(word.volume, spoken)}", lead, sweep,
        )
        restore = _word_event_text(
            words, break_at, index, head_out,
            f"\\c{color}{whisper_return_tags(word.volume, spoken)}", "", "",
        )
        return [(word.start, back_at, shrink), (back_at, end, restore)]

    head = f"\\an2\\pos({_ANCHOR[0]},{_ANCHOR[1]})"
    lead, sweep = wipe_tags(word.start, word.start, spoken_end(word))
    tags = f"\\c{color}{loud_tags(word.volume, spoken_end(word) - word.start)}"
    return [(word.start, end, _word_event_text(words, break_at, index, head, tags, lead, sweep))]

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
        lines.append(_dialogue(start, end, speaker, _box_text(group, break_at, True), "Box", 0))
        lines.append(_dialogue(start, end, speaker, _fill_text(group, break_at), "Fill", 1))
        for idx in range(len(group)):
            for w_start, w_end, text in _pop_events(group, break_at, idx, color, start, end):
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