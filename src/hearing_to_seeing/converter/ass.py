from itertools import groupby

from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.design.speaker import assign_speaker_colors
from hearing_to_seeing.design.sync import karaoke_duration
from hearing_to_seeing.design.volume import compute_font_size

# TODO: PlayResX/PlayResY가 1920×1080으로 고정되어 있음 — 파라미터로 받거나
#       ffprobe로 입력 영상 해상도를 자동 감지하도록 개선 필요.
_HEADER = """\
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
Style: Default,Arial,32,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _group_by_speaker(words: list[WordEntry]) -> list[tuple[str, list[WordEntry]]]:
    groups = []
    for speaker, group in groupby(words, key=lambda w: w.speaker):
        groups.append((speaker, list(group)))
    return groups


def _render_line(words: list[WordEntry], color: str) -> str:
    # TODO: \k(와이프) vs \kf(채우기) — sync.py의 결정에 맞춰 함께 수정 필요.
    #       기획서는 와이프가 아닌 fill 애니메이션을 요구함.
    # TODO: 최대 줄 길이 제한 없음 — 화자 발화가 길면 가독성이 심각하게 떨어지는
    #       자막 줄이 생성됨. 단어 수 또는 글자 수 기준으로 Dialogue 이벤트 분할 필요.
    parts = []
    for word in words:
        cs = karaoke_duration(word)
        fs = compute_font_size(word.volume)
        parts.append(f"{{\\k{cs}\\fs{fs}\\c{color}}}{word.text}")
    return " ".join(parts)


def generate_ass(transcript: Transcript) -> str:
    color_map = assign_speaker_colors(transcript.speakers())
    lines = [_HEADER]

    for speaker, group in _group_by_speaker(transcript.words):
        if not group:
            continue
        start = group[0].start
        end = group[-1].end
        color = color_map.get(speaker, "&H00FFFFFF")
        text = _render_line(group, color)
        lines.append(
            f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Default,{speaker},0,0,0,,{text}"
        )

    return "\n".join(lines) + "\n"


def write_ass(transcript: Transcript, output_path: str) -> None:
    content = generate_ass(transcript)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
