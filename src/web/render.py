"""Burn an ASS subtitle file onto its source media to produce a preview mp4.

The pipeline's real output is the .ass file; this module exists so the result
can actually be watched. Audio-only sources (mp3) get a solid-colour video
track synthesised for the subtitles to sit on; sources that already carry
video are used as-is.

Requires `ffmpeg` (built with libass) on PATH. Set the H2S_FFMPEG environment
variable to point at a specific binary instead.
"""

import os

from hearing_to_seeing.converter.ass import PLAY_RES
from hearing_to_seeing.media import MediaToolError, run as run_tool, tool

# An audio-only source gets a canvas the size of the script's own coordinate
# space, so the subtitle layout renders at the scale it was designed for.
DEFAULT_RESOLUTION = PLAY_RES
DEFAULT_FPS = 24
DEFAULT_BACKGROUND = "black"


def has_video_stream(media_path: str) -> bool:
    out = run_tool([
        tool("ffprobe"), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        media_path,
    ]).decode(errors="replace")
    return out.strip().startswith("video")


def _escape_filter_path(path: str) -> str:
    """Escapes a path for use as an ffmpeg filtergraph argument.

    A filtergraph splits options on ':' and filters on ',', and libass' own
    option parser then unescapes '\\' and quotes, so both layers need escaping.
    """
    escaped = path.replace("\\", "\\\\").replace("'", r"\'")
    for char in (":", ",", "[", "]", ";"):
        escaped = escaped.replace(char, "\\" + char)
    return escaped


def build_render_command(
    media_path: str,
    ass_path: str,
    output_path: str,
    *,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    fps: int = DEFAULT_FPS,
    background: str = DEFAULT_BACKGROUND,
    fonts_dir: str | None = None,
    force_style: str | None = None,
) -> list[str]:
    width, height = resolution
    subtitles = f"subtitles=filename='{_escape_filter_path(ass_path)}'"
    if fonts_dir:
        subtitles += f":fontsdir='{_escape_filter_path(fonts_dir)}'"
    if force_style:
        # Overrides [V4+ Styles] fields at render time, e.g. to supply a font
        # that can actually draw the script (the ASS default, Arial, has no
        # Hangul coverage).
        subtitles += f":force_style='{_escape_filter_path(force_style)}'"

    cmd = [tool("ffmpeg"), "-y", "-loglevel", "error", "-stats"]
    if has_video_stream(media_path):
        cmd += ["-i", media_path]
    else:
        # No video track: synthesise a background for the subtitles to sit on.
        cmd += [
            "-f", "lavfi",
            "-i", f"color=c={background}:s={width}x{height}:r={fps}",
            "-i", media_path,
            "-map", "0:v:0", "-map", "1:a:0",
        ]
    cmd += [
        "-vf", subtitles,
        "-shortest",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    return cmd


def render(
    media_path: str,
    ass_path: str,
    output_path: str,
    *,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    fps: int = DEFAULT_FPS,
    background: str = DEFAULT_BACKGROUND,
    fonts_dir: str | None = None,
    force_style: str | None = None,
) -> str:
    """Burns `ass_path` onto `media_path` and writes an mp4 to `output_path`."""
    for path, label in ((media_path, "media"), (ass_path, "subtitle")):
        if not os.path.isfile(path):
            raise MediaToolError(f"{label} file not found: {path}")

    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)

    run_tool(build_render_command(
        media_path, ass_path, output_path,
        resolution=resolution, fps=fps,
        background=background, fonts_dir=fonts_dir,
        force_style=force_style,
    ))
    return output_path


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m web.render",
        description="Burn an ASS subtitle file onto its source media as an mp4.",
    )
    parser.add_argument("media", help="source mp3 or mp4")
    parser.add_argument("ass", help="ASS subtitle file")
    parser.add_argument("output", help="destination mp4")
    parser.add_argument(
        "--resolution", default="x".join(str(n) for n in DEFAULT_RESOLUTION),
        help="background size for audio-only sources (default: the script's PlayRes)",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--background", default=DEFAULT_BACKGROUND)
    parser.add_argument("--fonts-dir", default=None, help="extra font directory for libass")
    parser.add_argument(
        "--force-style", default=None,
        help="ASS style overrides, e.g. \"FontName=Noto Sans KR\"",
    )
    args = parser.parse_args(argv)

    width, _, height = args.resolution.partition("x")
    out = render(
        args.media, args.ass, args.output,
        resolution=(int(width), int(height)),
        fps=args.fps,
        background=args.background,
        fonts_dir=args.fonts_dir,
        force_style=args.force_style,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
