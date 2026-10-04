"""Getting media in and out of the shapes the pipeline and the browser need.

The pipeline reads WAV only, and a browser plays H.264/AAC mp4 reliably, so each
source is split into both: `audio.wav` for STT and volume, `media.mp4` for the
player.
"""

import json
import os

from hearing_to_seeing.render import run_tool, tool

# Codecs every mainstream browser decodes inside an mp4 without re-encoding.
_PLAYABLE_VIDEO = {"h264"}
_PLAYABLE_AUDIO = {"aac", "mp3"}

# A player needs a frame to show; audio-only sources get a black 16:9 still.
AUDIO_ONLY_SIZE = (1280, 720)


def download_youtube(url: str, dest_dir: str) -> tuple[str, str | None]:
    """Downloads `url` as an mp4 into `dest_dir`; returns `(path, title)`.

    Video is requested alongside audio because the player shows it, and H.264
    is preferred so the file usually plays without a transcode.
    """
    from yt_dlp import YoutubeDL

    options = {
        "format": (
            "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]"
            "/b[ext=mp4][height<=1080]/bv*[height<=1080]+ba/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(dest_dir, "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Error messages end up in the web UI; keep terminal colour codes out.
        "color": {"stdout": "no_color", "stderr": "no_color"},
        "ffmpeg_location": tool("ffmpeg"),
        # YouTube hides formats behind a JS challenge; without a runtime many
        # videos fail as "not available". yt-dlp only tries deno by default,
        # so allow every runtime it supports (the solver is yt-dlp-ejs).
        "js_runtimes": {"deno": {}, "node": {}, "bun": {}, "quickjs": {}},
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        path = info.get("requested_downloads", [{}])[0].get("filepath") or ydl.prepare_filename(info)
    return path, info.get("title")


def probe(media_path: str) -> dict:
    """Duration and first video/audio codec names, as reported by ffprobe."""
    out = run_tool([
        tool("ffprobe"), "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,codec_name:stream_disposition=attached_pic",
        "-of", "json",
        media_path,
    ])
    data = json.loads(out or b"{}")
    codecs: dict[str, str] = {}
    for stream in data.get("streams", []):
        # Cover art embedded in an mp3 shows up as a one-frame video stream.
        if stream.get("disposition", {}).get("attached_pic"):
            continue
        codecs.setdefault(stream.get("codec_type"), stream.get("codec_name"))
    duration = data.get("format", {}).get("duration")
    return {
        "duration": float(duration) if duration else None,
        "video": codecs.get("video"),
        "audio": codecs.get("audio"),
    }


def extract_wav(media_path: str, wav_path: str) -> str:
    """Mono 16 kHz PCM — what WhisperX resamples to anyway, at a fraction of the upload."""
    run_tool([
        tool("ffmpeg"), "-y", "-loglevel", "error",
        "-i", media_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        wav_path,
    ])
    return wav_path


def prepare_playback(media_path: str, output_path: str) -> str:
    """Writes a browser-playable, seekable mp4 of `media_path` to `output_path`.

    Streams already in browser codecs are copied as-is; anything else is
    re-encoded. `+faststart` puts the index up front so playback and seeking
    start before the whole file has loaded.
    """
    info = probe(media_path)
    cmd = [tool("ffmpeg"), "-y", "-loglevel", "error"]

    if info["video"] is None:
        width, height = AUDIO_ONLY_SIZE
        cmd += [
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=1",
            "-i", media_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-shortest",
        ]
        cmd += _audio_args(info["audio"])
    else:
        cmd += ["-i", media_path, "-map", "0:v:0", "-map", "0:a:0?"]
        if info["video"] in _PLAYABLE_VIDEO:
            cmd += ["-c:v", "copy"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p"]
        cmd += _audio_args(info["audio"])

    cmd += ["-movflags", "+faststart", output_path]
    run_tool(cmd)
    return output_path


def _audio_args(codec: str | None) -> list[str]:
    if codec in _PLAYABLE_AUDIO:
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", "192k"]


def extract_thumbnail(media_path: str, output_path: str, duration: float | None) -> str:
    """One frame a little way in, past the black lead-in most videos open on."""
    at = min(1.0, duration / 2) if duration else 0.0
    run_tool([
        tool("ffmpeg"), "-y", "-loglevel", "error",
        "-ss", f"{at:.2f}", "-i", media_path,
        "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4",
        output_path,
    ])
    return output_path
