"""Getting YouTube media into the shape the pipeline needs.

The pipeline reads WAV only: each source is downloaded, split into `audio.wav`
for STT and volume, and kept as video only while the review watches it.
"""

import json
import os

from hearing_to_seeing.render import run_tool, tool


def _ydl_options(dest_dir: str | None = None) -> dict:
    options = {
        "format": (
            "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]"
            "/b[ext=mp4][height<=1080]/bv*[height<=1080]+ba/b"
        ),
        "merge_output_format": "mp4",
        "noplaylist": True,
        # Some videos (e.g. channel uploads restricted on the web player) come
        # back "not available" from the default clients; the android client
        # still serves them.
        "extractor_args": {"youtube": {"player_client": ["default", "android"]}},
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
    if dest_dir:
        options["outtmpl"] = os.path.join(dest_dir, "source.%(ext)s")
    return options


def youtube_info(url: str) -> dict:
    """What YouTube says about `url`, without downloading it."""
    from yt_dlp import YoutubeDL

    with YoutubeDL(_ydl_options()) as ydl:
        info = ydl.extract_info(url, download=False)
    keys = ("id", "title", "channel", "description", "duration", "categories", "tags",
            "playable_in_embed", "live_status")
    return {key: info.get(key) for key in keys}


def download_youtube(url: str, dest_dir: str) -> str:
    """Downloads `url` as an mp4 into `dest_dir`; returns its path.

    Video is requested alongside audio because the review watches who is on
    screen, and H.264 is preferred so the review upload needs no transcode.
    """
    from yt_dlp import YoutubeDL

    with YoutubeDL(_ydl_options(dest_dir)) as ydl:
        info = ydl.extract_info(url, download=True)
        return info.get("requested_downloads", [{}])[0].get("filepath") or ydl.prepare_filename(info)


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
