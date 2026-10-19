"""Getting YouTube media into the shape the pipeline needs.

The pipeline reads WAV only: each source's audio is downloaded on its own and
turned into `audio.wav` for STT and volume; a small copy of the video, fetched
alongside, is kept only while the review watches it.
"""

import json
import os
import sys

from hearing_to_seeing.render import run_tool, tool


# The audio alone downloads in seconds, so STT can start before the video is in.
AUDIO_FORMAT = "ba[ext=m4a]/ba"
AUDIO_FALLBACK_FORMAT = "ba/b"
# The review sees the video at Gemini's low resolution (66 tokens a frame), so
# 360p loses nothing, and its upload shares the uplink with STT's: video-only
# DASH (`bv`, so the throttled progressive 360p stream is not picked) with the
# low-bitrate audio track. H.264 needs no transcode for the upload.
VIDEO_FORMAT = "bv[height<=360][vcodec^=avc1]+ba[ext=m4a][abr<=64]/bv[height<=360][vcodec^=avc1]+ba[ext=m4a]"
# When those streams are refused (403 now and then), anything up to 1080p.
VIDEO_FALLBACK_FORMAT = "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/b[ext=mp4][height<=1080]/bv*[height<=1080]+ba/b"


def _ydl_options(dest_dir: str | None = None, fmt: str = VIDEO_FALLBACK_FORMAT, name: str = "source") -> dict:
    options = {
        "format": fmt,
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
        options["outtmpl"] = os.path.join(dest_dir, f"{name}.%(ext)s")
    return options


def youtube_info(url: str) -> dict:
    """What YouTube says about `url`, without downloading it."""
    from yt_dlp import YoutubeDL

    with YoutubeDL(_ydl_options()) as ydl:
        info = ydl.extract_info(url, download=False)
    keys = ("id", "title", "channel", "description", "duration", "categories", "tags",
            "playable_in_embed", "live_status")
    return {key: info.get(key) for key in keys}


def _download(url: str, options: dict) -> str:
    from yt_dlp import YoutubeDL

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        return info.get("requested_downloads", [{}])[0].get("filepath") or ydl.prepare_filename(info)


def download_audio(url: str, dest_dir: str) -> str:
    """Downloads `url`'s audio track into `dest_dir`; returns its path.

    A refused stream (403 now and then) is asked for again, which fetches
    fresh stream links, then for anything with sound in it.
    """
    for fmt in (AUDIO_FORMAT, AUDIO_FORMAT, AUDIO_FALLBACK_FORMAT):
        try:
            return _download(url, _ydl_options(dest_dir, fmt, "audio-source"))
        except Exception as exc:  # yt_dlp.DownloadError
            error = exc
            print(f"… 오디오 받기 실패, 다시 시도: {exc}", file=sys.stderr, flush=True)
    raise error


def download_youtube(url: str, dest_dir: str) -> str:
    """Downloads `url` as a small mp4 for the review into `dest_dir`; returns its path.

    Video is requested alongside audio because the review watches who is on
    screen and listens to who speaks.
    """
    try:
        return _download(url, _ydl_options(dest_dir, VIDEO_FORMAT))
    except Exception as exc:  # yt_dlp.DownloadError, mostly a refused stream
        print(f"… 작은 영상 받기 실패, 다른 형식으로 재시도: {exc}", file=sys.stderr, flush=True)
        return _download(url, _ydl_options(dest_dir))


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
