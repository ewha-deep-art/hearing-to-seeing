"""Turning whatever the user passed in into a local media file plus its metadata.

The pipeline needs a file on disk, but a user working from YouTube has a link.
This module accepts either and returns the same thing to both callers: a local
path, and a `MediaInfo` describing where it came from.

The metadata is the point of the download, not a side effect of it. Title,
channel and description are what a later speaker-colour step needs in order to
find out who appears in the video, and they are only available at fetch time —
so they are captured here and travel on the transcript instead of being thrown
away once the media has been written to disk.

Requires `yt-dlp` for URL inputs; local paths need nothing.
"""

import glob
import os
import re
from dataclasses import dataclass

from hearing_to_seeing.schema import MAX_DESCRIPTION_CHARS, MediaInfo

# Downloads land under data/input/, which .gitignore already excludes, so a
# fetched video is not mistaken for a committed fixture.
DEFAULT_CACHE_DIR = os.path.join("data", "input", "downloads")

_URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


class SourceError(RuntimeError):
    pass


@dataclass
class ResolvedSource:
    media_path: str
    info: MediaInfo


def is_url(source: str) -> bool:
    return bool(_URL_RE.match(source.strip()))


def _cached_download(cache_dir: str, video_id: str) -> str | None:
    """Returns an already-downloaded file for `video_id`, if there is one.

    Re-running the pipeline on the same link is the normal case while tuning
    the later steps, and re-downloading a film each time is the slowest thing
    in the run. Naming files by the source's own id makes the hit cheap to
    detect without trusting yt-dlp to report a skipped download.
    """
    matches = sorted(glob.glob(os.path.join(cache_dir, f"{video_id}.*")))
    # A merged download can leave the separate video/audio parts behind.
    matches = [m for m in matches if not m.endswith((".part", ".ytdl"))]
    return matches[0] if matches else None


def _resolve_local(path: str, title: str | None) -> ResolvedSource:
    if not os.path.isfile(path):
        raise SourceError(f"media file not found: {path}")
    # The filename is deliberately not used as a fallback title. "sample.mp4"
    # or "video_final2.mp4" would give the lookup step something confidently
    # wrong to search for, which is worse than knowing the title is unset.
    return ResolvedSource(media_path=path, info=MediaInfo(title=title))


def _resolve_url(
    url: str,
    cache_dir: str,
    title: str | None,
    audio_only: bool,
) -> ResolvedSource:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SourceError(
            "URL 입력에는 yt-dlp가 필요합니다. `uv sync` 또는 `uv add yt-dlp`로 설치하세요."
        ) from exc

    os.makedirs(cache_dir, exist_ok=True)
    options = {
        "outtmpl": os.path.join(cache_dir, "%(id)s.%(ext)s"),
        # A watch URL carrying `&list=` would otherwise pull the whole playlist.
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if audio_only:
        options["format"] = "bestaudio/best"
    else:
        # Video is the default because the preview renderer needs it, and so
        # will any later step that looks at what is on screen.
        options["format"] = "bestvideo*+bestaudio/best"
        options["merge_output_format"] = "mp4"

    try:
        with YoutubeDL(options) as ydl:
            # Metadata first, without downloading: it is what tells us the id,
            # and therefore whether the file is already on disk.
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise SourceError(f"영상 정보를 가져올 수 없습니다: {url}")
            if info.get("_type") == "playlist":
                entries = [e for e in (info.get("entries") or []) if e]
                if not entries:
                    raise SourceError(f"재생 목록에서 영상을 찾을 수 없습니다: {url}")
                info = entries[0]

            media_path = _cached_download(cache_dir, info["id"])
            if media_path is None:
                downloaded = ydl.extract_info(url, download=True)
                media_path = _downloaded_path(ydl, downloaded) or _cached_download(
                    cache_dir, info["id"]
                )
                info = downloaded or info
    except SourceError:
        raise
    except Exception as exc:  # yt-dlp raises its own hierarchy of errors
        raise SourceError(f"다운로드 실패: {url}\n{exc}") from exc

    if media_path is None or not os.path.isfile(media_path):
        raise SourceError(f"다운로드된 파일을 찾을 수 없습니다: {url}")

    description = info.get("description") or None
    if description:
        description = description[:MAX_DESCRIPTION_CHARS]

    return ResolvedSource(
        media_path=media_path,
        info=MediaInfo(
            title=title or info.get("title"),
            source_url=info.get("webpage_url") or url,
            uploader=info.get("uploader") or info.get("channel"),
            description=description,
        ),
    )


def _downloaded_path(ydl, info: dict | None) -> str | None:
    """Pulls the written file's path out of a yt-dlp result."""
    if not info:
        return None
    for entry in info.get("requested_downloads") or []:
        path = entry.get("filepath")
        if path and os.path.isfile(path):
            return path
    guess = ydl.prepare_filename(info)
    return guess if guess and os.path.isfile(guess) else None


def resolve(
    source: str,
    *,
    title: str | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    audio_only: bool = False,
) -> ResolvedSource:
    """Resolves a local path or a URL into a local file and its metadata.

    `title` always wins over whatever the source reports, so a link whose
    uploaded title is decorated ("[4K] 기생충 명장면 모음") can be corrected
    without editing the download.
    """
    source = source.strip()
    if is_url(source):
        return _resolve_url(source, cache_dir, title, audio_only)
    return _resolve_local(source, title)
