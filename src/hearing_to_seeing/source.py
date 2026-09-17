"""Collecting what the user knows about the media, alongside the media itself.

The pipeline reads a local file, but the file alone does not say which work it
holds. A later speaker-colour step cannot look up who appears in a video
without knowing which video it is, and that knowledge only exists in the
user's head at the point of invocation — so it is taken as input here and
travels on the transcript.

Nothing in this module touches the network: the media file is supplied by the
user, and the URL is recorded as a reference, not fetched.
"""

import os
import re
from dataclasses import dataclass

from hearing_to_seeing.schema import MediaInfo

_URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


class SourceError(RuntimeError):
    pass


@dataclass
class ResolvedSource:
    media_path: str
    info: MediaInfo


def is_url(source: str) -> bool:
    return bool(_URL_RE.match(source.strip()))


def resolve(
    media_path: str,
    *,
    title: str | None = None,
    url: str | None = None,
) -> ResolvedSource:
    """Checks the media file exists and packages the metadata that came with it.

    The filename is deliberately not used as a fallback title. "sample.mp4" or
    "video_final2.mp4" would hand the lookup step something confidently wrong
    to search for, which is worse than it knowing the title is unset.
    """
    media_path = media_path.strip()
    if not os.path.isfile(media_path):
        raise SourceError(f"media file not found: {media_path}")

    if url is not None:
        url = url.strip()
        if not is_url(url):
            # Caught here rather than stored, since a bare "youtu.be/abc" is a
            # typo the user can fix now and a bad reference nobody notices later.
            raise SourceError(
                f"URL 형식이 아닙니다 (https:// 로 시작해야 합니다): {url}"
            )

    return ResolvedSource(
        media_path=media_path,
        info=MediaInfo(title=title, source_url=url),
    )
