"""The clip catalogue and its queue.

Each clip is a row in the Supabase `clips` table, one per YouTube video; the
player streams from YouTube itself. The web server only queues clips; the
worker (`worker.convert`) takes them in turn, checks them with YouTube and
converts them.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timedelta, timezone

from web import catalog
from web.db import SupabaseStore

ID_RE = re.compile(r"^[0-9a-f]{12}$")


def check_id(video_id: str) -> None:
    """KeyError for a malformed id, before it reaches a query or a path."""
    if not ID_RE.match(video_id):
        raise KeyError(video_id)


# Longest clip accepted, and clips converted per day across the service: each
# conversion spends a video review (Gemini, 20 a day per model) and roughly its
# own length on the STT server.
MAX_DURATION = 900
DAILY_LIMIT = 20
# The day the limit counts starts at midnight here.
TIMEZONE = timezone(timedelta(hours=9))

# In order; the page shows the current one while a clip is processing.
QUEUED = "queued"
PENDING_STATES = (QUEUED, "downloading", "preparing", "transcribing", "finishing")
READY = "ready"
ERROR = "error"

# What the viewer reads when a conversion fails for any reason other than a
# rejected link; the cause goes to the worker's log.
FAILED = "자막을 만들지 못했어요. 잠시 뒤 다시 요청해 주세요."


class Rejected(ValueError):
    """A link the catalogue does not take; the message is shown to the user."""


class QuotaExceeded(Exception):
    """Today's conversions are used up."""


def youtube_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


class Library:
    def __init__(
        self,
        store: SupabaseStore | None = None,
        max_duration: float | None = None,
        daily_limit: int | None = None,
    ):
        self.store = store or SupabaseStore()
        self.max_duration = max_duration or MAX_DURATION
        self.daily_limit = daily_limit or DAILY_LIMIT

    # --- records ---------------------------------------------------------

    def get(self, video_id: str) -> dict:
        check_id(video_id)
        meta = self.store.get(video_id)
        if meta is None:
            raise KeyError(video_id)
        return meta

    def list(self, genre: str | None = None, query: str | None = None) -> list[dict]:
        """Ready clips, newest first."""
        return self.store.list(genre=genre, query=query)

    def claim(self) -> dict | None:
        """The oldest queued clip, taken for the calling worker alone."""
        return self.store.claim()

    def heartbeat(self, video_id: str) -> None:
        """Marks a claimed clip as still being worked on."""
        self.store.heartbeat(video_id)

    def update(self, video_id: str, **fields) -> None:
        self.store.update(video_id, **fields)

    def delete(self, video_id: str) -> None:
        self.get(video_id)
        self.store.delete(video_id)

    def quota(self) -> dict:
        return {
            "used": self.store.count_submitted_since(self._day_start()),
            "limit": self.daily_limit,
            "max_duration": self.max_duration,
        }

    def _day_start(self) -> float:
        now = datetime.now(TIMEZONE)
        return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    # --- submission ------------------------------------------------------

    def submit(self, url: str) -> tuple[str, bool]:
        """Queues `url` for the worker; returns its clip id and whether it was
        already in the catalogue (which spends nothing from the daily limit)."""
        youtube_id = catalog.youtube_id(url)
        if not youtube_id:
            raise Rejected("YouTube 링크만 자막을 만들 수 있어요.")

        result = self.store.submit(
            uuid.uuid4().hex[:12], youtube_id, time.time(), self._day_start(), self.daily_limit,
        )
        if result["over_quota"]:
            raise QuotaExceeded(f"오늘 만들 수 있는 자막 {self.daily_limit}건을 모두 썼어요. 내일 다시 요청해 주세요.")
        return result["id"], result["existing"]

    def transcript(self, video_id: str):
        from hearing_to_seeing.schema import Transcript

        self.get(video_id)
        data = self.store.get_transcript(video_id)
        if data is None:
            raise KeyError(video_id)
        return Transcript.from_dict(data)
