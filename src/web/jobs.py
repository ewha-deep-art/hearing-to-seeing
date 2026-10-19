"""The clip catalogue, and the background work that fills it.

Each clip is a row in the Supabase `clips` table, one per YouTube video; the
player streams from YouTube itself. A clip's directory under the library root
holds its downloaded source and audio only while it is processed.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from hearing_to_seeing.render import MediaToolError
from hearing_to_seeing import media
from web import catalog
from web.db import SupabaseStore

DEFAULT_LIBRARY_DIR = os.path.join("data", "output", "library")

ID_RE = re.compile(r"^[0-9a-f]{12}$")

# The only language the web app transcribes; the CLI still takes any.
LANGUAGE = "ko"

# Longest clip accepted, and clips converted per day across the service: each
# conversion spends a video review (Gemini, 20 a day per model) and roughly its
# own length on the STT server.
MAX_DURATION = 900
DAILY_LIMIT = 20
# The day the limit counts starts at midnight here.
TIMEZONE = timezone(timedelta(hours=9))

# Files inside a clip's working directory.
AUDIO = "audio.wav"

# In order; the page shows the current one while a clip is processing.
PENDING_STATES = ("queued", "downloading", "preparing", "transcribing", "finishing")
READY = "ready"
ERROR = "error"


class Rejected(ValueError):
    """A link the catalogue does not take; the message is shown to the user."""


class QuotaExceeded(Exception):
    """Today's conversions are used up."""


def youtube_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


class Library:
    def __init__(
        self,
        root: str | None = None,
        executor: Executor | None = None,
        store: SupabaseStore | None = None,
        max_duration: float | None = None,
        daily_limit: int | None = None,
    ):
        self.root = os.path.abspath(root or DEFAULT_LIBRARY_DIR)
        os.makedirs(self.root, exist_ok=True)
        self.store = store or SupabaseStore()
        # One at a time: STT runs on a remote server at roughly real-time
        # speed, so parallel jobs would only compete for it.
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
        self.max_duration = max_duration or MAX_DURATION
        self.daily_limit = daily_limit or DAILY_LIMIT
        # Holds the duplicate check, the count and the insert together.
        self._lock = threading.Lock()

    # --- records ---------------------------------------------------------

    def path(self, video_id: str, name: str = "") -> str:
        if not ID_RE.match(video_id):
            raise KeyError(video_id)
        return os.path.join(self.root, video_id, name)

    def get(self, video_id: str) -> dict:
        self.path(video_id)  # rejects malformed ids before they reach the query
        meta = self.store.get(video_id)
        if meta is None:
            raise KeyError(video_id)
        return meta

    def list(self, genre: str | None = None, query: str | None = None) -> list[dict]:
        """Ready clips, newest first."""
        return self.store.list(genre=genre, query=query)

    def pending(self) -> list[dict]:
        return self.store.with_status(PENDING_STATES)

    def update(self, video_id: str, **fields) -> None:
        self.store.update(video_id, **fields)

    def delete(self, video_id: str) -> None:
        self.get(video_id)
        self.store.delete(video_id)
        shutil.rmtree(self.path(video_id), ignore_errors=True)

    def quota(self) -> dict:
        return {
            "used": self.store.count_submitted_since(self._day_start()),
            "limit": self.daily_limit,
            "max_duration": self.max_duration,
        }

    def _day_start(self) -> float:
        now = datetime.now(TIMEZONE)
        return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def recover(self) -> None:
        """Marks work a previous server process left unfinished as failed."""
        for video in self.pending():
            self.update(video["id"], status=ERROR, error="서버 재시작으로 처리가 중단되었습니다.")

    # --- submission ------------------------------------------------------

    def submit(self, url: str) -> tuple[str, bool]:
        """Queues `url`; returns its clip id and whether it was already in the
        catalogue (which spends nothing from the daily limit)."""
        youtube_id = catalog.youtube_id(url)
        if not youtube_id:
            raise Rejected("YouTube 링크만 자막을 만들 수 있어요.")
        existing = self.store.find_by_youtube_id(youtube_id)
        if existing and existing["status"] != ERROR:  # skips the yt-dlp lookup
            return existing["id"], True
        try:
            info = media.youtube_info(url)
        except Exception as exc:
            print(f"… 영상 정보 실패 ({url}): {exc}", file=sys.stderr, flush=True)
            raise Rejected("영상을 찾지 못했어요. YouTube 링크가 맞는지 확인해 주세요.") from exc
        self._check(info)

        with self._lock:
            existing = self.store.find_by_youtube_id(info["id"])
            if existing and existing["status"] != ERROR:
                return existing["id"], True
            if self.store.count_submitted_since(self._day_start()) >= self.daily_limit:
                raise QuotaExceeded(f"오늘 만들 수 있는 자막 {self.daily_limit}건을 모두 썼어요. 내일 다시 요청해 주세요.")

            now = time.time()
            fields = {
                "title": info.get("title"),
                "channel": info.get("channel"),
                "duration": info.get("duration"),
                "status": "queued",
                "error": None,
                "timings": {},
                "submitted_at": now,
            }
            if existing:  # a failed clip goes round again under the same id
                video_id = existing["id"]
                self.store.update(video_id, **fields)
            else:
                video_id = uuid.uuid4().hex[:12]
                self.store.insert({"id": video_id, "youtube_id": info["id"], "created_at": now, **fields})

        self.executor.submit(self._process, video_id, info)
        return video_id, False

    def _check(self, info: dict) -> None:
        if info.get("live_status") in ("is_live", "is_upcoming"):
            raise Rejected("라이브 방송은 자막을 만들 수 없어요.")
        duration = info.get("duration")
        if not duration or duration > self.max_duration:
            raise Rejected(f"{int(self.max_duration // 60)}분 이하 영상만 자막을 만들 수 있어요.")
        if info.get("playable_in_embed") is False:
            raise Rejected("영상 제공자가 다른 사이트에서는 재생할 수 없도록 설정한 영상이라 자막을 만들 수 없어요.")

    # --- processing ------------------------------------------------------

    def _process(self, video_id: str, info: dict) -> None:
        os.makedirs(self.path(video_id), exist_ok=True)
        try:
            self._run_steps(video_id, info)
        except Exception as exc:  # recorded and logged, not lost in a thread
            error = str(exc) or type(exc).__name__
            print(f"… [{video_id}] 처리 실패: {error}", file=sys.stderr, flush=True)
            self.update(video_id, status=ERROR, error=error)
        finally:
            shutil.rmtree(self.path(video_id), ignore_errors=True)

    @contextmanager
    def _timed_step(self, video_id: str, name: str, timings: dict[str, float]):
        """Marks `status` as `name` and records its duration into `timings`."""
        self.update(video_id, status=name)
        start = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = time.perf_counter() - start
            self.update(video_id, timings=dict(timings))

    def _run_steps(self, video_id: str, info: dict) -> None:
        from hearing_to_seeing import pipeline

        meta = self.get(video_id)
        path = lambda name: self.path(video_id, name)  # noqa: E731
        timings: dict[str, float] = {}

        # The video only matters to the review, which waits for STT anyway:
        # it downloads alongside, and STT starts as soon as the audio is in.
        with ThreadPoolExecutor(max_workers=1) as fetch:
            video = fetch.submit(self._review_video, youtube_url(meta["youtube_id"]), path(""))
            with self._timed_step(video_id, "downloading", timings):
                audio_path = media.download_audio(youtube_url(meta["youtube_id"]), path(""))

            with self._timed_step(video_id, "preparing", timings):
                probed = media.probe(audio_path)
                if probed["audio"] is None:
                    raise MediaToolError("오디오 트랙이 없는 파일입니다.")
                media.extract_wav(audio_path, path(AUDIO))
                self.update(video_id, duration=probed["duration"], genre=catalog.genre_of(info))

            with self._timed_step(video_id, "transcribing", timings):
                # The title lets the speaker step look up the work's characters
                # (when GEMINI_API_KEY is set) — alongside STT, inside pipeline.run;
                # the source video lets the review see who is speaking.
                transcript = pipeline.run(
                    path(AUDIO), None, LANGUAGE, title=meta["title"], media_path=video,
                )

        with self._timed_step(video_id, "finishing", timings):
            # The ASS files are generated from the transcript when requested.
            data = transcript.to_dict()
            self.update(video_id, transcript=data,
                        search_text=catalog.search_text(meta["title"], meta["channel"], data))

        self.update(video_id, status=READY, error=None)

    @staticmethod
    def _review_video(url: str, dest_dir: str) -> str | None:
        """The video for the review; None (it listens to the audio) when it
        cannot be had."""
        try:
            return media.download_youtube(url, dest_dir)
        except Exception as exc:
            print(f"… 검토용 영상 받기 실패, 음성으로 검토: {exc}", file=sys.stderr, flush=True)
            return None

    def transcript(self, video_id: str):
        from hearing_to_seeing.schema import Transcript

        self.get(video_id)
        data = self.store.get_transcript(video_id)
        if data is None:
            raise KeyError(video_id)
        return Transcript.from_dict(data)
