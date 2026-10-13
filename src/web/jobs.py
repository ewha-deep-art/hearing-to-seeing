"""The video library, and the background work that fills it.

Each video's record (its state and, once ready, its transcript) is a row in
the Supabase `videos` table, so the library survives a restart. Its media —
the playback video and thumbnail — live in its own directory under the
library root.
"""

import glob
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager

from hearing_to_seeing.render import MediaToolError
from hearing_to_seeing import media
from web.db import SupabaseStore

DEFAULT_LIBRARY_DIR = os.path.join("data", "output", "library")

ID_RE = re.compile(r"^[0-9a-f]{12}$")

# The only language the web app transcribes; the CLI still takes any.
LANGUAGE = "ko"

# Files inside a video's directory. The audio and the downloaded source are
# only needed while the video is processed.
AUDIO = "audio.wav"
SOURCE = "source.*"
PLAYBACK = "media.mp4"
THUMBNAIL = "thumb.jpg"

# In order; the page shows the current one while a video is processing.
PENDING_STATES = ("queued", "downloading", "preparing", "transcribing", "finishing")
READY = "ready"
ERROR = "error"


class Library:
    def __init__(
        self,
        root: str | None = None,
        executor: Executor | None = None,
        store: SupabaseStore | None = None,
    ):
        self.root = os.path.abspath(
            root or os.environ.get("H2S_LIBRARY_DIR") or DEFAULT_LIBRARY_DIR
        )
        os.makedirs(self.root, exist_ok=True)
        self.store = store or SupabaseStore()
        # One at a time: STT runs on a remote server at roughly real-time
        # speed, so parallel jobs would only compete for it.
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
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

    def list(self) -> list[dict]:
        return self.store.list()

    def update(self, video_id: str, **fields) -> dict:
        with self._lock:
            meta = self.get(video_id)
            meta.update(fields)
            self.store.update_meta(video_id, meta)
            return meta

    def delete(self, video_id: str) -> None:
        self.get(video_id)
        self.store.delete(video_id)
        shutil.rmtree(self.path(video_id), ignore_errors=True)

    def _create(self, **fields) -> str:
        video_id = uuid.uuid4().hex[:12]
        os.makedirs(self.path(video_id))
        meta = {
            "id": video_id,
            "title": None,
            "source": "youtube",
            "source_url": None,
            "language": LANGUAGE,
            "status": "queued",
            "error": None,
            "duration": None,
            "created_at": time.time(),
            "timings": {},
            **fields,
        }
        self.store.insert(meta)
        return video_id

    def recover(self) -> None:
        """Marks work a previous server process left unfinished as failed."""
        for video in self.list():
            if video["status"] in PENDING_STATES:
                self.update(video["id"], status=ERROR, error="서버 재시작으로 처리가 중단되었습니다.")

    # --- submission ------------------------------------------------------

    def add_url(self, url: str, title: str | None) -> str:
        video_id = self._create(title=title, source_url=url)
        self.executor.submit(self._process, video_id)
        return video_id

    # --- processing ------------------------------------------------------

    def _process(self, video_id: str) -> None:
        try:
            self._run_steps(video_id)
        except Exception as exc:  # surfaced on the page, not lost in a thread
            self.update(video_id, status=ERROR, error=str(exc) or type(exc).__name__)
        finally:
            for file in [self.path(video_id, AUDIO), *glob.glob(self.path(video_id, SOURCE))]:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass

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

    def _run_steps(self, video_id: str) -> None:
        from hearing_to_seeing import pipeline

        meta = self.get(video_id)
        path = lambda name: self.path(video_id, name)  # noqa: E731
        timings: dict[str, float] = {}

        with self._timed_step(video_id, "downloading", timings):
            source_path, fetched_title = media.download_youtube(meta["source_url"], path(""))
            if not meta["title"]:
                self.update(video_id, title=fetched_title or meta["source_url"])

        with self._timed_step(video_id, "preparing", timings):
            info = media.probe(source_path)
            if info["audio"] is None:
                raise MediaToolError("오디오 트랙이 없는 파일입니다.")
            self.update(video_id, duration=info["duration"])
            media.extract_wav(source_path, path(AUDIO))
            media.prepare_playback(source_path, path(PLAYBACK))
            media.extract_thumbnail(path(PLAYBACK), path(THUMBNAIL), info["duration"])

        with self._timed_step(video_id, "transcribing", timings):
            # The title lets the speaker step look up the work's characters
            # (when GEMINI_API_KEY is set) — alongside STT, inside pipeline.run;
            # the source video lets the review see who is speaking.
            transcript = pipeline.run(
                path(AUDIO), None, meta["language"],
                title=self.get(video_id)["title"], media_path=source_path,
            )

        with self._timed_step(video_id, "finishing", timings):
            # The ASS files are generated from the transcript when requested.
            self.store.set_transcript(video_id, transcript.to_dict())

        self.update(video_id, status=READY, error=None)

    def transcript(self, video_id: str):
        from hearing_to_seeing.schema import Transcript

        self.get(video_id)
        data = self.store.get_transcript(video_id)
        if data is None:
            raise KeyError(video_id)
        return Transcript.from_dict(data)
