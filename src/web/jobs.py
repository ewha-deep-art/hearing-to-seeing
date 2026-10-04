"""The video library on disk, and the background work that fills it.

Each video lives in its own directory under the library root; `meta.json`
there is the single record of its state, so the library survives a restart
and listing it is a directory scan.
"""

import json
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager

from hearing_to_seeing.converter.ass import generate_plain_ass
from hearing_to_seeing.render import MediaToolError
from hearing_to_seeing import media

DEFAULT_LIBRARY_DIR = os.path.join("data", "output", "library")

ID_RE = re.compile(r"^[0-9a-f]{12}$")

# The only language the web app transcribes; the CLI still takes any.
LANGUAGE = "ko"

# Files inside a video's directory.
AUDIO = "audio.wav"
PLAYBACK = "media.mp4"
THUMBNAIL = "thumb.jpg"
ASS = "subtitles.ass"
PLAIN_ASS = "subtitles.plain.ass"
TRANSCRIPT = "transcript.json"
META = "meta.json"

# In order; the page shows the current one while a video is processing.
PENDING_STATES = ("queued", "downloading", "preparing", "transcribing", "finishing")
READY = "ready"
ERROR = "error"


class Library:
    def __init__(self, root: str | None = None, executor: Executor | None = None):
        self.root = os.path.abspath(
            root or os.environ.get("H2S_LIBRARY_DIR") or DEFAULT_LIBRARY_DIR
        )
        os.makedirs(self.root, exist_ok=True)
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
        try:
            with open(self.path(video_id, META), encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise KeyError(video_id) from None

    def list(self) -> list[dict]:
        videos = []
        for name in os.listdir(self.root):
            if ID_RE.match(name):
                try:
                    videos.append(self.get(name))
                except (KeyError, json.JSONDecodeError):
                    continue
        return sorted(videos, key=lambda v: v["created_at"], reverse=True)

    def update(self, video_id: str, **fields) -> dict:
        with self._lock:
            meta = self.get(video_id)
            meta.update(fields)
            tmp = self.path(video_id, META + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path(video_id, META))
            return meta

    def delete(self, video_id: str) -> None:
        self.get(video_id)
        shutil.rmtree(self.path(video_id))

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
        with open(self.path(video_id, META), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
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
            # (when GEMINI_API_KEY is set) — alongside STT, inside pipeline.run.
            transcript = pipeline.run(
                path(AUDIO), path(ASS), meta["language"], path(TRANSCRIPT),
                title=self.get(video_id)["title"],
            )

        with self._timed_step(video_id, "finishing", timings):
            with open(path(PLAIN_ASS), "w", encoding="utf-8") as f:
                f.write(generate_plain_ass(transcript))

        self.update(video_id, status=READY, error=None)

    def transcript(self, video_id: str):
        from hearing_to_seeing.schema import Transcript

        with open(self.path(video_id, TRANSCRIPT), encoding="utf-8") as f:
            return Transcript.from_dict(json.load(f))
