"""The worker: converts the clips the web server queues, one at a time.

Run with `docker compose up -d` on a machine with a residential
connection (YouTube turns away most data-centre addresses). A clip's directory
under the library root holds its downloaded source and audio only while it is
processed.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from hearing_to_seeing.render import MediaToolError
from hearing_to_seeing import media
from web import catalog
from web.jobs import ERROR, FAILED, READY, Library, Rejected, check_id, youtube_url

DEFAULT_LIBRARY_DIR = os.path.join("data", "output", "library")

# The only language the web app transcribes; the CLI still takes any.
LANGUAGE = "ko"

# How often an idle worker looks for queued clips.
POLL_SECONDS = 5.0

# A working worker marks its clip this often; `claim_clip` fails a clip left
# unmarked for two minutes.
HEARTBEAT_SECONDS = 30.0

# Files inside a clip's working directory.
AUDIO = "audio.wav"


class Worker:
    def __init__(self, library: Library, root: str | None = None):
        self.library = library
        self.root = os.path.abspath(root or DEFAULT_LIBRARY_DIR)

    def path(self, video_id: str, name: str = "") -> str:
        check_id(video_id)
        return os.path.join(self.root, video_id, name)

    def work(self, interval: float = POLL_SECONDS) -> None:
        """Converts queued clips one at a time, oldest first, forever."""
        while True:
            try:
                busy = self.work_once()
            except Exception:  # e.g. Supabase unreachable: try again later
                traceback.print_exc()
                busy = False
            if not busy:
                time.sleep(interval)

    def work_once(self) -> bool:
        """Converts the oldest queued clip; False when none is waiting."""
        clip = self.library.claim()
        if clip is None:
            return False
        with self._heartbeat(clip["id"]):
            self._process(clip)
        return True

    @contextmanager
    def _heartbeat(self, video_id: str):
        """Marks the clip as worked on every HEARTBEAT_SECONDS until the block ends."""
        done = threading.Event()

        def beat():
            while not done.wait(HEARTBEAT_SECONDS):
                try:
                    self.library.heartbeat(video_id)
                except Exception:  # a missed beat only matters after two minutes
                    traceback.print_exc()

        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()

    def _process(self, clip: dict) -> None:
        video_id = clip["id"]
        os.makedirs(self.path(video_id), exist_ok=True)
        try:
            url = youtube_url(clip["youtube_id"])
            self._run_steps(video_id, url, self._inspect(video_id, url))
        except Rejected as exc:
            # Turned away before any work: today's limit gets it back.
            print(f"… [{video_id}] 거절: {exc}", file=sys.stderr, flush=True)
            self.library.update(video_id, status=ERROR, error=str(exc), submitted_at=0)
        except Exception:  # the details go to the worker's log, not the viewer
            print(f"… [{video_id}] 처리 실패", file=sys.stderr, flush=True)
            traceback.print_exc()
            self.library.update(video_id, status=ERROR, error=FAILED)
        finally:
            shutil.rmtree(self.path(video_id), ignore_errors=True)

    def _inspect(self, video_id: str, url: str) -> dict:
        """What YouTube says about a queued clip, recorded on it; Rejected when
        the clip is not one the catalogue takes."""
        try:
            info = media.youtube_info(url)
        except Exception as exc:
            print(f"… 영상 정보 실패 ({url}): {exc}", file=sys.stderr, flush=True)
            raise Rejected("영상을 찾지 못했어요. YouTube 링크가 맞는지 확인해 주세요.") from exc
        self._check(info)
        self.library.update(video_id, title=info.get("title"), channel=info.get("channel"), duration=info.get("duration"))
        return info

    def _check(self, info: dict) -> None:
        max_duration = self.library.max_duration
        if info.get("live_status") in ("is_live", "is_upcoming"):
            raise Rejected("라이브 방송은 자막을 만들 수 없어요.")
        duration = info.get("duration")
        if not duration or duration > max_duration:
            raise Rejected(f"{int(max_duration // 60)}분 이하 영상만 자막을 만들 수 있어요.")
        if info.get("playable_in_embed") is False:
            raise Rejected("영상 제공자가 유튜브 외의 사이트에서는 재생할 수 없도록 설정한 영상이라 자막을 만들 수 없어요.")

    @contextmanager
    def _timed_step(self, video_id: str, name: str, timings: dict[str, float]):
        """Marks `status` as `name` and records its duration into `timings`."""
        self.library.update(video_id, status=name)
        start = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = time.perf_counter() - start
            self.library.update(video_id, timings=dict(timings))

    def _run_steps(self, video_id: str, url: str, info: dict) -> None:
        from hearing_to_seeing import pipeline

        title, channel = info.get("title"), info.get("channel")
        path = lambda name: self.path(video_id, name)  # noqa: E731
        timings: dict[str, float] = {}

        # The video only matters to the review, which waits for STT anyway, and
        # the genre (a Gemini call) only to the catalogue: both run alongside,
        # and STT starts as soon as the audio is in.
        with ThreadPoolExecutor(max_workers=2) as fetch:
            video = fetch.submit(self._review_video, url, path(""))
            genre = fetch.submit(catalog.genre_of, info)
            with self._timed_step(video_id, "downloading", timings):
                audio_path = media.download_audio(url, path(""))

            with self._timed_step(video_id, "preparing", timings):
                probed = media.probe(audio_path)
                if probed["audio"] is None:
                    raise MediaToolError("오디오 트랙이 없는 파일입니다.")
                media.extract_wav(audio_path, path(AUDIO))
                self.library.update(video_id, duration=probed["duration"])

            with self._timed_step(video_id, "transcribing", timings):
                # The title lets the speaker step look up the work's characters
                # (when GEMINI_API_KEY is set) — alongside STT, inside pipeline.run;
                # the source video lets the review see who is speaking.
                transcript = pipeline.run(
                    path(AUDIO), None, LANGUAGE, title=title, media_path=video,
                )

        with self._timed_step(video_id, "finishing", timings):
            # The ASS files are generated from the transcript when requested.
            data = transcript.to_dict()
            self.library.update(video_id, transcript=data, genre=genre.result(),
                                search_text=catalog.search_text(title, channel, data))

        self.library.update(video_id, status=READY, error=None)

    @staticmethod
    def _review_video(url: str, dest_dir: str) -> str | None:
        """The video for the review; None (it listens to the audio) when it
        cannot be had."""
        try:
            return media.download_youtube(url, dest_dir)
        except Exception as exc:
            print(f"… 검토용 영상 받기 실패, 음성으로 검토: {exc}", file=sys.stderr, flush=True)
            return None


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="hearing-to-seeing-worker",
        description="Convert the clips queued in the Hearing to Seeing catalogue.",
    )
    parser.add_argument("--interval", type=float, default=POLL_SECONDS,
                        help="seconds between looks at an empty queue")
    args = parser.parse_args(argv)

    Worker(Library()).work(args.interval)


if __name__ == "__main__":
    main()
