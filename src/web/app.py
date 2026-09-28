"""HTTP layer: the library and watch pages plus the JSON/media API behind them.

Run with `uv run hearing-to-seeing-web`.
"""

import os
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web import jobs, subtitles

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# kind → (file in the video directory, download filename suffix, media type)
_FILES = {
    "ass": (jobs.ASS, ".ass", "text/plain; charset=utf-8"),
    "plain": (jobs.PLAIN_ASS, ".plain.ass", "text/plain; charset=utf-8"),
    "json": (jobs.TRANSCRIPT, ".json", "application/json"),
}


def create_app(library: jobs.Library | None = None) -> FastAPI:
    library = library or jobs.Library()
    library.recover()

    app = FastAPI(title="Hearing to Seeing")
    app.state.library = library
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def _meta(video_id: str) -> dict:
        try:
            return library.get(video_id)
        except KeyError:
            raise HTTPException(404, "영상을 찾을 수 없습니다.") from None

    def _file(video_id: str, name: str) -> str:
        _meta(video_id)
        path = library.path(video_id, name)
        if not os.path.isfile(path):
            raise HTTPException(404, "파일이 아직 준비되지 않았습니다.")
        return path

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/watch/{video_id}", include_in_schema=False)
    def watch(video_id: str):
        _meta(video_id)
        return FileResponse(os.path.join(STATIC_DIR, "watch.html"))

    @app.get("/api/videos")
    def list_videos():
        return library.list()

    @app.post("/api/videos", status_code=201)
    def add_video(
        url: Annotated[str | None, Form()] = None,
        title: Annotated[str | None, Form()] = None,
    ):
        title = (title or "").strip() or None
        url = (url or "").strip() or None
        if not url:
            raise HTTPException(422, "링크를 입력하세요.")
        if not url.startswith(("http://", "https://")):
            raise HTTPException(422, "http:// 또는 https:// 로 시작하는 링크를 입력하세요.")
        return {"id": library.add_url(url, title)}

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str):
        meta = _meta(video_id)
        if meta["status"] == jobs.READY:
            transcript = library.transcript(video_id)
            meta = {
                **meta,
                "speakers": subtitles.speakers(transcript),
                "cues": subtitles.cues(transcript),
            }
        return meta

    @app.delete("/api/videos/{video_id}", status_code=204)
    def delete_video(video_id: str):
        meta = _meta(video_id)
        if meta["status"] in jobs.PENDING_STATES:
            raise HTTPException(409, "처리 중인 영상은 삭제할 수 없습니다.")
        library.delete(video_id)

    @app.get("/api/videos/{video_id}/media")
    def video_media(video_id: str):
        # FileResponse answers Range requests, which the player needs to seek.
        return FileResponse(_file(video_id, jobs.PLAYBACK), media_type="video/mp4")

    @app.get("/api/videos/{video_id}/thumb")
    def video_thumb(video_id: str):
        return FileResponse(_file(video_id, jobs.THUMBNAIL), media_type="image/jpeg")

    @app.get("/api/videos/{video_id}/files/{kind}")
    def video_file(video_id: str, kind: str, download: bool = False):
        if kind not in _FILES:
            raise HTTPException(404, "알 수 없는 파일 종류입니다.")
        name, suffix, media_type = _FILES[kind]
        path = _file(video_id, name)
        if not download:
            return FileResponse(path, media_type=media_type)
        title = _meta(video_id)["title"] or video_id
        return FileResponse(path, media_type=media_type, filename=title + suffix)

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="hearing-to-seeing-web",
        description="Serve the Hearing to Seeing web player.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
