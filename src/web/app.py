"""HTTP layer: the catalogue and watch pages plus the JSON API behind them.

Run with `docker compose --profile web up web`, alongside the `worker` service
(which converts the clips this queues). On Vercel, `index.py` serves it.
"""

import os
import secrets
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, Form, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from hearing_to_seeing.design.layout import FONT_FILE, FONTS_DIR
from web import jobs, subtitles
from web.db import GENRES

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_app(library: jobs.Library | None = None) -> FastAPI:
    library = library or jobs.Library()

    app = FastAPI(title="Hearing to Seeing")
    app.state.library = library
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def revalidate(request, call_next):
        # The browser checks (ETag) every page and file on each load, so a deploy
        # is never half-applied from cache — new HTML with last week's script.
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    def _meta(video_id: str) -> dict:
        try:
            return library.get(video_id)
        except KeyError:
            raise HTTPException(404, "영상을 찾을 수 없습니다.") from None

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/watch/{video_id}", include_in_schema=False)
    def watch(video_id: str):
        _meta(video_id)
        return FileResponse(os.path.join(STATIC_DIR, "watch.html"))

    @app.get("/api/videos")
    def list_videos(q: str | None = None, genre: str | None = None):
        if genre and genre not in GENRES:
            raise HTTPException(422, "알 수 없는 장르입니다.")
        return library.list(genre=genre, query=(q or "").strip() or None)

    @app.get("/api/queue")
    def queue():
        return library.quota()

    @app.post("/api/videos", status_code=201)
    def add_video(url: Annotated[str | None, Form()] = None):
        url = (url or "").strip() or None
        if not url:
            raise HTTPException(422, "YouTube 링크를 입력해 주세요.")
        if not url.startswith(("http://", "https://")):
            raise HTTPException(422, "https://로 시작하는 YouTube 링크를 붙여 넣어 주세요.")
        try:
            video_id, existing = library.submit(url)
        except jobs.Rejected as exc:
            raise HTTPException(422, str(exc)) from None
        except jobs.QuotaExceeded as exc:
            raise HTTPException(429, str(exc)) from None
        if existing:
            return JSONResponse({"id": video_id, "existing": True})
        return {"id": video_id}

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
    def delete_video(video_id: str, x_admin_token: Annotated[str | None, Header()] = None):
        # Only whoever holds ADMIN_TOKEN deletes; unset, nobody does.
        expected = os.environ.get("ADMIN_TOKEN")
        if not (expected and x_admin_token and secrets.compare_digest(x_admin_token, expected)):
            raise HTTPException(403, "삭제 권한이 없습니다.")
        meta = _meta(video_id)
        if meta["status"] in jobs.PENDING_STATES:
            raise HTTPException(409, "처리 중인 영상은 삭제할 수 없습니다.")
        library.delete(video_id)

    @app.get("/api/subtitle-font")
    def subtitle_font():
        # The one font the ASS styles name (design/layout.py), so the player
        # draws the subtitles exactly as the preview renderer does.
        return FileResponse(os.path.join(FONTS_DIR, FONT_FILE), media_type="font/ttf")

    @app.get("/api/videos/{video_id}/files/{kind}")
    def video_file(video_id: str, kind: str, download: bool = False):
        meta = _meta(video_id)
        if meta["status"] != jobs.READY:
            raise HTTPException(404, "파일이 아직 준비되지 않았습니다.")
        transcript = library.transcript(video_id)
        try:
            content, suffix, media_type = subtitles.export(transcript, kind)
        except KeyError:
            raise HTTPException(404, "알 수 없는 파일 종류입니다.") from None
        headers = {}
        if download:
            filename = quote((meta["title"] or video_id) + suffix)
            headers["Content-Disposition"] = f"attachment; filename*=utf-8''{filename}"
        return Response(content, media_type=media_type, headers=headers)

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
