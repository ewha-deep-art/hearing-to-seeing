"""The Supabase `clips` table: one row per YouTube video.

Rows come back as plain dicts without the heavy `transcript`/`search_text`
columns, with timestamps as epoch seconds.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

GENRES = ("드라마", "영화", "애니메이션", "기타")

# Everything but `transcript` and `search_text`.
COLUMNS = "id,youtube_id,title,channel,genre,duration,status,error,timings,created_at,submitted_at,heartbeat_at"
TIMESTAMPS = ("created_at", "submitted_at", "heartbeat_at")


def _to_epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _from_row(row: dict) -> dict:
    return {**row, **{k: _to_epoch(row[k]) for k in TIMESTAMPS if row.get(k)}}


def _to_row(fields: dict) -> dict:
    return {**fields, **{k: _to_iso(fields[k]) for k in TIMESTAMPS if fields.get(k) is not None}}


def _like_pattern(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class SupabaseStore:
    def __init__(self, url: str | None = None, key: str | None = None):
        from supabase import create_client

        url = url or os.environ.get("SUPABASE_URL")
        key = key or os.environ.get("SUPABASE_SECRET_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY must be set.")
        client = create_client(url, key)
        self.table = lambda: client.table("clips")
        self.rpc = client.rpc

    def get(self, clip_id: str) -> dict | None:
        rows = self.table().select(COLUMNS).eq("id", clip_id).execute().data
        return _from_row(rows[0]) if rows else None

    def list(self, genre: str | None = None, query: str | None = None, limit: int = 200) -> list[dict]:
        """Ready clips, newest first; `query` matches the title, channel,
        character names and dialogue."""
        request = self.table().select(COLUMNS).eq("status", "ready")
        if genre:
            request = request.eq("genre", genre)
        if query:
            request = request.ilike("search_text", _like_pattern(query.lower()))
        rows = request.order("created_at", desc=True).limit(limit).execute().data
        return [_from_row(row) for row in rows]

    def claim(self) -> dict | None:
        """Fails clips a stopped worker left behind, then takes the oldest
        queued clip for this worker (`claim_clip` in supabase/migrations); its
        `id` and `youtube_id`, or None."""
        rows = self.rpc("claim_clip", {}).execute().data
        return rows[0] if rows else None

    def heartbeat(self, clip_id: str) -> None:
        self.rpc("heartbeat_clip", {"p_id": clip_id}).execute()

    def count_submitted_since(self, epoch: float) -> int:
        return self.table().select("id", count="exact").gte("submitted_at", _to_iso(epoch)).execute().count

    def submit(self, clip_id: str, youtube_id: str, now: float, day_start: float, limit: int) -> dict:
        """Queues `youtube_id` under `clip_id` unless it is already in the
        catalogue or today's `limit` is spent, atomically (`submit_clip` in
        supabase/migrations); returns `id`, `existing` and `over_quota`."""
        params = {
            "p_id": clip_id, "p_youtube_id": youtube_id, "p_now": _to_iso(now),
            "p_day_start": _to_iso(day_start), "p_limit": limit,
        }
        return self.rpc("submit_clip", params).execute().data[0]

    def update(self, clip_id: str, **fields) -> None:
        self.table().update(_to_row(fields)).eq("id", clip_id).execute()

    def get_transcript(self, clip_id: str) -> dict | None:
        rows = self.table().select("transcript").eq("id", clip_id).execute().data
        return rows[0]["transcript"] if rows else None

    def delete(self, clip_id: str) -> None:
        self.table().delete().eq("id", clip_id).execute()
