"""The Supabase `videos` table (id, created_at, meta jsonb, transcript jsonb):
one row per video."""

import os


class SupabaseStore:
    def __init__(self, url: str | None = None, key: str | None = None):
        from supabase import create_client

        url = url or os.environ.get("SUPABASE_URL")
        key = key or os.environ.get("SUPABASE_SECRET_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY must be set.")
        self.table = create_client(url, key).table

    def get(self, video_id: str) -> dict | None:
        rows = self.table("videos").select("meta").eq("id", video_id).execute().data
        return rows[0]["meta"] if rows else None

    def list(self) -> list[dict]:
        rows = self.table("videos").select("meta").order("created_at", desc=True).execute().data
        return [row["meta"] for row in rows]

    def insert(self, meta: dict) -> None:
        self.table("videos").insert(
            {"id": meta["id"], "created_at": meta["created_at"], "meta": meta}
        ).execute()

    def update_meta(self, video_id: str, meta: dict) -> None:
        self.table("videos").update({"meta": meta}).eq("id", video_id).execute()

    def set_transcript(self, video_id: str, transcript: dict) -> None:
        self.table("videos").update({"transcript": transcript}).eq("id", video_id).execute()

    def get_transcript(self, video_id: str) -> dict | None:
        rows = self.table("videos").select("transcript").eq("id", video_id).execute().data
        return rows[0]["transcript"] if rows else None

    def delete(self, video_id: str) -> None:
        self.table("videos").delete().eq("id", video_id).execute()
