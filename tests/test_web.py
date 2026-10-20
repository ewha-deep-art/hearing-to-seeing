import copy
import os

import pytest
from fastapi.testclient import TestClient

from hearing_to_seeing.converter.ass import generate_ass
from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing import media
from web import catalog, jobs, subtitles
from web.app import create_app
from worker import convert


class MemoryStore:
    """The `clips` table in a dict, so the tests run without Supabase."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def _public(self, row):
        return {k: copy.deepcopy(v) for k, v in row.items() if k not in ("transcript", "search_text")}

    def get(self, clip_id):
        row = self.rows.get(clip_id)
        return self._public(row) if row else None

    def find_by_youtube_id(self, youtube_id):
        return next((self._public(r) for r in self.rows.values() if r["youtube_id"] == youtube_id), None)

    def list(self, genre=None, query=None, limit=200):
        rows = [
            r for r in self.rows.values()
            if r["status"] == "ready"
            and (not genre or r["genre"] == genre)
            and (not query or query.lower() in (r.get("search_text") or ""))
        ]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return [self._public(r) for r in rows[:limit]]

    def with_status(self, statuses):
        rows = sorted((r for r in self.rows.values() if r["status"] in statuses), key=lambda r: r["submitted_at"])
        return [self._public(r) for r in rows]

    def count_submitted_since(self, epoch):
        return sum(r["submitted_at"] >= epoch for r in self.rows.values())

    def insert(self, fields):
        self.rows[fields["id"]] = {"genre": "기타", "transcript": None, "search_text": None, **copy.deepcopy(fields)}

    def update(self, clip_id, **fields):
        self.rows[clip_id].update(copy.deepcopy(fields))

    def get_transcript(self, clip_id):
        row = self.rows.get(clip_id)
        return copy.deepcopy(row["transcript"]) if row else None

    def delete(self, clip_id):
        del self.rows[clip_id]


def _transcript():
    return Transcript(words=[
        WordEntry("hello", 0.0, 0.4, "SPEAKER_00", 0.5),
        WordEntry("world", 0.4, 0.8, "SPEAKER_00", 0.5),
        WordEntry("bye", 1.0, 1.3, "SPEAKER_01", 1.0),
    ])


def _info(youtube_id="abcdefghijk", **fields):
    return {
        "id": youtube_id, "title": "Fetched title", "channel": "Clips", "description": "",
        "duration": 60, "categories": ["Film & Animation"], "tags": [],
        "playable_in_embed": True, "live_status": "not_live", **fields,
    }


@pytest.fixture
def fake_media(monkeypatch):
    """Replaces ffmpeg/yt-dlp/STT with stubs that write placeholder files.

    Returns the list of `pipeline.run` calls, for checking what it was given.
    `youtube_info` answers `_info()` for any link not in `fake_media.infos`.
    """
    runs = _Runs()
    infos = runs.infos

    def touch(path, *_):
        open(path, "wb").close()
        return path

    def info(url):
        runs.asked.append(url)
        return infos.get(url) or _info()

    def download(url, dest_dir):
        return touch(f"{dest_dir}/source.mp4")

    def run(wav, ass, language, title=None, media_path=None):
        runs.append({"ass": ass, "title": title, "media_path": media_path, "language": language})
        return _transcript()

    monkeypatch.setattr(media, "youtube_info", info)
    monkeypatch.setattr(media, "download_youtube", download)
    monkeypatch.setattr(media, "download_audio", lambda url, dest_dir: touch(f"{dest_dir}/audio-source.m4a"))
    monkeypatch.setattr(media, "probe", lambda _: {"duration": 1.3, "video": "h264", "audio": "aac"})
    monkeypatch.setattr(media, "extract_wav", touch)
    monkeypatch.setattr("hearing_to_seeing.pipeline.run", run)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return runs


class _Runs(list):
    """`pipeline.run` calls, plus `infos`: `youtube_info` answers by link, and
    `asked`: the links it was asked about."""

    def __init__(self):
        super().__init__()
        self.infos: dict[str, dict] = {}
        self.asked: list[str] = []


@pytest.fixture
def library():
    return jobs.Library(store=MemoryStore())


@pytest.fixture
def worker(library, tmp_path):
    return convert.Worker(library, str(tmp_path))


@pytest.fixture
def client(library, worker):
    client = TestClient(create_app(library))
    client.worker = worker  # takes what the posts queue
    return client


# --- subtitles ---------------------------------------------------------------


def test_ass_to_hex_reverses_byte_order():
    assert subtitles.ass_to_hex("&H00009FE6") == "#E69F00"


def test_speakers_are_numbered_in_label_order():
    legend = subtitles.speakers(_transcript())
    assert [s["label"] for s in legend] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s["name"] for s in legend] == ["화자 1", "화자 2"]
    assert all(s["color"].startswith("#") and len(s["color"]) == 7 for s in legend)


def test_speakers_use_character_names_and_most_talk_first():
    transcript = _transcript()
    transcript.words.append(WordEntry("again and again", 9.0, 12.0, "SPEAKER_01"))
    transcript.speaker_profiles = {"SPEAKER_01": {"color": "#E69F00", "name": "기택"}}
    legend = subtitles.speakers(transcript)
    assert [s["name"] for s in legend] == ["기택", "화자 1"]


def test_cues_match_ass_dialogue_lines():
    transcript = _transcript()
    cues = subtitles.cues(transcript)
    # One Box event per on-screen subtitle; Fill and Pop ride on top of it.
    dialogues = [l for l in generate_ass(transcript).splitlines() if l.startswith("Dialogue: 0,")]

    assert len(cues) == len(dialogues)
    assert cues[0] == {"start": 0.0, "end": cues[0]["end"], "speaker": "SPEAKER_00", "text": "hello world"}
    assert cues[1]["speaker"] == "SPEAKER_01"
    assert cues[0]["end"] <= cues[1]["start"]


# --- jobs --------------------------------------------------------------------


def _submit(worker, url="https://youtu.be/abcdefghijk"):
    """Queues `url` and has the worker convert it."""
    video_id, existing = worker.library.submit(url)
    assert not existing
    assert worker.work_once()
    return video_id


def test_url_runs_to_ready(library, worker, fake_media):
    video_id = _submit(worker)
    meta = library.get(video_id)

    assert meta["status"] == jobs.READY
    assert meta["youtube_id"] == "abcdefghijk"
    assert meta["title"] == "Fetched title"
    assert meta["channel"] == "Clips"
    assert meta["duration"] == 1.3
    assert fake_media[0]["language"] == "ko"
    assert library.transcript(video_id).to_dict() == _transcript().to_dict()


def test_nothing_stays_on_disk(worker, fake_media):
    video_id = _submit(worker)
    assert fake_media[0]["ass"] is None
    assert not os.path.exists(worker.path(video_id))


def test_failed_job_drops_working_files(worker, fake_media, monkeypatch):
    monkeypatch.setattr("hearing_to_seeing.pipeline.run", lambda *_, **__: 1 / 0)
    video_id = _submit(worker)
    assert not os.path.exists(worker.path(video_id))


def test_url_run_records_step_timings(library, worker, fake_media):
    timings = library.get(_submit(worker))["timings"]
    assert set(timings) == {"downloading", "preparing", "transcribing", "finishing"}
    assert all(isinstance(v, float) and v >= 0 for v in timings.values())


def test_pipeline_gets_the_title_for_speaker_lookup(worker, fake_media):
    _submit(worker)
    assert fake_media[0]["title"] == "Fetched title"


def test_pipeline_gets_the_source_video_for_the_review(worker, fake_media):
    _submit(worker)
    assert fake_media[0]["media_path"].result().endswith("source.mp4")


def test_the_review_listens_when_the_video_cannot_be_had(library, worker, fake_media, monkeypatch):
    monkeypatch.setattr(media, "download_youtube", lambda *_: 1 / 0)
    assert library.get(_submit(worker))["status"] == "ready"
    assert fake_media[0]["media_path"].result() is None


def test_genre_falls_back_to_other_without_a_key(library, worker, fake_media):
    assert library.get(_submit(worker))["genre"] == "기타"


def test_genre_comes_from_the_model(library, worker, fake_media, monkeypatch):
    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.configured", lambda: True)
    monkeypatch.setattr("hearing_to_seeing.knowledge.llm.ask", lambda *_, **__: {"genre": "영화"})
    assert library.get(_submit(worker))["genre"] == "영화"


def test_dialogue_is_searchable(library, worker, fake_media):
    video_id = _submit(worker)
    assert library.store.rows[video_id]["search_text"].endswith("hello world bye")
    assert [v["id"] for v in library.list(query="WORLD")] == [video_id]
    assert library.list(query="nowhere") == []


def test_same_video_is_not_converted_twice(library, worker, fake_media):
    video_id = _submit(worker, "https://youtu.be/abcdefghijk")
    again = library.submit("https://www.youtube.com/watch?v=abcdefghijk&t=3")
    assert again == (video_id, True)
    assert len(fake_media) == 1
    assert len(library.store.rows) == 1


def test_failed_video_is_submitted_again(library, worker, fake_media, monkeypatch):
    calls = []

    def flaky(*_, **__):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("STT server down")
        return _transcript()

    monkeypatch.setattr("hearing_to_seeing.pipeline.run", flaky)
    video_id = _submit(worker)
    assert library.get(video_id)["status"] == jobs.ERROR
    assert library.get(video_id)["error"] == jobs.FAILED  # the cause is only logged

    assert library.submit("https://youtu.be/abcdefghijk") == (video_id, False)
    assert worker.work_once()
    assert library.get(video_id)["status"] == jobs.READY
    assert len(library.store.rows) == 1


@pytest.mark.parametrize("fields, message", [
    ({"duration": 901}, "15분 이하"),
    ({"duration": None}, "15분 이하"),
    ({"playable_in_embed": False}, "영상 제공자가"),
    ({"live_status": "is_live"}, "라이브"),
])
def test_unsuitable_videos_are_rejected(library, worker, fake_media, fields, message):
    fake_media.infos[jobs.youtube_url("xxxxxxxxxxx")] = _info("xxxxxxxxxxx", **fields)
    video_id = _submit(worker, "https://youtu.be/xxxxxxxxxxx")
    meta = library.get(video_id)
    assert meta["status"] == jobs.ERROR
    assert message in meta["error"]
    assert fake_media == []
    # Turned away before any work: nothing spent from today's limit.
    assert library.quota()["used"] == 0


def test_the_worker_asks_youtube_about_the_id_alone(worker, fake_media):
    _submit(worker, "https://www.youtube.com/watch?feature=share&v=abcdefghijk&t=3")
    assert fake_media.asked == [jobs.youtube_url("abcdefghijk")]


def test_an_empty_queue_has_no_work(worker):
    assert worker.work_once() is False


def test_the_worker_outlasts_an_unreachable_store(library, worker, monkeypatch):
    calls = []

    def flaky(statuses):
        calls.append(statuses)
        if len(calls) == 1:
            raise ConnectionError("no address")  # at start-up, before recover() gets through
        if len(calls) == 4:
            raise KeyboardInterrupt  # stops the loop
        return []

    monkeypatch.setattr(library.store, "with_status", flaky)
    monkeypatch.setattr(convert.time, "sleep", lambda _: None)
    with pytest.raises(KeyboardInterrupt):
        worker.work()
    assert calls[1] == jobs.PENDING_STATES[1:]  # recover() tried again
    assert calls[2] == (jobs.QUEUED,)


def test_only_youtube_links_are_taken(library, fake_media):
    with pytest.raises(jobs.Rejected, match="YouTube 링크만"):
        library.submit("https://vimeo.com/123456")
    assert library.store.rows == {}


def test_daily_limit_counts_todays_submissions(tmp_path, fake_media):
    library = jobs.Library(store=MemoryStore(), daily_limit=1)
    worker = convert.Worker(library, str(tmp_path))
    _submit(worker)
    with pytest.raises(jobs.QuotaExceeded):
        library.submit("https://youtu.be/bbbbbbbbbbb")
    # A video already in the catalogue still opens.
    assert library.submit("https://youtu.be/abcdefghijk")[1] is True
    assert library.quota()["used"] == 1


def test_source_without_audio_fails(library, worker, fake_media, monkeypatch):
    monkeypatch.setattr(media, "probe", lambda _: {"duration": 1.0, "video": "h264", "audio": None})
    assert library.get(_submit(worker))["status"] == jobs.ERROR


def test_recover_marks_unfinished_jobs_failed():
    store = MemoryStore()
    store.insert({"id": "0123456789ab", "youtube_id": "x", "status": "transcribing", "created_at": 0, "submitted_at": 0})
    store.insert({"id": "ba9876543210", "youtube_id": "y", "status": "queued", "created_at": 0, "submitted_at": 0})
    restarted = jobs.Library(store=store)
    convert.Worker(restarted).recover()
    assert restarted.get("0123456789ab")["status"] == jobs.ERROR
    assert restarted.get("ba9876543210")["status"] == "queued"  # the worker takes it next


def test_path_rejects_ids_outside_the_library(worker):
    with pytest.raises(KeyError):
        worker.path("../etc")


def test_youtube_id_reads_every_link_shape():
    for url in [
        "https://youtu.be/abcdefghijk",
        "https://www.youtube.com/watch?v=abcdefghijk&t=3",
        "https://youtube.com/watch?feature=share&v=abcdefghijk",
        "https://www.youtube.com/shorts/abcdefghijk",
    ]:
        assert catalog.youtube_id(url) == "abcdefghijk"
    assert catalog.youtube_id("https://example.com/") is None


# --- API ---------------------------------------------------------------------


def _post(client, url="https://youtu.be/abcdefghijk"):
    """Posts `url`, then lets the worker take whatever was queued."""
    res = client.post("/api/videos", data={"url": url})
    client.worker.work_once()
    return res


def test_post_then_fetch_ready_video(client, fake_media):
    res = _post(client)
    assert res.status_code == 201
    video_id = res.json()["id"]

    listing = client.get("/api/videos").json()
    assert [v["id"] for v in listing] == [video_id]
    assert "transcript" not in listing[0]

    video = client.get(f"/api/videos/{video_id}").json()
    assert video["title"] == "Fetched title"
    assert [c["text"] for c in video["cues"]] == ["hello world", "bye"]
    assert len(video["speakers"]) == 2


def test_post_of_a_known_video_returns_it(client, fake_media):
    video_id = _post(client).json()["id"]
    res = _post(client)
    assert res.status_code == 200
    assert res.json() == {"id": video_id, "existing": True}


def test_post_requires_a_valid_url(client):
    assert client.post("/api/videos", data={}).status_code == 422
    assert _post(client, "youtube.com/x").status_code == 422
    res = _post(client, "https://vimeo.com/123456")
    assert res.status_code == 422
    assert res.json()["detail"] == "YouTube 링크만 자막을 만들 수 있어요."


def test_post_rejections_and_quota(client, fake_media, library):
    fake_media.infos[jobs.youtube_url("lllllllllll")] = _info("lllllllllll", duration=3600)
    res = _post(client, "https://youtu.be/lllllllllll")
    assert res.status_code == 201
    video = client.get(f"/api/videos/{res.json()['id']}").json()
    assert video["status"] == jobs.ERROR
    assert "15분" in video["error"]

    library.daily_limit = 1
    _post(client)
    assert _post(client, "https://youtu.be/bbbbbbbbbbb").status_code == 429


def test_listing_filters_by_genre_and_search(client, fake_media, library):
    video_id = _post(client).json()["id"]
    library.update(video_id, genre="영화")
    assert [v["id"] for v in client.get("/api/videos?genre=영화").json()] == [video_id]
    assert client.get("/api/videos?genre=드라마").json() == []
    assert client.get("/api/videos?genre=예능").status_code == 422
    assert [v["id"] for v in client.get("/api/videos?q=bye").json()] == [video_id]
    assert client.get("/api/videos?q=nowhere").json() == []


def test_listing_leaves_out_unfinished_videos(client, fake_media, library, monkeypatch):
    monkeypatch.setattr("hearing_to_seeing.pipeline.run", lambda *_, **__: 1 / 0)
    _post(client)
    assert client.get("/api/videos").json() == []


def test_queue_shows_quota(client, library):
    library.store.insert({"id": "0123456789ab", "youtube_id": "x", "status": "transcribing",
                          "created_at": 0, "submitted_at": 9e9})
    queue = client.get("/api/queue").json()
    assert queue["used"] == 1
    assert queue["limit"] == library.daily_limit


def test_unknown_or_malformed_id_is_404(client):
    assert client.get("/api/videos/0123456789ab").status_code == 404
    assert client.get("/api/videos/not-an-id").status_code == 404
    assert client.get("/watch/0123456789ab").status_code == 404


def test_subtitle_download_is_an_attachment(client, fake_media):
    fake_media.infos[jobs.youtube_url("ttttttttttt")] = _info("ttttttttttt", title="talk")
    video_id = _post(client, "https://youtu.be/ttttttttttt").json()["id"]

    inline = client.get(f"/api/videos/{video_id}/files/ass")
    assert inline.status_code == 200
    assert "Dialogue:" in inline.text
    assert "attachment" not in inline.headers.get("content-disposition", "")

    download = client.get(f"/api/videos/{video_id}/files/plain?download=1")
    assert "attachment" in download.headers["content-disposition"]
    assert "talk.plain.ass" in download.headers["content-disposition"]
    assert "\\kf" not in download.text

    transcript = client.get(f"/api/videos/{video_id}/files/json").json()
    assert transcript == _transcript().to_dict()

    assert client.get(f"/api/videos/{video_id}/files/exe").status_code == 404


def test_subtitle_download_keeps_a_korean_title(client, fake_media):
    fake_media.infos[jobs.youtube_url("kkkkkkkkkkk")] = _info("kkkkkkkkkkk", title="대화")
    video_id = _post(client, "https://youtu.be/kkkkkkkkkkk").json()["id"]
    download = client.get(f"/api/videos/{video_id}/files/ass?download=1")
    assert download.headers["content-disposition"] == (
        "attachment; filename*=utf-8''%EB%8C%80%ED%99%94.ass"
    )


def test_files_wait_for_the_transcript(client, library):
    library.store.insert({"id": "0123456789ab", "youtube_id": "x", "status": "queued",
                          "created_at": 0, "submitted_at": 0})
    assert client.get("/api/videos/0123456789ab/files/ass").status_code == 404


def test_delete_takes_the_admin_token(client, fake_media, monkeypatch):
    video_id = _post(client).json()["id"]
    url = f"/api/videos/{video_id}"
    assert client.delete(url, headers={"X-Admin-Token": "anything"}).status_code == 403  # unset: nobody

    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.delete(url).status_code == 403
    assert client.delete(url, headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.get(url).status_code == 200

    assert client.delete(url, headers={"X-Admin-Token": "s3cret"}).status_code == 204
    assert client.get(url).status_code == 404


def test_pages_are_served(client):
    assert "자막 만들기" in client.get("/").text
    assert client.get("/static/watch.js").status_code == 200
    assert client.get("/static/tray.js").status_code == 200


def test_player_draws_subtitles_in_the_bundled_font(client):
    from hearing_to_seeing.design.layout import FONT_FILE, FONT_NAME, FONTS_DIR

    font = client.get("/api/subtitle-font")
    assert font.status_code == 200
    with open(f"{FONTS_DIR}/{FONT_FILE}", "rb") as f:
        assert font.content == f.read()
    # libass in the player finds the face by the name the ASS styles use.
    assert f'const SUBTITLE_FONT_NAME = "{FONT_NAME}";' in client.get("/static/watch.js").text


def test_pages_and_scripts_are_revalidated(client):
    # Cached scripts under fresh HTML break the page after a deploy.
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/static/library.js").headers["cache-control"] == "no-cache"


def test_a_refused_audio_stream_is_asked_for_again(monkeypatch, tmp_path):
    asked = []

    def download(url, options):
        asked.append(options["format"])
        if len(asked) < 3:
            raise RuntimeError("HTTP Error 403: Forbidden")
        return "audio-source.webm"

    monkeypatch.setattr(media, "_download", download)
    assert media.download_audio("https://youtu.be/x", str(tmp_path)) == "audio-source.webm"
    assert asked == [media.AUDIO_FORMAT, media.AUDIO_FORMAT, media.AUDIO_FALLBACK_FORMAT]
