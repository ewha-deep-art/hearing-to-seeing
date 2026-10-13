import json

import pytest
from fastapi.testclient import TestClient

from hearing_to_seeing.converter.ass import generate_ass
from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing import media
from web import jobs, subtitles
from web.app import create_app


class InlineExecutor:
    """Runs submitted work immediately, so a request returns with the job done."""

    def submit(self, fn, *args):
        fn(*args)


def _transcript():
    return Transcript(words=[
        WordEntry("hello", 0.0, 0.4, "SPEAKER_00", 0.5),
        WordEntry("world", 0.4, 0.8, "SPEAKER_00", 0.5),
        WordEntry("bye", 1.0, 1.3, "SPEAKER_01", 1.0),
    ])


@pytest.fixture
def fake_media(monkeypatch):
    """Replaces ffmpeg/yt-dlp/STT with stubs that write placeholder files.

    Returns the list of `pipeline.run` calls, for checking what it was given.
    """
    runs: list[dict] = []

    def touch(path, *_):
        open(path, "wb").close()
        return path

    def download(url, dest_dir):
        return touch(f"{dest_dir}/source.mp4"), "Fetched title"

    def run(wav, ass, language, output_json, title=None, media_path=None):
        runs.append({"title": title, "media_path": media_path})
        transcript = _transcript()
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(transcript.to_dict(), f)
        with open(ass, "w", encoding="utf-8") as f:
            f.write(generate_ass(transcript))
        return transcript

    monkeypatch.setattr(media, "download_youtube", download)
    monkeypatch.setattr(media, "probe", lambda _: {"duration": 1.3, "video": "h264", "audio": "aac"})
    monkeypatch.setattr(media, "extract_wav", touch)
    monkeypatch.setattr(media, "prepare_playback", touch)
    monkeypatch.setattr(media, "extract_thumbnail", touch)
    monkeypatch.setattr("hearing_to_seeing.pipeline.run", run)
    return runs


@pytest.fixture
def library(tmp_path):
    return jobs.Library(str(tmp_path), executor=InlineExecutor())


@pytest.fixture
def client(library):
    return TestClient(create_app(library))


# --- subtitles ---------------------------------------------------------------


def test_ass_to_hex_reverses_byte_order():
    assert subtitles.ass_to_hex("&H00009FE6") == "#E69F00"


def test_speakers_are_numbered_in_label_order():
    legend = subtitles.speakers(_transcript())
    assert [s["label"] for s in legend] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s["name"] for s in legend] == ["화자 1", "화자 2"]
    assert all(s["color"].startswith("#") and len(s["color"]) == 7 for s in legend)


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


def test_url_runs_to_ready(library, fake_media):
    video_id = library.add_url("https://youtu.be/x", "clip")
    meta = library.get(video_id)

    assert meta["status"] == jobs.READY
    assert meta["title"] == "clip"
    assert meta["language"] == "ko"
    assert meta["duration"] == 1.3
    with open(library.path(video_id, jobs.PLAIN_ASS), encoding="utf-8") as f:
        assert "\\kf" not in f.read()


def test_url_run_records_step_timings(library, fake_media):
    video_id = library.add_url("https://youtu.be/x", "clip")
    meta = library.get(video_id)

    timings = meta["timings"]
    assert set(timings) == {"downloading", "preparing", "transcribing", "finishing"}
    assert all(isinstance(v, float) and v >= 0 for v in timings.values())


def test_youtube_uses_fetched_title(library, fake_media):
    video_id = library.add_url("https://youtu.be/x", None)
    meta = library.get(video_id)
    assert meta["status"] == jobs.READY
    assert meta["title"] == "Fetched title"


def test_pipeline_gets_the_title_for_speaker_lookup(library, fake_media):
    library.add_url("https://youtu.be/x", None)
    assert fake_media[0]["title"] == "Fetched title"


def test_pipeline_gets_the_source_video_for_the_review(library, fake_media):
    library.add_url("https://youtu.be/x", None)
    assert fake_media[0]["media_path"].endswith("source.mp4")


def test_failure_is_recorded(library, fake_media, monkeypatch):
    def boom(*_, **__):
        raise RuntimeError("STT server down")

    monkeypatch.setattr("hearing_to_seeing.pipeline.run", boom)
    video_id = library.add_url("https://youtu.be/x", None)
    meta = library.get(video_id)
    assert meta["status"] == jobs.ERROR
    assert meta["error"] == "STT server down"


def test_source_without_audio_fails(library, fake_media, monkeypatch):
    monkeypatch.setattr(media, "probe", lambda _: {"duration": 1.0, "video": "h264", "audio": None})
    video_id = library.add_url("https://youtu.be/silent", None)
    assert library.get(video_id)["status"] == jobs.ERROR


def test_recover_marks_unfinished_jobs_failed(tmp_path):
    lib = jobs.Library(str(tmp_path), executor=InlineExecutor())
    video_id = lib._create(status="transcribing")
    lib.recover()
    assert lib.get(video_id)["status"] == jobs.ERROR


def test_path_rejects_ids_outside_the_library(library):
    with pytest.raises(KeyError):
        library.path("../etc")


# --- API ---------------------------------------------------------------------


def test_post_then_fetch_ready_video(client, fake_media):
    res = client.post("/api/videos", data={"url": "https://youtu.be/x", "title": "  "})
    assert res.status_code == 201
    video_id = res.json()["id"]

    listing = client.get("/api/videos").json()
    assert [v["id"] for v in listing] == [video_id]

    video = client.get(f"/api/videos/{video_id}").json()
    assert video["title"] == "Fetched title"
    assert video["language"] == "ko"
    assert [c["text"] for c in video["cues"]] == ["hello world", "bye"]
    assert len(video["speakers"]) == 2


def test_post_requires_a_valid_url(client):
    assert client.post("/api/videos", data={"title": "x"}).status_code == 422
    assert client.post("/api/videos", data={"url": "youtube.com/x"}).status_code == 422


def test_unknown_or_malformed_id_is_404(client):
    assert client.get("/api/videos/0123456789ab").status_code == 404
    assert client.get("/api/videos/not-an-id").status_code == 404
    assert client.get("/watch/0123456789ab").status_code == 404


def test_subtitle_download_is_an_attachment(client, fake_media):
    video_id = client.post(
        "/api/videos", data={"url": "https://youtu.be/x", "title": "talk"}
    ).json()["id"]

    inline = client.get(f"/api/videos/{video_id}/files/ass")
    assert inline.status_code == 200
    assert "Dialogue:" in inline.text
    assert "attachment" not in inline.headers.get("content-disposition", "")

    download = client.get(f"/api/videos/{video_id}/files/plain?download=1")
    assert "attachment" in download.headers["content-disposition"]
    assert "talk.plain.ass" in download.headers["content-disposition"]

    assert client.get(f"/api/videos/{video_id}/files/exe").status_code == 404


def test_media_supports_range_requests(client, fake_media, library):
    video_id = client.post("/api/videos", data={"url": "https://youtu.be/x"}).json()["id"]
    with open(library.path(video_id, jobs.PLAYBACK), "wb") as f:
        f.write(b"0123456789")

    res = client.get(f"/api/videos/{video_id}/media", headers={"Range": "bytes=2-5"})
    assert res.status_code == 206
    assert res.content == b"2345"


def test_delete_removes_video(client, fake_media):
    video_id = client.post("/api/videos", data={"url": "https://youtu.be/x"}).json()["id"]
    assert client.delete(f"/api/videos/{video_id}").status_code == 204
    assert client.get(f"/api/videos/{video_id}").status_code == 404


def test_pages_are_served(client):
    assert "영상 추가" in client.get("/").text
    assert client.get("/static/watch.js").status_code == 200


def test_player_draws_subtitles_in_the_bundled_font(client):
    from hearing_to_seeing.design.layout import FONT_FILE, FONT_NAME, FONTS_DIR

    font = client.get("/api/subtitle-font")
    assert font.status_code == 200
    with open(f"{FONTS_DIR}/{FONT_FILE}", "rb") as f:
        assert font.content == f.read()
    # libass in the player finds the face by the name the ASS styles use.
    assert f'const SUBTITLE_FONT_NAME = "{FONT_NAME}";' in client.get("/static/watch.js").text
