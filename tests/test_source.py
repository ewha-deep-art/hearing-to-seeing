import pytest

from hearing_to_seeing.schema import MediaInfo
from hearing_to_seeing.source import SourceError, is_url, resolve


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")
    return str(path)


def test_is_url_accepts_http_and_https():
    assert is_url("https://www.youtube.com/watch?v=abc123")
    assert is_url("http://example.com/clip.mp4")


def test_is_url_tolerates_surrounding_whitespace():
    assert is_url("  https://youtu.be/abc123  ")


def test_is_url_rejects_paths_and_scheme_less_urls():
    assert not is_url("data/input/sample.mp4")
    assert not is_url("/home/user/sample.mp4")
    assert not is_url("youtu.be/abc123")
    # A Windows drive letter has a colon but no scheme separator.
    assert not is_url(r"C:\Users\cse\sample.mp4")


def test_resolve_returns_the_media_path(media):
    assert resolve(media).media_path == media


def test_resolve_records_title_and_url(media):
    resolved = resolve(
        media, title="기생충", url="https://www.youtube.com/watch?v=abc123",
    )
    assert resolved.info.title == "기생충"
    assert resolved.info.source_url == "https://www.youtube.com/watch?v=abc123"


def test_resolve_does_not_guess_title_from_filename(tmp_path):
    # A filename is not a title: "sample" would give the lookup step something
    # confidently wrong to search for.
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"")

    assert resolve(str(path)).info.title is None


def test_resolve_leaves_metadata_unset_when_not_given(media):
    assert resolve(media).info == MediaInfo()


def test_resolve_rejects_a_url_without_a_scheme(media):
    with pytest.raises(SourceError):
        resolve(media, url="youtu.be/abc123")


def test_resolve_missing_media_file_raises():
    with pytest.raises(SourceError):
        resolve("data/input/does-not-exist.mp4")


def test_media_info_merge_fills_only_unset_fields():
    explicit = MediaInfo(title="기생충")
    stored = MediaInfo(title="[4K] 기생충 명장면", source_url="https://youtu.be/abc")

    merged = explicit.merge(stored)

    assert merged.title == "기생충"                     # the caller's value wins
    assert merged.source_url == "https://youtu.be/abc"  # and the rest is kept


def test_media_info_merge_leaves_original_untouched():
    explicit = MediaInfo(title="기생충")
    explicit.merge(MediaInfo(source_url="https://youtu.be/abc"))
    assert explicit.source_url is None
