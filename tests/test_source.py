import pytest

from hearing_to_seeing.schema import MediaInfo
from hearing_to_seeing.source import SourceError, is_url, resolve


def test_is_url_accepts_http_and_https():
    assert is_url("https://www.youtube.com/watch?v=abc123")
    assert is_url("http://example.com/clip.mp4")


def test_is_url_tolerates_surrounding_whitespace():
    assert is_url("  https://youtu.be/abc123  ")


def test_is_url_rejects_local_paths():
    assert not is_url("data/input/sample.mp4")
    assert not is_url("/home/user/sample.mp4")
    # A Windows drive letter has a colon but no scheme separator.
    assert not is_url(r"C:\Users\cse\sample.mp4")


def test_resolve_local_file_returns_its_path(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")

    resolved = resolve(str(media))

    assert resolved.media_path == str(media)


def test_resolve_local_file_keeps_explicit_title(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")

    resolved = resolve(str(media), title="기생충")

    assert resolved.info.title == "기생충"


def test_resolve_local_file_does_not_guess_title_from_filename(tmp_path):
    # A filename is not a title: "sample" would give the lookup step something
    # confidently wrong to search for.
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"")

    resolved = resolve(str(media))

    assert resolved.info.title is None


def test_resolve_missing_local_file_raises():
    with pytest.raises(SourceError):
        resolve("data/input/does-not-exist.mp4")


def test_media_info_merge_fills_only_unset_fields():
    explicit = MediaInfo(title="기생충")
    stored = MediaInfo(title="[4K] 기생충 명장면", source_url="https://youtu.be/abc", uploader="채널")

    merged = explicit.merge(stored)

    assert merged.title == "기생충"           # the caller's value wins
    assert merged.source_url == "https://youtu.be/abc"  # and the rest is kept
    assert merged.uploader == "채널"


def test_media_info_merge_leaves_original_untouched():
    explicit = MediaInfo(title="기생충")
    explicit.merge(MediaInfo(uploader="채널"))
    assert explicit.uploader is None
