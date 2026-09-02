import pytest
from hearing_to_seeing.schema import WordEntry
from hearing_to_seeing.design.sync import seconds_to_centiseconds, karaoke_duration, build_karaoke_text


def test_seconds_to_centiseconds_basic():
    assert seconds_to_centiseconds(0.5) == 50
    assert seconds_to_centiseconds(1.0) == 100
    assert seconds_to_centiseconds(0.1) == 10


def test_seconds_to_centiseconds_rounds():
    assert seconds_to_centiseconds(0.555) == 56
    assert seconds_to_centiseconds(0.554) == 55


def test_seconds_to_centiseconds_minimum_one():
    # Very short durations should not produce 0 centiseconds
    assert seconds_to_centiseconds(0.001) == 1


def test_karaoke_duration_uses_word_duration():
    w = WordEntry("hello", start=1.0, end=1.5)
    assert karaoke_duration(w) == 50


def test_build_karaoke_text_format():
    words = [
        WordEntry("Hello", start=0.0, end=0.3),
        WordEntry("world", start=0.4, end=0.7),
    ]
    text = build_karaoke_text(words)
    assert r"{\k30}Hello" in text
    assert r"{\k30}world" in text


def test_build_karaoke_text_single_word():
    words = [WordEntry("안녕", start=0.0, end=0.5)]
    text = build_karaoke_text(words)
    assert r"{\k50}안녕" in text


def test_build_karaoke_text_empty():
    assert build_karaoke_text([]) == ""
