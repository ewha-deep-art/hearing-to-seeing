import pytest
import requests

from hearing_to_seeing import stt

# Trimmed from a real /whisperx response (align=true, diarize=true).
RESPONSE = {
    "transcript": "Morning, Scott. Your vector's 090 for bogey.",
    "language": "en",
    "segments": [
        {
            "start": 29.191, "end": 29.852, "text": " Morning, Scott.",
            "speaker": "SPEAKER_01",
            "words": [
                {"word": "Morning,", "start": 29.191, "end": 29.471, "score": 0.314, "speaker": "SPEAKER_01"},
                {"word": "Scott.", "start": 29.491, "end": 29.852, "score": 0.741, "speaker": "SPEAKER_01"},
            ],
        },
        {
            "start": 40.0, "end": 41.5, "text": " Your vector's 090 for bogey.",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "Your", "start": 40.0, "end": 40.2, "score": 0.9, "speaker": "SPEAKER_00"},
                {"word": "090"},
                {"word": "bogey.", "start": 41.0, "end": 41.5, "score": 0.8, "speaker": "SPEAKER_00"},
            ],
        },
    ],
}


class FakeResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Stubs the WAV file and the HTTP call, recording what was posted."""
    calls = []
    monkeypatch.setattr(stt, "load_dotenv", lambda: None)
    monkeypatch.setenv("WHISPERX_API_KEY", "secret")
    monkeypatch.setenv("WHISPERX_API_URL", "http://example.test/whisperx")

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return api.response

    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"RIFF")

    api.response = FakeResponse(200, RESPONSE)
    api.calls = calls
    api.wav_path = str(wav_path)
    monkeypatch.setattr(stt.requests, "post", post)
    return api


def test_transcribe_posts_audio_with_api_key(api):
    stt.transcribe(api.wav_path, language="en")

    url, kwargs = api.calls[0]
    assert url == "http://example.test/whisperx"
    assert kwargs["headers"] == {"X-API-Key": "secret"}
    assert kwargs["files"]["file"] == ("audio.wav", b"RIFF", "audio/wav")
    assert kwargs["data"] == {"align": "true", "diarize": "true", "language": "en"}


def test_transcribe_omits_language_for_auto_detection(api):
    transcript = stt.transcribe(api.wav_path)

    assert "language" not in api.calls[0][1]["data"]
    assert transcript.language == "en"


def test_transcribe_maps_words_and_skips_untimed(api):
    transcript = stt.transcribe(api.wav_path)

    assert [w.text for w in transcript.words] == ["Morning,", "Scott.", "Your", "bogey."]
    assert transcript.words[0].speaker == "SPEAKER_01"
    assert transcript.words[2].speaker == "SPEAKER_00"
    assert (transcript.words[0].start, transcript.words[0].end) == (29.191, 29.471)


def test_transcribe_raises_on_error_status(api):
    api.response = FakeResponse(401, {"detail": "Invalid or missing X-API-Key header"})

    with pytest.raises(stt.STTRequestError, match="401: Invalid or missing"):
        stt.transcribe(api.wav_path)


def test_transcribe_wraps_connection_errors(api, monkeypatch):
    def post(url, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(stt.requests, "post", post)
    with pytest.raises(stt.STTRequestError, match="request failed"):
        stt.transcribe(api.wav_path)


def test_transcribe_requires_api_key(api, monkeypatch):
    monkeypatch.delenv("WHISPERX_API_KEY")

    with pytest.raises(stt.STTRequestError, match="WHISPERX_API_KEY"):
        stt.transcribe(api.wav_path)
    assert api.calls == []


def test_transcribe_requires_api_url(api, monkeypatch):
    monkeypatch.delenv("WHISPERX_API_URL")

    with pytest.raises(stt.STTRequestError, match="WHISPERX_API_URL"):
        stt.transcribe(api.wav_path)
    assert api.calls == []
