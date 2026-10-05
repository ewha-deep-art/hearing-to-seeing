import pytest
import requests

from hearing_to_seeing import stt
from hearing_to_seeing.schema import Transcript, WordEntry

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
    monkeypatch.setattr(stt, "RETRY_WAIT", 0)

    def post(url, **kwargs):
        calls.append((url, kwargs))
        if api.failures:
            return FakeResponse(api.failures.pop(0), {"detail": "busy"})
        return api.response

    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"RIFF")

    api.response = FakeResponse(200, RESPONSE)
    api.failures = []
    api.calls = calls
    api.wav_path = str(wav_path)
    api.loaded = None  # the model /health reports the server holds
    monkeypatch.setattr(stt.requests, "post", post)
    monkeypatch.setattr(stt.requests, "get", lambda url, **_: FakeResponse(
        200, {"status": "ok", "loaded_model": {"model": api.loaded}}))
    return api


def test_transcribe_posts_audio_with_api_key(api):
    stt.transcribe(api.wav_path, language="en")

    url, kwargs = api.calls[0]
    assert url == "http://example.test/whisperx"
    assert kwargs["headers"] == {"X-API-Key": "secret"}
    assert kwargs["files"]["file"] == ("audio.wav", b"RIFF", "audio/wav")
    model, size = stt.PASSES[0]
    assert kwargs["data"] == {
        "align": "true", "diarize": "true", "language": "en",
        "model": model, "chunk_size": str(size),
    }


def _passes(api):
    """The decoding requests, leaving out the language detection one."""
    return [kwargs for _, kwargs in api.calls if "align" in kwargs["data"]]


def _sent(api):
    return [(kwargs["data"]["model"], kwargs["data"]["chunk_size"]) for kwargs in _passes(api)]


def test_transcribe_decodes_once_per_model_and_window(api):
    stt.transcribe(api.wav_path, language="ko")

    assert sorted(_sent(api)) == sorted((model, str(size)) for model, size in stt.PASSES)


def test_passes_are_sent_grouped_by_model_starting_with_the_loaded_one(api):
    api.loaded = "large-v2"
    stt.transcribe(api.wav_path, language="ko")

    models = [model for model, _ in _sent(api)]
    assert models[0] == "large-v2"
    assert models == sorted(models, key=models.index)  # each model's passes together


def test_only_the_first_pass_is_diarized(api):
    stt.transcribe(api.wav_path, language="ko", num_speakers=6)

    diarized = [(k["data"]["model"], k["data"]["chunk_size"]) for k in _passes(api) if k["data"]["diarize"] == "true"]
    model, size = stt.PASSES[0]
    assert diarized == [(model, str(size))]
    assert all("num_speakers" not in k["data"] for k in _passes(api) if k["data"]["diarize"] == "false")


def test_transcribe_passes_a_known_speaker_count_to_diarization(api):
    stt.transcribe(api.wav_path, num_speakers=6)

    assert [k["data"].get("num_speakers") for k in _passes(api) if k["data"]["diarize"] == "true"] == ["6"]
    api.calls.clear()
    stt.transcribe(api.wav_path)
    assert all("num_speakers" not in k["data"] for k in _passes(api))


def test_transcribe_detects_an_unspecified_language_first(api):
    transcript = stt.transcribe(api.wav_path)

    # A quick decode with nothing aligned or diarized, then every pass in it.
    assert api.calls[0][1]["data"] == {"model": stt.DETECTION_MODEL}
    assert {kwargs["data"]["language"] for kwargs in _passes(api)} == {"en"}
    assert len(_passes(api)) == len(api.calls) - 1
    assert transcript.language == "en"


def test_korean_passes_use_the_korean_aligner(api):
    stt.transcribe(api.wav_path, language="ko")

    assert {kwargs["data"]["align_model_name"] for _, kwargs in api.calls} == {stt.ALIGN_MODEL}
    stt.transcribe(api.wav_path, language="en")
    assert "align_model_name" not in api.calls[-1][1]["data"]


def test_transcribe_maps_words_and_times_untimed_ones(api):
    transcript = stt.transcribe(api.wav_path)

    assert [w.text for w in transcript.words] == ["Morning,", "Scott.", "Your", "090", "bogey."]
    assert transcript.words[0].speaker == "SPEAKER_01"
    assert transcript.words[2].speaker == "SPEAKER_00"
    assert (transcript.words[0].start, transcript.words[0].end) == (29.191, 29.471)
    # Alignment skipped "090"; it takes the gap between its neighbours.
    assert (transcript.words[3].start, transcript.words[3].end) == (40.2, 41.0)
    assert transcript.words[3].speaker == "SPEAKER_00"


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


def test_transcribe_retries_a_busy_server(api):
    api.failures = [503, 503]

    assert stt.transcribe(api.wav_path).words
    assert len(_passes(api)) == len(stt.PASSES)
    assert len(api.calls) == 1 + len(stt.PASSES) + 2


def test_transcribe_gives_up_when_the_server_stays_busy(api):
    api.failures = [503] * (stt.REQUEST_RETRIES + 1)

    with pytest.raises(stt.STTRequestError, match="503"):
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


# --- parse_result: cleaning up WhisperX's output --------------------------------

def _word(text, start, end, speaker="SPEAKER_00"):
    return {"word": text, "start": start, "end": end, "speaker": speaker}


def _result(*segments):
    return {"language": "ko", "segments": [
        {"text": " ".join(w["word"] for w in words), "speaker": words[0].get("speaker"), "words": words}
        for words in segments
    ]}


def test_words_keep_their_own_speaker_within_a_segment():
    result = _result([
        _word("가버려", 1.0, 1.4, "SPEAKER_00"),
        _word("오히려", 1.6, 2.0, "SPEAKER_03"),
        _word("좋지", 2.1, 2.4, None),
    ])

    words = stt.parse_result(result).words
    assert [w.speaker for w in words] == ["SPEAKER_00", "SPEAKER_03", "SPEAKER_03"]


def test_decoding_loops_are_cut_down():
    loop = [_word("스폰지밥!", 10 + i, 10.5 + i) for i in range(12)]
    result = _result([_word("아하하핳핳핳핳핳핳핳핳핳핳", 0.0, 0.8)], loop)

    words = stt.parse_result(result).words
    assert words[0].text == "아하하핳핳핳핳"
    assert [w.text for w in words[1:]] == ["스폰지밥!"] * stt.MAX_WORD_REPEAT


def test_broadcast_credit_hallucinations_are_dropped():
    result = _result(
        [_word("잘", 1.0, 1.2), _word("자라.", 1.3, 1.6)],
        [_word("MBC", 5.0, 5.2), _word("뉴스", 5.2, 5.4), _word("이학수입니다.", 5.4, 6.0)],
    )

    assert [w.text for w in stt.parse_result(result).words] == ["잘", "자라."]


def test_words_stretched_over_a_pause_are_clamped():
    result = _result([_word("봐요!", 150.5, 155.4), _word("못생겼죠?", 155.5, 156.6)])

    words = stt.parse_result(result).words
    assert (words[0].start, words[0].end) == (150.5, 151.5)
    assert words[1].end == 156.6


def test_merge_keeps_the_pass_the_others_agree_with():
    good = [_word("핑핑아", 63.0, 63.5), _word("너", 63.6, 63.8), _word("정말", 63.9, 64.2), _word("똑똑하다", 64.3, 65.0)]
    looped = [_word("하하핳핳핳핳핳핳핳핳", 63.0, 65.0)]
    slightly_off = [_word("핑핑아", 63.0, 63.5), _word("넌", 63.6, 63.8), _word("정말", 63.9, 64.2), _word("똑똑하다", 64.3, 65.0)]

    words = stt.parse_result([_result(looped), _result(good), _result(slightly_off)]).words
    assert [w.text for w in words] == ["핑핑아", "너", "정말", "똑똑하다"]


def test_merge_matches_speaker_labels_across_passes():
    first = _result([_word("안녕", 1.0, 1.5, "SPEAKER_00"), _word("뚱아", 1.6, 2.0, "SPEAKER_00")],
                    [_word("안녕", 5.0, 5.5, "SPEAKER_01")])
    # Same audio diarized separately: same people, swapped labels, and this
    # pass is the one picked for the first region.
    second = _result([_word("안녕,", 1.0, 1.5, "SPEAKER_01"), _word("뚱아", 1.6, 2.0, "SPEAKER_01")],
                     [_word("안녕", 5.0, 5.5, "SPEAKER_00")])
    third = _result([_word("안녕,", 1.0, 1.5, "SPEAKER_01"), _word("뚱아", 1.6, 2.0, "SPEAKER_01")])

    words = stt.parse_result([first, second, third]).words
    assert [w.speaker for w in words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]


def test_undiarized_passes_borrow_the_first_pass_labels_by_time():
    first = _result([_word("안녕", 1.0, 1.5, "SPEAKER_01"), _word("뚱아", 1.6, 2.0, "SPEAKER_01")],
                    [_word("어", 5.0, 5.5, "SPEAKER_00")])
    # Not diarized: no labels at all, and timed slightly differently.
    second = _result([_word("안녕", 1.1, 1.5, None), _word("뚱아", 1.6, 1.9, None)],
                     [_word("어", 5.1, 5.4, None), _word("응", 7.0, 7.2, None)])

    first_words = stt._clean_words(first)
    borrowed = stt._clean_words(second, first_words)
    # Overlapping words take the overlapped word's label; 응 the nearest one's.
    assert [w["speaker"] for w in borrowed] == ["SPEAKER_01", "SPEAKER_01", "SPEAKER_00", "SPEAKER_00"]
    words = stt.parse_result([first, second, second]).words
    assert {w.speaker for w in words} == {"SPEAKER_00", "SPEAKER_01"}


def test_merge_times_and_labels_words_by_the_passes_that_agree():
    # One pass failed to align the line and squeezed it into its first moment.
    squeezed = _result([_word("어디", 9.8, 9.85, "SPEAKER_01"), _word("갔지?", 9.85, 9.9, "SPEAKER_01")])
    placed = _result([_word("어디", 10.0, 10.3, "SPEAKER_00"), _word("갔지?", 10.4, 10.8, "SPEAKER_00")])
    also_placed = _result([_word("어디", 10.1, 10.3, "SPEAKER_00"), _word("갔지?", 10.4, 10.7, "SPEAKER_00")])

    words = stt.parse_result([placed, squeezed, also_placed]).words
    assert [w.text for w in words] == ["어디", "갔지?"]
    assert [w.start for w in words] == [10.0, 10.4]
    assert {w.speaker for w in words} == {"SPEAKER_00"}



def _disputed():
    """Three passes of one line: 다리가 is heard once, 파리가 twice; one pass adds 어."""
    return [
        _result([_word("파리가", 1.0, 1.4), _word("안", 1.5, 1.6), _word("떨어져요", 1.7, 2.2)]),
        _result([_word("다리가", 1.1, 1.5), _word("안", 1.5, 1.6), _word("떨어져요", 1.7, 2.2)]),
        _result([_word("파리가", 1.0, 1.4), _word("어", 1.45, 1.5), _word("안", 1.5, 1.6),
                 _word("떨어져요", 1.7, 2.2)]),
    ]


def test_decide_is_asked_only_where_the_passes_disagree():
    asked = []

    def decide(options, pick, at):
        asked.append((sorted((o.text, o.votes) for o in options), pick))
        return pick

    words = stt.parse_result(_disputed(), decide=decide).words

    assert [w.text for w in words] == ["파리가", "안", "떨어져요"]
    assert asked == [([("다리가", 1), ("파리가", 2)], "파리가"), ([("", 2), ("어", 1)], "")]


def test_decide_can_take_the_minority_reading_with_its_timing():
    words = stt.parse_result(_disputed(), decide=lambda options, pick, at: "다리가" if pick else pick).words

    assert [w.text for w in words] == ["다리가", "안", "떨어져요"]
    assert words[0].start == 1.1


def test_an_answer_that_is_no_option_keeps_the_vote():
    words = stt.parse_result(_disputed(), decide=lambda options, pick, at: "모기가").words
    assert [w.text for w in words] == ["파리가", "안", "떨어져요"]


def test_merged_words_carry_the_spots_they_came_from():
    words = stt.merged_words(_disputed(), decide=lambda options, pick, at: pick)

    assert {o.text for o in words[0]["spot"]} == {"파리가", "다리가"}
    # The 어 nobody else heard is left out, and the next word says where it would go.
    assert [{o.text for o in spot} for spot in words[1]["spot_before"]] == [{"", "어"}]
    assert "spot" not in words[2]

# --- correct_names ---------------------------------------------------------------

NAMES = ["스폰지밥", "뚱이", "징징이", "집게사장", "플랑크톤", "핑핑이", "진주"]


def _names_fixed(*texts):
    transcript = Transcript(words=[WordEntry(t, 0.0, 1.0) for t in texts])
    stt.correct_names(transcript, NAMES)
    return [w.text for w in transcript.words]


def test_near_miss_names_are_respelled_keeping_particles():
    assert _names_fixed("스펀지밥!", "스펀지밥한테", "징지아,") == ["스폰지밥!", "스폰지밥한테", "징징아,"]


def test_names_already_right_or_called_out_are_left_alone():
    assert _names_fixed("징징아", "핑핑아,", "플랑크톤이", "스폰지밥은") == ["징징아", "핑핑아,", "플랑크톤이", "스폰지밥은"]


def test_everyday_words_near_short_names_are_left_alone():
    assert _names_fixed("진짜", "진정이", "사장님", "집게리아") == ["진짜", "진정이", "사장님", "집게리아"]
