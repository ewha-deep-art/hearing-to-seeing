import numpy as np

from hearing_to_seeing.pitch import frame_f0, gender, speaker_pitches, _frames
from hearing_to_seeing.schema import Transcript, WordEntry
from hearing_to_seeing.voice import RATE


def _voice(hz: float, seconds: float) -> np.ndarray:
    """A buzz with harmonics, as a voiced vowel has."""
    t = np.arange(int(RATE * seconds)) / RATE
    return (sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 8)) * 3000).astype(np.float32)


def test_yin_finds_the_pitch_of_a_voiced_tone():
    for hz in (95, 120, 210, 260):
        f0 = frame_f0(_frames(_voice(hz, 1.0), [(0.0, 1.0)]))
        assert abs(np.nanmedian(f0) - hz) < hz * 0.02


def test_noise_and_silence_have_no_pitch():
    noise = np.random.default_rng(0).normal(0, 3000, RATE).astype(np.float32)
    assert np.isnan(frame_f0(_frames(noise, [(0.0, 1.0)]))).all()
    assert np.isnan(frame_f0(_frames(np.zeros(RATE, np.float32), [(0.0, 1.0)]))).all()


def test_low_voices_are_male_high_ones_female_and_between_unknown():
    assert gender(120.0) == "male"
    assert gender(220.0) == "female"
    assert gender(172.0) is None
    assert gender(None) is None


def test_each_speaker_is_judged_by_their_own_speech():
    audio = np.concatenate([_voice(115, 4.0), _voice(225, 4.0), _voice(225, 1.0)])
    words = [WordEntry(f"m{i}", i * 0.5, i * 0.5 + 0.45, "SPEAKER_00") for i in range(8)]
    words += [WordEntry(f"f{i}", 4 + i * 0.5, 4 + i * 0.5 + 0.45, "SPEAKER_01") for i in range(8)]
    words += [WordEntry("short", 8.0, 8.9, "SPEAKER_02")]  # under MIN_SPEAKER_SECONDS
    pitches = speaker_pitches(Transcript(words=words), RATE, audio)
    assert {label: gender(f0) for label, f0 in pitches.items()} == {
        "SPEAKER_00": "male", "SPEAKER_01": "female", "SPEAKER_02": None,
    }
