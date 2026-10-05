from hearing_to_seeing.design.speaker import BASE_COLOUR, hex_to_ass, speaker_colors
from hearing_to_seeing.schema import Transcript, WordEntry


def _transcript(*labels: str) -> Transcript:
    return Transcript(words=[
        WordEntry(text="a", start=i, end=i + 0.5, speaker=label) for i, label in enumerate(labels)
    ])


def test_hex_to_ass_reorders_to_bgr():
    assert hex_to_ass("#C0392B") == "&H002B39C0"
    assert hex_to_ass("c0392b") == "&H002B39C0"


def test_hex_to_ass_passes_ass_through_and_drops_junk():
    assert hex_to_ass("&H00123456") == "&H00123456"
    assert hex_to_ass(None) == BASE_COLOUR
    assert hex_to_ass("red") == BASE_COLOUR


def test_speaker_colors_reads_the_profiles():
    transcript = _transcript("SPEAKER_00", "SPEAKER_01")
    transcript.speaker_profiles = {"SPEAKER_00": {"color": "#563412", "name": "기택"}}
    assert speaker_colors(transcript) == {"SPEAKER_00": "&H00123456", "SPEAKER_01": BASE_COLOUR}
