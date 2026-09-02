import sys
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Stub heavy deps not installed in dev/CI so module-level imports in stt.py succeed
for _mod in ("torch", "whisperx"):
    sys.modules.setdefault(_mod, MagicMock())

from hearing_to_seeing.schema import Transcript, WordEntry


def _sample_transcript():
    return Transcript(
        words=[
            WordEntry("Hello", 0.0, 0.3, speaker="SPEAKER_00", volume=0.0),
            WordEntry("world", 0.4, 0.7, speaker="SPEAKER_01", volume=0.0),
        ],
        language="en",
    )


def test_pipeline_run_produces_ass_file():
    from hearing_to_seeing.pipeline import run

    sample_rate = 100
    audio_data = np.ones(100, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
        output_path = tmp.name

    try:
        with (
            patch("hearing_to_seeing.pipeline.load_models", return_value=MagicMock()),
            patch("hearing_to_seeing.pipeline.transcribe", return_value=_sample_transcript()),
            patch("hearing_to_seeing.pipeline._extract_audio", return_value=(sample_rate, audio_data)),
        ):
            result = run("fake_audio.mp4", output_path)

        assert os.path.exists(output_path)
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "[Script Info]" in content
        assert "Dialogue:" in content
        assert isinstance(result, Transcript)
    finally:
        os.unlink(output_path)


def test_pipeline_run_annotates_volumes():
    from hearing_to_seeing.pipeline import run

    sample_rate = 100
    audio_data = np.concatenate([np.full(30, 100.0), np.full(30, 10.0), np.full(40, 0.0)])
    transcript = _sample_transcript()

    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
        output_path = tmp.name

    try:
        with (
            patch("hearing_to_seeing.pipeline.load_models", return_value=MagicMock()),
            patch("hearing_to_seeing.pipeline.transcribe", return_value=transcript),
            patch("hearing_to_seeing.pipeline._extract_audio", return_value=(sample_rate, audio_data)),
        ):
            result = run("fake.mp4", output_path)

        volumes = [w.volume for w in result.words]
        assert not all(a == 0.0 for a in volumes)
    finally:
        os.unlink(output_path)
