"""Locating and invoking the ffmpeg toolchain.

Both the volume step (which decodes audio to PCM) and the preview renderer
shell out to ffmpeg, so the binary lookup and the decode live here rather
than in either caller. Set H2S_FFMPEG / H2S_FFPROBE to use binaries that are
not on PATH.
"""

import io
import os
import shutil
import subprocess

import numpy as np


class MediaToolError(RuntimeError):
    pass


def tool(name: str) -> str:
    """Returns the path to an ffmpeg-family binary."""
    override = os.environ.get(f"H2S_{name.upper()}")
    if override:
        return override
    # ffprobe normally ships beside ffmpeg, so an ffmpeg override locates it too.
    ffmpeg_override = os.environ.get("H2S_FFMPEG")
    if ffmpeg_override:
        sibling = os.path.join(os.path.dirname(ffmpeg_override), name)
        if os.path.exists(sibling):
            return sibling
    found = shutil.which(name)
    if not found:
        raise MediaToolError(
            f"{name} not found on PATH. Install ffmpeg, or set H2S_{name.upper()} "
            "to the binary's path."
        )
    return found


def run(cmd: list[str]) -> bytes:
    """Runs a command, raising MediaToolError with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise MediaToolError(
            f"{os.path.basename(cmd[0])} failed:\n{proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout


def decode_audio(media_path: str, sample_rate: int = 16000) -> tuple[int, np.ndarray]:
    """Decodes `media_path` to mono PCM for the volume step.

    16 kHz matches what WhisperX's `load_audio` uses for STT, so the volume
    step and the STT path share one decode rate instead of running the audio
    through ffmpeg twice at different rates for no benefit to the RMS calc.

    Returns the sample rate reported by the WAV header alongside the samples,
    so callers index by time without assuming the requested rate held.
    """
    # scipy arrives as a whisperx dependency and is only needed here, so the
    # import stays local rather than loading it for every media.py caller.
    from scipy.io import wavfile

    out = run([
        tool("ffmpeg"), "-i", media_path,
        "-f", "wav", "-ac", "1", "-ar", str(sample_rate),
        "-loglevel", "quiet", "pipe:1",
    ])
    rate, audio_data = wavfile.read(io.BytesIO(out))
    return rate, audio_data
