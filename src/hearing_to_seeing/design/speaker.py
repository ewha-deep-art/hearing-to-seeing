from hearing_to_seeing.schema import Transcript

# Wong colorblind-friendly palette, ASS BGR hex 형식(&HAABBGGRR)으로 정확히 변환됨
_PALETTE = [
    "&H00009FE6",  # Orange   (RGB #E69F00)
    "&H00E9B456",  # Sky Blue (RGB #56B4E9)
    "&H00739E00",  # Bluish Green (RGB #009E73)
    "&H0042E4F0",  # Yellow   (RGB #F0E442)
    "&H00B27200",  # Blue     (RGB #0072B2)
    "&H00005ED5",  # Vermillion (RGB #D55E00)
    "&H00A779CC",  # Reddish Purple (RGB #CC79A7)
]


def assign_speaker_colors(speakers: list[str]) -> dict[str, str]:
    return {
        speaker: _PALETTE[i % len(_PALETTE)]
        for i, speaker in enumerate(sorted(set(speakers)))
    }


def annotate_speakers(transcript: Transcript) -> dict[str, str]:
    return assign_speaker_colors(transcript.speakers())