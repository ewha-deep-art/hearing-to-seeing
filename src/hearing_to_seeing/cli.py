"""Command line entry point for the `hearing-to-seeing` console script.

`run` sends the media to the remote WhisperX API for STT, alignment and
diarization, then builds the subtitles locally. See `pipeline.run`.
"""

import argparse
import os
import sys


def _default_output(media_path: str, suffix: str) -> str:
    """`<media>.ass` / `<media>.json` beside the input, so a bare run still lands somewhere predictable."""
    return os.path.splitext(media_path)[0] + suffix


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hearing-to-seeing",
        description="Turn speaker, timing and volume into a dynamic ASS subtitle file.",
    )
    parser.add_argument("media", help="source WAV file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="destination .ass file (default: alongside the input media)",
    )
    parser.add_argument(
        "--json", nargs="?", const=True, default=None, metavar="PATH",
        help="also write the intermediate transcript schema "
             "(bare flag: alongside the input media)",
    )
    parser.add_argument(
        "--language", default=None, help="spoken language code (default: none)",
    )
    parser.add_argument(
        "--speakers", type=int, default=None, metavar="N",
        help="number of speakers, if known; passed to diarization as the exact count "
             "(default: detected)",
    )
    parser.add_argument(
        "--video", default=None, metavar="PATH",
        help="the source video, for the model to check speakers against what is on "
             "screen (needs GEMINI_API_KEY; default: it listens to the WAV)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    for path in filter(None, [args.media, args.video]):
        if not os.path.isfile(path):
            print(f"media file not found: {path}", file=sys.stderr)
            return 2

    output_ass = args.output or _default_output(args.media, ".ass")
    # `--json` doubles as a flag and an option: True means "yes, at the default path".
    output_json = args.json
    if output_json is True:
        output_json = _default_output(args.media, ".json")

    parent = os.path.dirname(os.path.abspath(output_ass))
    os.makedirs(parent, exist_ok=True)

    # Imported here so `--help` and the argument errors above stay fast; pipeline
    # pulls in numpy and requests.
    from hearing_to_seeing import pipeline
    from hearing_to_seeing.stt import STTRequestError

    try:
        transcript = pipeline.run(
            args.media, output_ass, args.language, output_json,
            num_speakers=args.speakers, media_path=args.video,
        )
    except STTRequestError as exc:
        print(exc, file=sys.stderr)
        return 1

    speakers = len(transcript.speakers())
    print(f"wrote {output_ass} ({len(transcript.words)} words, {speakers} speakers)")
    if output_json:
        print(f"wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
