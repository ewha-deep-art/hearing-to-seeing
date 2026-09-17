"""Command line entry point for the `hearing-to-seeing` console script.

Two subcommands, matching the two entry points in `pipeline`:

  run        the regular path — STT, alignment and diarization run here, so it
             needs whisperx and (in practice) a GPU.
  from-json  a stopgap for machines without one: those three steps are taken
             from a transcript produced elsewhere and only the rest of the
             pipeline runs locally. See `pipeline.run_from_json`.

Both accept a local file or a URL as their media argument; see `source`.
"""

import argparse
import os
import sys

from hearing_to_seeing.media import MediaToolError
from hearing_to_seeing.source import DEFAULT_CACHE_DIR, SourceError, resolve


def _default_output(media_path: str, suffix: str) -> str:
    """`<media>.ass` / `<media>.json` beside the input, so a bare run still lands somewhere predictable."""
    return os.path.splitext(media_path)[0] + suffix


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "media",
        help="source audio/video file, or a URL to download (YouTube etc.)",
    )
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
        "--title", default=None,
        help="title of the work being subtitled. Recorded in the transcript for "
             "the speaker-colour step; overrides the title a URL reports",
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR, metavar="DIR",
        help=f"where downloaded media is kept (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--audio-only", action="store_true",
        help="download audio only — faster, but leaves nothing for the preview "
             "renderer or any later step that looks at the picture (URL input only)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hearing-to-seeing",
        description="Turn speaker, timing and volume into a dynamic ASS subtitle file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="transcribe media and generate subtitles (regular entry point)",
    )
    _add_common_arguments(run_parser)
    run_parser.add_argument(
        "--language", default="ko", help="spoken language code (default: ko)",
    )

    json_parser = subparsers.add_parser(
        "from-json",
        help="generate subtitles from an existing transcript (temporary, no-GPU path)",
        description=(
            "Skips STT, alignment and diarization by reading their output from a "
            "file — a stopgap for environments without a usable GPU. Accepts the "
            "intermediate schema written by `run`, or a character-level timestamp "
            "export (which carries no speaker labels, so the colour effect is lost)."
        ),
    )
    _add_common_arguments(json_parser)
    json_parser.add_argument("transcript", help="transcript JSON to build subtitles from")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Resolving first: a URL has no local path to derive the default output
    # names from until it has been fetched.
    try:
        source = resolve(
            args.media,
            title=args.title,
            cache_dir=args.cache_dir,
            audio_only=args.audio_only,
        )
    except SourceError as exc:
        print(exc, file=sys.stderr)
        return 2

    media_path = source.media_path
    if media_path != args.media:
        print(f"media: {media_path}")

    output_ass = args.output or _default_output(media_path, ".ass")
    # `--json` doubles as a flag and an option: True means "yes, at the default path".
    output_json = args.json
    if output_json is True:
        output_json = _default_output(media_path, ".json")

    parent = os.path.dirname(os.path.abspath(output_ass))
    os.makedirs(parent, exist_ok=True)

    # Imported here so `--help` and the argument errors above stay fast; pipeline
    # pulls in numpy and, for `run`, torch and whisperx.
    from hearing_to_seeing import pipeline

    try:
        if args.command == "from-json":
            if not os.path.isfile(args.transcript):
                print(f"transcript file not found: {args.transcript}", file=sys.stderr)
                return 2
            transcript = pipeline.run_from_json(
                media_path, args.transcript, output_ass, output_json, source.info,
            )
        else:
            transcript = pipeline.run(
                media_path, output_ass, args.language, output_json, source.info,
            )
    except MediaToolError as exc:
        print(exc, file=sys.stderr)
        return 1

    speakers = len(transcript.speakers())
    print(f"wrote {output_ass} ({len(transcript.words)} words, {speakers} speakers)")
    if output_json:
        print(f"wrote {output_json}")

    if transcript.media.title:
        print(f"title: {transcript.media.title}")
    else:
        # The lookup step cannot search for a work it has no name for, and a
        # missing title is easy to not notice until that step runs.
        print("title: (none — pass --title to record one)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
