"""Command line entry point for the `hearing-to-seeing` console script.

Two subcommands, matching the two entry points in `pipeline`:

  run        the regular path — STT, alignment and diarization run here, so it
             needs whisperx and (in practice) a GPU.
  from-json  a stopgap for machines without one: those three steps are taken
             from a transcript produced elsewhere and only the rest of the
             pipeline runs locally. See `pipeline.run_from_json`.

Both also take `--title` and `--url`, which say which work the media holds;
see `source`.
"""

import argparse
import os
import sys

from hearing_to_seeing.design.identity import SpeakerMapError, manual_mapping, parse_speaker_map
from hearing_to_seeing.media import MediaToolError
from hearing_to_seeing.source import SourceError, resolve


def _default_output(media_path: str, suffix: str) -> str:
    """`<media>.ass` / `<media>.json` beside the input, so a bare run still lands somewhere predictable."""
    return os.path.splitext(media_path)[0] + suffix


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("media", help="source audio or video file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="destination .ass file (default: alongside the input media)",
    )
    parser.add_argument(
        "--json", nargs="?", const=True, default=None, metavar="PATH",
        help="also write the intermediate transcript schema "
             "(bare flag: alongside the input media)",
    )
    # Recorded in the transcript, not used by the current effects: the
    # speaker-colour step needs to know which work this is, and only the user
    # can say. Neither option causes anything to be fetched.
    parser.add_argument(
        "--title", default=None, metavar="TITLE",
        help="title of the work being subtitled, e.g. --title \"기생충\"",
    )
    parser.add_argument(
        "--url", default=None, metavar="URL",
        help="URL the media came from (YouTube etc.), recorded as a reference",
    )
    parser.add_argument(
        "--speaker-map", default=None, metavar="MAP",
        help="name the speakers yourself, e.g. "
             "--speaker-map \"SPEAKER_00=기택,SPEAKER_01=충숙\". "
             "Taken as certain, so it overrides anything inferred",
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

    try:
        # Argument shape before the filesystem, so a typo in the mapping is not
        # hidden behind an unrelated complaint about the media path.
        speaker_map = parse_speaker_map(args.speaker_map or "")
        source = resolve(args.media, title=args.title, url=args.url)
    except (SourceError, SpeakerMapError) as exc:
        print(exc, file=sys.stderr)
        return 2

    speaker_strategy = manual_mapping(speaker_map) if speaker_map else None

    media_path = source.media_path

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
                media_path, args.transcript, output_ass, output_json,
                source.info, speaker_strategy,
            )
        else:
            transcript = pipeline.run(
                media_path, output_ass, args.language, output_json,
                source.info, speaker_strategy,
            )
    except MediaToolError as exc:
        print(exc, file=sys.stderr)
        return 1

    speakers = len(transcript.speakers())
    print(f"wrote {output_ass} ({len(transcript.words)} words, {speakers} speakers)")
    if output_json:
        print(f"wrote {output_json}")

    named = {
        label: profile.name
        for label, profile in transcript.speaker_profiles.items()
        if profile.name
    }
    if named:
        print("speakers: " + ", ".join(f"{k}={v}" for k, v in sorted(named.items())))
    # A label that matched nothing means that speaker silently kept its
    # default colour, which looks no different from a run that worked.
    unmatched = sorted(set(speaker_map) - set(transcript.speaker_profiles))
    if unmatched:
        print(
            f"warning: transcript에 없는 화자 라벨: {', '.join(unmatched)}",
            file=sys.stderr,
        )

    if transcript.media.title:
        print(f"title: {transcript.media.title}")
    else:
        # The lookup step cannot search for a work it has no name for, and a
        # missing title is easy to not notice until that step runs.
        print("title: (none — pass --title to record one)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
