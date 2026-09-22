"""Command line entry point for the `hearing-to-seeing` console script.

Three subcommands, matching the entry points in `pipeline`:

  run        the regular path — STT, alignment and diarization run here, so it
             needs whisperx and (in practice) a GPU.
  from-json  a stopgap for machines without one: those three steps are taken
             from a transcript produced elsewhere and only the rest of the
             pipeline runs locally. See `pipeline.run_from_json`.
  lookup     the identification steps alone — work lookup, character colours,
             speaker → character mapping — from a transcript, with no media
             at all. For working on those steps without a video to hand.

Both also take `--title` and `--url`, which say which work the media holds;
see `source`. With `--lookup` that title is used to find the work's cast,
argue a colour for each character and infer which speaker is which character
from the dialogue (`knowledge`, `design.character_color`, `design.inference`);
that is the only path on which anything leaves the machine. The confirmed
mapping is written to the `--json` output as `character_map`, where it can be
corrected by hand and is replayed by the next `from-json` run.
"""

import argparse
import os
import sys

from hearing_to_seeing.design.identity import SpeakerMapError, manual_mapping, parse_speaker_map
from hearing_to_seeing.knowledge.llm import KnowledgeError, api_key_var, configured, make_llm
from hearing_to_seeing.media import MediaToolError
from hearing_to_seeing.schema import MediaInfo
from hearing_to_seeing.source import SourceError, is_url, resolve


# Where outputs go when no path is given: `data/output/<input name><suffix>`,
# relative to the working directory. Inputs live in `data/input/`, and mixing
# generated files in with them made the two indistinguishable at a glance.
OUTPUT_DIR = os.path.join("data", "output")


def _default_output(input_path: str, suffix: str) -> str:
    """`data/output/<input name>.ass` etc., so a bare run still lands somewhere predictable."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(OUTPUT_DIR, stem + suffix)


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    """Options every subcommand takes: what the work is, and who the speakers are."""
    # Recorded in the transcript; the lookup step needs to know which work
    # this is, and only the user can say. Neither option by itself causes
    # anything to be fetched — that takes --lookup (or the lookup subcommand).
    parser.add_argument(
        "--title", default=None, metavar="TITLE",
        help="title of the work being subtitled, e.g. --title \"기생충\"",
    )
    parser.add_argument(
        "--url", default=None, metavar="URL",
        help="URL the media came from (YouTube etc.), recorded as a reference",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="ignore the work and character_map a transcript already carries "
             "and redo lookup and mapping",
    )
    parser.add_argument(
        "--speaker-map", default=None, metavar="MAP",
        help="name the speakers yourself, e.g. "
             "--speaker-map \"SPEAKER_00=기택,SPEAKER_01=충숙\". "
             "Taken as certain, so it overrides anything inferred",
    )


def _add_media_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("media", help="source audio or video file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="destination .ass file (default: data/output/<media name>.ass)",
    )
    parser.add_argument(
        "--json", nargs="?", const=True, default=None, metavar="PATH",
        help="also write the intermediate transcript schema "
             "(bare flag: data/output/<media name>.json)",
    )
    parser.add_argument(
        "--lookup", action="store_true",
        help="identify the work from --title, look up its characters on the web, "
             "argue a colour for each and infer which speaker is which character "
             "from the dialogue (needs a Gemini API key, GEMINI_API_KEY; results "
             "are stored in the --json output and reused on later runs)",
    )
    _add_identity_arguments(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hearing-to-seeing",
        description="Turn speaker, timing and volume into a dynamic ASS subtitle file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="transcribe media and generate subtitles (regular entry point)",
    )
    _add_media_arguments(run_parser)
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
    _add_media_arguments(json_parser)
    json_parser.add_argument("transcript", help="transcript JSON to build subtitles from")

    lookup_parser = subparsers.add_parser(
        "lookup",
        help="identify the work, colour its characters and map speakers — from a transcript, no media",
        description=(
            "Runs only the identification steps: work lookup from the title, "
            "character colours, speaker → character mapping. Takes a transcript "
            "JSON (intermediate schema, segment export or character export) and "
            "writes the annotated JSON. No media is read, so the volume effect "
            "is not computed; an .ass can still be written with -o. Needs a "
            "Gemini API key (GEMINI_API_KEY)."
        ),
    )
    lookup_parser.add_argument("transcript", help="transcript JSON to annotate")
    lookup_parser.add_argument(
        "--json", default=None, metavar="PATH",
        help="where to write the annotated transcript "
             "(default: data/output/<transcript name>.lookup.json)",
    )
    lookup_parser.add_argument(
        "-o", "--output", default=None, metavar="ASS",
        help="also write an .ass (font size flat, since no audio was read)",
    )
    _add_identity_arguments(lookup_parser)

    return parser


def _make_llm_or_explain(title_known: bool) -> tuple[object | None, int]:
    """The configured model, or (None, exit code) with the reason printed.

    Checked before any work is done: a missing key should fail in the first
    second, not after STT has run for ten minutes.
    """
    try:
        if not configured():
            print(
                f"인물 검색에는 {api_key_var()} 가 필요합니다 (.env 에 설정하세요)",
                file=sys.stderr,
            )
            return None, 2
        if not title_known:
            print("인물 검색에는 검색할 --title 이 필요합니다", file=sys.stderr)
            return None, 2
        return make_llm(), 0
    except KnowledgeError as exc:
        print(exc, file=sys.stderr)
        return None, 2


def _report(transcript, speaker_map: dict[str, str], looked_up: bool) -> None:
    """Prints who each speaker was taken to be and what the lookup found."""
    # One line per speaker: who they were taken to be, how surely, and on
    # whose word — or why they were left unnamed. This is the whole of the
    # confirmation step until there is a UI for it.
    for label, profile in sorted(transcript.speaker_profiles.items()):
        if profile.name:
            line = f"  {label} = {profile.name}  [{profile.source}, {profile.confidence:.2f}]"
        else:
            line = f"  {label} = ?  [{profile.source}]"
        if profile.color:
            line += f"  {profile.color}"
        if profile.note:
            line += f"  — {profile.note}"
        print(line)
    if transcript.character_map:
        print("character_map: " + ", ".join(f"{k}={v}" for k, v in sorted(transcript.character_map.items())))
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

    work = transcript.work
    if work is not None:
        year = f" ({work.year})" if work.year else ""
        print(f"work: {work.title}{year} [confidence {work.confidence:.2f}]")
        if work.scene:
            print(f"scene: {work.scene}")
        for character in work.characters:
            hue = f"hue {character.hue:.0f}° ({character.hue_basis})" if character.hue is not None else f"no hue ({character.hue_basis})"
            print(f"  {character.name}: {hue}" + (f" — {character.hue_evidence}" if character.hue_evidence else ""))
        if work.sources:
            print("sources: " + ", ".join(work.sources[:5]))
        # Names the user mapped that the cast sheet does not know cannot pick
        # up a colour, and look exactly like ones that did.
        unknown = sorted(name for name in speaker_map.values() if work.find(name) is None)
        if unknown:
            print(
                f"warning: 인물 목록에 없는 이름 (색상 근거 없음): {', '.join(unknown)}",
                file=sys.stderr,
            )
    elif looked_up:
        print("work: (not identified — --title 이 비어 있음)", file=sys.stderr)


def _run_lookup(args: argparse.Namespace) -> int:
    """The `lookup` subcommand: identification from a transcript alone."""
    try:
        speaker_map = parse_speaker_map(args.speaker_map or "")
    except SpeakerMapError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not os.path.isfile(args.transcript):
        print(f"transcript file not found: {args.transcript}", file=sys.stderr)
        return 2
    if args.url is not None and not is_url(args.url.strip()):
        print(f"URL 형식이 아닙니다 (https:// 로 시작해야 합니다): {args.url}", file=sys.stderr)
        return 2

    # The title may be on the command line or inside the file (a segment
    # export carries video_title); which it is only shows once the file is
    # read, so the title check is left to the pipeline's own report.
    llm, code = _make_llm_or_explain(title_known=True)
    if llm is None:
        return code

    output_json = args.json or _default_output(args.transcript, ".lookup.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

    from hearing_to_seeing import pipeline

    try:
        transcript = pipeline.run_lookup(
            args.transcript, output_json, args.output,
            MediaInfo(title=args.title, source_url=(args.url or "").strip() or None),
            manual_mapping(speaker_map) if speaker_map else None,
            llm, args.fresh,
        )
    except KnowledgeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"wrote {output_json} ({len(transcript.words)} words, {len(transcript.speakers())} speakers)")
    if args.output:
        print(f"wrote {args.output}")
    _report(transcript, speaker_map, looked_up=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "lookup":
        return _run_lookup(args)

    try:
        # Argument shape before the filesystem, so a typo in the mapping is not
        # hidden behind an unrelated complaint about the media path.
        speaker_map = parse_speaker_map(args.speaker_map or "")
        source = resolve(args.media, title=args.title, url=args.url)
    except (SourceError, SpeakerMapError) as exc:
        print(exc, file=sys.stderr)
        return 2

    speaker_strategy = manual_mapping(speaker_map) if speaker_map else None

    llm = None
    if args.lookup:
        llm, code = _make_llm_or_explain(
            title_known=bool(args.title) or args.command == "from-json",
        )
        if llm is None:
            return code

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
                source.info, speaker_strategy, llm, args.fresh,
            )
        else:
            transcript = pipeline.run(
                media_path, output_ass, args.language, output_json,
                source.info, speaker_strategy, llm, args.fresh,
            )
    except (MediaToolError, KnowledgeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    speakers = len(transcript.speakers())
    print(f"wrote {output_ass} ({len(transcript.words)} words, {speakers} speakers)")
    if output_json:
        print(f"wrote {output_json}")
    _report(transcript, speaker_map, looked_up=args.lookup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
