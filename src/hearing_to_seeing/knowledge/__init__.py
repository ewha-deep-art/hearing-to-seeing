"""Working out which work a clip is from and who appears in it.

The pipeline's lookup step. `title` reads the work out of a clip title,
`retrieve` fetches what Wikipedia and Namu wiki say about it, `characters`
has the model read those into a cast, and `lookup()` here runs the three in
order against a transcript, reusing whatever an earlier run already stored.
"""

import sys

from hearing_to_seeing.knowledge.characters import EXCERPT_CHARS, retrieve_characters
from hearing_to_seeing.knowledge.llm import LLM, KnowledgeError, configured, make_llm
from hearing_to_seeing.knowledge.retrieve import Document, retrieve
from hearing_to_seeing.knowledge.title import WorkQuery, parse_title
from hearing_to_seeing.schema import Transcript, WorkInfo

__all__ = [
    "LLM",
    "KnowledgeError",
    "configured",
    "make_llm",
    "WorkQuery",
    "Document",
    "lookup",
    "retrieve",
    "parse_title",
    "retrieve_characters",
    "transcript_excerpt",
]


def transcript_excerpt(transcript: Transcript, limit: int = EXCERPT_CHARS) -> str:
    """The opening dialogue, speaker by speaker, for recognising the scene by."""
    lines: list[str] = []
    current: str | None = None
    buffer: list[str] = []
    for word in transcript.words:
        if word.speaker != current:
            if buffer:
                lines.append(f"{current}: {' '.join(buffer)}")
            current, buffer = word.speaker, []
        buffer.append(word.text)
    if buffer:
        lines.append(f"{current}: {' '.join(buffer)}")
    return "\n".join(lines)[:limit]


def progress(message: str) -> None:
    """One line on stderr per slow step — each model call can take a minute."""
    print(f"… {message}", file=sys.stderr, flush=True)


def lookup(
    transcript: Transcript, llm: LLM, *, refresh: bool = False,
) -> WorkInfo | None:
    """Identifies the work and its cast, storing the result on the transcript.

    A transcript that already carries a `work` block is returned as is unless
    `refresh` is set: the cast of a film does not change between runs, and
    the searches are the slow, paid part of the pipeline. Returns None when
    there is no title to search for — the user is told so at the CLI, since
    nothing downstream can make up for it.
    """
    if transcript.work is not None and not refresh:
        return transcript.work

    progress(f"제목 파싱 중: {transcript.media.title!r}")
    query = parse_title(transcript.media, llm)
    if query is None:
        return None

    progress(f"문서 검색 중 (위키백과·나무위키): {query.title!r}")
    documents = retrieve(query.title)
    progress(
        "인물 정리 중: " + (", ".join(f"{d.source} {len(d.text)}자" for d in documents) or "문서 없음")
    )
    work = retrieve_characters(query, llm, documents, excerpt=transcript_excerpt(transcript))
    progress(f"인물 {len(work.characters)}명 확보")
    transcript.work = work
    return work
