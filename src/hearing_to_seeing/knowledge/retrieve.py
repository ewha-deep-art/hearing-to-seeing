"""Fetching what the web knows about a work — the retrieval half of RAG.

Two sources, read directly rather than through a search API:

  * **Korean Wikipedia**, through the MediaWiki API. Its search settles which
    page the title means, and its plaintext extract gives a reliable plot
    summary and cast list — short, but never wrong about who is in the film.
  * **Namu wiki**, fetched as a page. It has what Wikipedia lacks: a
    character section with a paragraph per person — temperament, relations,
    how they are addressed, sometimes what they wear. It is the source the
    colour and mapping steps actually feed on. It is also a fan wiki with no
    API, so it is best-effort: any failure leaves the Wikipedia text to carry
    the run alone.

Whole articles are too long to hand to a model on a free-tier token budget,
and most of an article (box office, awards, trivia) says nothing about the
characters. So each document is cut to the sections that do — overview,
plot, cast — under a character budget, and the model sees only those.

Nothing here calls a model. The documents go to `characters.py`, which asks
the model to read them; the URLs become the `sources` on the result, so what
the model was shown is on record.
"""

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

# What the request identifies itself as. Wikipedia asks for a descriptive
# agent; Namu wiki answers a browser-shaped one.
WIKIPEDIA_AGENT = "hearing-to-seeing/0.1 (student project; https://github.com/)"
BROWSER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
TIMEOUT = 20.0

WIKIPEDIA_API = "https://ko.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE = "https://ko.wikipedia.org/wiki/"
NAMU_PAGE = "https://namu.wiki/w/"

# How much of each document the model sees. Namu's character section for a
# well-known film runs to ten or twenty thousand characters; the budget keeps
# a whole cast in while cutting the trivia that follows it.
NAMU_BUDGET = 24_000
WIKIPEDIA_BUDGET = 8_000

# Sections worth keeping, by heading. Order is priority under the budget.
CHARACTER_HEADINGS = re.compile(r"등장\s*인물|인물|출연|배역|캐릭터|주요\s*인물|주변\s*인물")
CONTEXT_HEADINGS = re.compile(r"개요|줄거리|시놉시스|스토리|예고편")

# What a search hit must look like to count as the work itself, not an
# article that mentions it.
_WORK_QUALIFIERS = ("(영화)", "(드라마)", "(애니메이션)", "(연극)", "(뮤지컬)")

Fetch = Callable[[str, str], str]  # (url, user_agent) -> body text


@dataclass
class Document:
    source: str   # "위키백과" | "나무위키"
    title: str
    url: str
    text: str


class RetrievalError(RuntimeError):
    pass


def fetch_url(url: str, user_agent: str) -> str:
    """GETs a page as text. Raises RetrievalError on any failure."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Language": "ko"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        raise RetrievalError(f"HTTP {exc.code}: {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RetrievalError(f"{exc}: {url}") from exc


# --- Wikipedia ---------------------------------------------------------------

def _fold(text: str) -> str:
    return "".join(text.split()).casefold()


def wikipedia_search(title: str, fetch: Fetch = fetch_url, limit: int = 8) -> list[str]:
    """Page titles matching `title`, the work itself first.

    Two searches — with "영화" appended and bare — because the qualifier finds
    the film where a bare title finds the novel, but a title that is already
    unambiguous gets worse results with it.
    """
    seen: list[str] = []
    for query in (f"{title} 영화", title):
        params = urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": limit, "format": "json", "utf8": 1,
        })
        try:
            data = json.loads(fetch(f"{WIKIPEDIA_API}?{params}", WIKIPEDIA_AGENT))
        except (RetrievalError, json.JSONDecodeError):
            continue
        for hit in data.get("query", {}).get("search", []):
            name = hit.get("title", "")
            if name and name not in seen:
                seen.append(name)

    wanted = _fold(title)
    exact = [n for n in seen if _fold(_strip_qualifier(n)) == wanted]
    # The qualified article ("기생충 (영화)") over the bare one, which may be
    # the insect; then anything whose title contains the query.
    exact.sort(key=lambda n: (not n.endswith(_WORK_QUALIFIERS), n))
    partial = [n for n in seen if n not in exact and wanted in _fold(n)]
    return exact + partial


def _strip_qualifier(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", name)


def wikipedia_document(title: str, fetch: Fetch = fetch_url) -> Document | None:
    """The plaintext of one Wikipedia page, cut to the sections that matter."""
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
        "titles": title, "format": "json", "utf8": 1,
    })
    try:
        data = json.loads(fetch(f"{WIKIPEDIA_API}?{params}", WIKIPEDIA_AGENT))
    except (RetrievalError, json.JSONDecodeError):
        return None
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    text = page.get("extract") or ""
    if not text.strip() or "missing" in page:
        return None

    sections = _split_sections(text, re.compile(r"^={2,4}\s*(.+?)\s*={2,4}\s*$", re.M))
    kept = _select_sections(sections, WIKIPEDIA_BUDGET)
    resolved = page.get("title") or title
    return Document(
        source="위키백과",
        title=resolved,
        url=WIKIPEDIA_PAGE + urllib.parse.quote(resolved.replace(" ", "_")),
        text=kept,
    )


# --- Namu wiki ---------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
# A body heading as Namu renders it in text: "6.1. 주요 인물 [편집]". The
# table of contents repeats the titles without "[편집]", which is what keeps
# it from matching.
_NAMU_HEADING = re.compile(r"(?<!\S)(\d+(?:\.\d+)*)\.\s+([^\[\]\n]{1,60}?)\s*\[편집\]")

# The page Namu serves for a title it does not have.
_NAMU_MISSING = ("해당 문서를 찾을 수 없습니다", "문서가 존재하지 않습니다")


def namu_document(title: str, fetch: Fetch = fetch_url) -> Document | None:
    """Namu wiki's article for `title`, cut to overview, plot and characters.

    Tries the title as given and with a "(영화)" qualifier, since Namu puts
    the film under whichever it needed to disambiguate. Best-effort: None on
    any failure, and the caller carries on with Wikipedia alone.
    """
    candidates = [title]
    bare = _strip_qualifier(title)
    for variant in (f"{bare}(영화)", f"{bare}(드라마)", bare):
        if variant not in candidates:
            candidates.append(variant)

    for name in candidates:
        url = NAMU_PAGE + urllib.parse.quote(name)
        try:
            body = fetch(url, BROWSER_AGENT)
        except RetrievalError:
            continue
        if any(marker in body for marker in _NAMU_MISSING):
            continue
        text = _html_to_text(body)
        if len(text) < 500:
            continue
        sections = _namu_sections(text)
        if not sections:
            continue
        return Document(source="나무위키", title=name, url=url, text=_select_sections(sections, NAMU_BUDGET))
    return None


def _html_to_text(body: str) -> str:
    body = _SCRIPT.sub(" ", body)
    body = re.sub(r"<(br|/p|/div|/li|/h\d|/tr)[^>]*>", "\n", body, flags=re.I)
    text = html.unescape(_TAG.sub(" ", body))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _namu_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) pairs from a flattened Namu page, TOC excluded."""
    matches = list(_NAMU_HEADING.finditer(text))
    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


# --- section selection -------------------------------------------------------

def _split_sections(text: str, heading: re.Pattern) -> list[tuple[str, str]]:
    """(heading, body) pairs; the text before the first heading is the lead."""
    matches = list(heading.finditer(text))
    sections: list[tuple[str, str]] = []
    lead = text[: matches[0].start()].strip() if matches else text.strip()
    if lead:
        sections.append(("개요", lead))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(1).strip(), body))
    return sections


def _select_sections(sections: list[tuple[str, str]], budget: int) -> str:
    """Character sections first, then context, each trimmed to fit the budget.

    Characters before plot because the character sheet is what the colour
    and mapping steps argue from; a plot summary helps place the scene but
    is the first thing to lose when space runs out.
    """
    characters = [s for s in sections if CHARACTER_HEADINGS.search(s[0])]
    context = [s for s in sections if s not in characters and CONTEXT_HEADINGS.search(s[0])]
    chosen = characters + context
    if not chosen:
        chosen = sections[:3]

    out: list[str] = []
    remaining = budget
    for heading, body in chosen:
        if remaining <= 200:
            break
        piece = body if len(body) <= remaining else body[:remaining].rsplit(" ", 1)[0] + " …"
        out.append(f"## {heading}\n{piece}")
        remaining -= len(piece) + len(heading) + 4
    return "\n\n".join(out)


# --- the retriever -----------------------------------------------------------

def retrieve(title: str, fetch: Fetch = fetch_url) -> list[Document]:
    """Documents about the work called `title`, richest first.

    Wikipedia's search decides which page the title means; that page and
    Namu's article on the same name are fetched. Returns an empty list when
    nothing was found — the caller decides whether a cast can still be asked
    for from the model's own memory.
    """
    documents: list[Document] = []
    names = wikipedia_search(title, fetch)
    resolved = names[0] if names else title

    namu = namu_document(resolved, fetch) or (
        namu_document(title, fetch) if _fold(resolved) != _fold(title) else None
    )
    if namu:
        documents.append(namu)
    wiki = wikipedia_document(resolved, fetch) if names else None
    if wiki:
        documents.append(wiki)
    return documents
