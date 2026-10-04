"""Fetching what the web knows about a work — the retrieval half of RAG.

Two sources, read directly rather than through a search API:

  * **Korean Wikipedia**, through the MediaWiki API. Its search settles which
    page the title means, and its plaintext extract gives a reliable plot
    summary and cast list.
  * **Namu wiki**, fetched as a page. It has what Wikipedia lacks: a
    character section with a paragraph per person — temperament, relations,
    how they are addressed. It has no API, so it is best-effort.

Whole articles are too long for a free-tier token budget and mostly say
nothing about the characters, so each is cut to its overview, plot and cast
sections under a character budget. Nothing here calls a model.
"""

import html
import json
import re
import urllib.parse
import urllib.request

# Wikipedia asks for a descriptive agent; Namu wiki answers a browser-shaped one.
WIKIPEDIA_AGENT = "hearing-to-seeing/0.1 (student project; https://github.com/)"
BROWSER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
TIMEOUT = 20.0

WIKIPEDIA_API = "https://ko.wikipedia.org/w/api.php"
NAMU_PAGE = "https://namu.wiki/w/"

# How much of each document the model sees.
NAMU_BUDGET = 24_000
WIKIPEDIA_BUDGET = 8_000

# Sections worth keeping, by heading. Characters first under the budget.
CHARACTER_HEADINGS = re.compile(r"등장\s*인물|인물|출연|배역|캐릭터|주요\s*인물|주변\s*인물")
CONTEXT_HEADINGS = re.compile(r"개요|줄거리|시놉시스|스토리|예고편")

# A search hit with one of these is the work itself, not an article about it.
_WORK_QUALIFIERS = ("(영화)", "(드라마)", "(애니메이션)", "(연극)", "(뮤지컬)")

_WIKI_HEADING = re.compile(r"^={2,4}\s*(.+?)\s*={2,4}\s*$", re.M)
# A body heading as Namu renders it in text: "6.1. 주요 인물 [편집]". The
# table of contents repeats the titles without "[편집]", so it does not match.
_NAMU_HEADING = re.compile(r"(?<!\S)\d+(?:\.\d+)*\.\s+([^\[\]\n]{1,60}?)\s*\[편집\]")
_NAMU_MISSING = ("해당 문서를 찾을 수 없습니다", "문서가 존재하지 않습니다")
_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)


def fetch_url(url: str, user_agent: str) -> str | None:
    """GETs a page as text, or None on any failure."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Language": "ko"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except OSError:  # URLError, HTTPError and timeouts are all OSErrors
        return None


def _fetch_json(params: dict) -> dict:
    body = fetch_url(f"{WIKIPEDIA_API}?{urllib.parse.urlencode(params)}", WIKIPEDIA_AGENT)
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError:
        return {}


def _fold(text: str) -> str:
    return "".join(text.split()).casefold()


def _strip_qualifier(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", name)


def wikipedia_search(title: str) -> list[str]:
    """Page titles matching `title`, the work itself first.

    Searched with "영화" appended and bare: the qualifier finds the film where
    a bare title finds the novel, but hurts a title that is already unique.
    """
    seen: list[str] = []
    for query in (f"{title} 영화", title):
        data = _fetch_json({
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": 8, "format": "json", "utf8": 1,
        })
        for hit in data.get("query", {}).get("search", []):
            if hit.get("title") and hit["title"] not in seen:
                seen.append(hit["title"])

    wanted = _fold(title)
    exact = [n for n in seen if _fold(_strip_qualifier(n)) == wanted]
    # "기생충 (영화)" over the bare "기생충", which may be the insect.
    exact.sort(key=lambda n: (not n.endswith(_WORK_QUALIFIERS), n))
    partial = [n for n in seen if n not in exact and wanted in _fold(n)]
    return exact + partial


def wikipedia_document(title: str) -> str | None:
    """One Wikipedia page's plaintext, cut to the sections that matter."""
    data = _fetch_json({
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
        "titles": title, "format": "json", "utf8": 1,
    })
    page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
    text = page.get("extract") or ""
    if not text.strip() or "missing" in page:
        return None
    return _select_sections(_split_sections(text, _WIKI_HEADING, lead=True), WIKIPEDIA_BUDGET)


def namu_document(title: str) -> str | None:
    """Namu wiki's article for `title`, cut to overview, plot and characters.

    Also tries "(영화)"/"(드라마)" qualifiers, since Namu files the work under
    whichever it needed to disambiguate.
    """
    bare = _strip_qualifier(title)
    for name in dict.fromkeys([title, f"{bare}(영화)", f"{bare}(드라마)", bare]):
        body = fetch_url(NAMU_PAGE + urllib.parse.quote(name), BROWSER_AGENT)
        if not body or any(marker in body for marker in _NAMU_MISSING):
            continue
        text = _html_to_text(body)
        sections = _split_sections(text, _NAMU_HEADING) if len(text) >= 500 else []
        if sections:
            return _select_sections(sections, NAMU_BUDGET)
    return None


def _html_to_text(body: str) -> str:
    body = _SCRIPT.sub(" ", body)
    body = re.sub(r"<(br|/p|/div|/li|/h\d|/tr)[^>]*>", "\n", body, flags=re.I)
    text = html.unescape(_TAG.sub(" ", body))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _split_sections(text: str, heading: re.Pattern, lead: bool = False) -> list[tuple[str, str]]:
    """(heading, body) pairs. With `lead`, the text before the first heading is the overview."""
    matches = list(heading.finditer(text))
    sections: list[tuple[str, str]] = []
    if lead:
        intro = text[: matches[0].start()].strip() if matches else text.strip()
        if intro:
            sections.append(("개요", intro))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections.append((match.group(1).strip(), body))
    return sections


def _select_sections(sections: list[tuple[str, str]], budget: int) -> str:
    """Character sections first, then context, trimmed to fit the budget."""
    characters = [s for s in sections if CHARACTER_HEADINGS.search(s[0])]
    context = [s for s in sections if s not in characters and CONTEXT_HEADINGS.search(s[0])]
    chosen = characters + context or sections[:3]

    out: list[str] = []
    remaining = budget
    for heading, body in chosen:
        if remaining <= 200:
            break
        piece = body if len(body) <= remaining else body[:remaining].rsplit(" ", 1)[0] + " …"
        out.append(f"## {heading}\n{piece}")
        remaining -= len(piece) + len(heading) + 4
    return "\n\n".join(out)


def retrieve(title: str) -> str:
    """What Namu wiki and Wikipedia say about `title`, ready for a prompt.

    Wikipedia's search decides which page the title means. Returns "" when
    nothing was found — the model is then asked from memory.
    """
    names = wikipedia_search(title)
    resolved = names[0] if names else title
    namu = namu_document(resolved) or (
        namu_document(title) if _fold(resolved) != _fold(title) else None
    )
    wiki = wikipedia_document(resolved) if names else None

    documents = [("나무위키", namu), ("위키백과", wiki)]
    return "\n\n".join(
        f'<document source="{source}">\n{text}\n</document>'
        for source, text in documents if text
    )
