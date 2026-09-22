"""The retriever, against canned Wikipedia and Namu responses."""

import json
import urllib.parse

import pytest

from hearing_to_seeing.knowledge.retrieve import (
    NAMU_BUDGET,
    RetrievalError,
    _select_sections,
    namu_document,
    retrieve,
    wikipedia_document,
    wikipedia_search,
)


def _wiki_search_body(*titles: str) -> str:
    return json.dumps({"query": {"search": [{"title": t} for t in titles]}})


def _wiki_extract_body(title: str, extract: str) -> str:
    return json.dumps({"query": {"pages": {"1": {"pageid": 1, "title": title, "extract": extract}}}})


# Long enough to clear the retriever's "this is a JavaScript shell" size check.
NAMU_HTML = """<html><head><script>var x=1;</script><style>.a{}</style></head><body>
<div>목차 1. 개요 2. 등장인물 2.1. 주요 인물 3. 흥행</div>
<h2>1. 개요 <a>[편집]</a></h2><p>2026년 개봉한 사극 영화. 조선 제6대 국왕 단종과 호장 엄흥도의 이야기를 각색하여 다룬 사극 영화이며, 장항준 감독의 여섯 번째 장편 연출작이다. 조선 제6대 국왕 단종과 호장 엄흥도의 이야기를 각색하여 다룬 사극 영화이며, 장항준 감독의 여섯 번째 장편 연출작이다. 조선 제6대 국왕 단종과 호장 엄흥도의 이야기를 각색하여 다룬 사극 영화이며, 장항준 감독의 여섯 번째 장편 연출작이다. 조선 제6대 국왕 단종과 호장 엄흥도의 이야기를 각색하여 다룬 사극 영화이며, 장항준 감독의 여섯 번째 장편 연출작이다. 조선 제6대 국왕 단종과 호장 엄흥도의 이야기를 각색하여 다룬 사극 영화이며, 장항준 감독의 여섯 번째 장편 연출작이다. 조선 제6대 국왕 단종과 호장 엄흥도의 이야기를 각색하여 다룬 사극 영화이며, 장항준 감독의 여섯 번째 장편 연출작이다. </p>
<h2>2. 등장인물 <a>[편집]</a></h2>
<h3>2.1. 주요 인물 <a>[편집]</a></h3><p>엄흥도 (배우: 유해진) 광천골 촌장.</p><p>이홍위 (배우: 박지훈) 유배 온 어린 왕.</p>
<h2>3. 흥행 <a>[편집]</a></h2><p>천만 관객을 넘었다. 이 문장은 인물과 무관하다.</p>
</body></html>"""


class FakeFetch:
    """Serves canned bodies by URL substring; records what was asked for."""

    def __init__(self, routes: dict[str, str]):
        self.routes = routes
        self.urls: list[str] = []

    def __call__(self, url: str, user_agent: str) -> str:
        self.urls.append(urllib.parse.unquote(url))
        for needle, body in self.routes.items():
            if needle in urllib.parse.unquote(url):
                if body is None:
                    raise RetrievalError(f"HTTP 404: {url}")
                return body
        raise RetrievalError(f"HTTP 404: {url}")


# --- Wikipedia ---------------------------------------------------------------

def test_wikipedia_search_prefers_the_qualified_article_for_the_work():
    fetch = FakeFetch({"list=search": _wiki_search_body("기생충", "기생충 (영화)", "고래 기생충", "기생충 (동음이의)")})
    assert wikipedia_search("기생충", fetch) == ["기생충 (영화)", "기생충", "기생충 (동음이의)", "고래 기생충"]


def test_wikipedia_search_survives_a_failed_request():
    assert wikipedia_search("기생충", FakeFetch({})) == []


def test_wikipedia_document_keeps_cast_and_plot_and_drops_the_rest():
    extract = (
        "《왕과 사는 남자》는 2026년 영화이다.\n\n"
        "== 줄거리 ==\n어린 왕이 유배를 온다.\n\n"
        "== 주요 출연진 ==\n유해진: 엄흥도 역\n박지훈: 이홍위 역\n\n"
        "== 흥행 ==\n천만 관객.\n\n"
        "== 각주 ==\n1. 기사"
    )
    fetch = FakeFetch({"prop=extracts": _wiki_extract_body("왕과 사는 남자", extract)})
    document = wikipedia_document("왕과 사는 남자", fetch)

    assert document.source == "위키백과"
    assert document.url == "https://ko.wikipedia.org/wiki/%EC%99%95%EA%B3%BC_%EC%82%AC%EB%8A%94_%EB%82%A8%EC%9E%90"
    assert document.text.index("## 주요 출연진") < document.text.index("## 개요") < document.text.index("## 줄거리")
    assert "엄흥도 역" in document.text
    assert "천만 관객" not in document.text and "각주" not in document.text


def test_wikipedia_document_is_none_for_a_missing_page():
    body = json.dumps({"query": {"pages": {"-1": {"title": "없는영화", "missing": ""}}}})
    assert wikipedia_document("없는영화", FakeFetch({"prop=extracts": body})) is None


# --- Namu --------------------------------------------------------------------

def test_namu_document_extracts_body_sections_not_the_table_of_contents():
    fetch = FakeFetch({"namu.wiki/w/왕과 사는 남자": NAMU_HTML})
    document = namu_document("왕과 사는 남자", fetch)

    assert document.source == "나무위키"
    assert "## 주요 인물" in document.text
    assert "엄흥도 (배우: 유해진)" in document.text
    assert "## 개요" in document.text and "사극 영화" in document.text
    assert "천만 관객" not in document.text           # 흥행 is neither cast nor context
    assert "var x=1" not in document.text             # scripts stripped


def test_namu_document_tries_the_film_qualifier_when_the_bare_title_is_missing():
    missing = "<html><body>해당 문서를 찾을 수 없습니다</body></html>"
    fetch = FakeFetch({"namu.wiki/w/기생충(영화)": NAMU_HTML, "namu.wiki/w/기생충": missing})
    document = namu_document("기생충", fetch)
    assert document is not None
    assert document.title == "기생충(영화)"
    assert fetch.urls[0].endswith("/w/기생충") and fetch.urls[1].endswith("/w/기생충(영화)")


def test_namu_document_is_none_when_nothing_usable_comes_back():
    assert namu_document("없는영화", FakeFetch({})) is None
    assert namu_document("없는영화", FakeFetch({"namu.wiki": "<html><body>짧음</body></html>"})) is None


# --- selection / budget ------------------------------------------------------

def test_select_sections_puts_characters_first_and_respects_the_budget():
    sections = [("개요", "a" * 100), ("등장인물", "b" * 300), ("흥행", "c" * 100), ("줄거리", "d" * 100)]
    text = _select_sections(sections, budget=350)
    assert text.startswith("## 등장인물")
    assert "cccc" not in text
    assert len(text) <= 350 + 40                      # headings and the ellipsis are the slack


def test_select_sections_falls_back_to_the_first_sections_when_none_match():
    text = _select_sections([("서론", "x"), ("본론", "y"), ("결론", "z"), ("부록", "w")], budget=1000)
    assert "## 서론" in text and "## 결론" in text and "부록" not in text


# --- retrieve ----------------------------------------------------------------

def test_retrieve_returns_namu_then_wikipedia_for_the_resolved_title():
    fetch = FakeFetch({
        "list=search": _wiki_search_body("기생충 (영화)", "기생충"),
        "prop=extracts": _wiki_extract_body("기생충 (영화)", "《기생충》은 2019년 영화.\n\n== 줄거리 ==\n반지하."),
        "namu.wiki/w/기생충(영화)": NAMU_HTML,
    })
    documents = retrieve("기생충", fetch)
    assert [d.source for d in documents] == ["나무위키", "위키백과"]
    assert documents[0].title == "기생충(영화)"
    assert documents[1].title == "기생충 (영화)"


def test_retrieve_returns_what_it_could_get_and_never_raises():
    fetch = FakeFetch({"list=search": _wiki_search_body("왕과 사는 남자"), "namu.wiki/w/왕과 사는 남자": NAMU_HTML})
    documents = retrieve("왕과 사는 남자", fetch)
    assert [d.source for d in documents] == ["나무위키"]     # Wikipedia extract failed, Namu carried on
    assert retrieve("아무것도없음", FakeFetch({})) == []
