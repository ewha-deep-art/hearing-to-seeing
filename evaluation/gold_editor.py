"""Gold-data editor: play the video and fix a transcript JSON beside it.

    uv run evaluation/gold_editor.py            # http://localhost:8765
    uv run evaluation/gold_editor.py --port 9000

Lists the transcript JSONs in data/output/{library,clips,golden,whisperx} and
the runs (<id>/transcript.json; clips are exported by export_clips.py). The
video is found by name (a library run whose meta title is the JSON's stem, or
data/input/<stem>.*); a clip opens its YouTube video; otherwise pick a local
file or paste a YouTube link in the page. Saving writes
data/output/golden/<stem>.json (an existing file there is kept as .bak first);
a clip's gold is named "<title> [<id>].json" so evaluate.py pairs it by id.
Standard library only.
"""

import argparse
import json
import mimetypes
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/output"
SOURCES = ["library", "clips", "golden", "whisperx"]
RUN_SOURCES = ["library", "clips"]
GOLDEN = OUTPUT / "golden"
INPUT = ROOT / "data/input"
PAGE = Path(__file__).with_name("gold_editor.html")
# Characters a file name cannot hold (here or on Windows) → full-width look-alikes.
UNSAFE = str.maketrans('/\\:*?"<>|', "／＼：＊？＂＜＞｜")
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".wav"}


def run_titles() -> dict[str, Path]:
    """Run title → run directory."""
    titles = {}
    for meta in sorted(m for src in RUN_SOURCES for m in (OUTPUT / src).glob("*/meta.json")):
        try:
            titles[json.loads(meta.read_text(encoding="utf-8"))["title"]] = meta.parent
        except (ValueError, KeyError):
            pass
    return titles


def list_files() -> list[dict]:
    files = []
    titles = {d: t for t, d in run_titles().items()}
    for src in SOURCES:
        for p in sorted((OUTPUT / src).glob("*.json")):
            files.append({"path": str(p.relative_to(OUTPUT)), "name": p.stem, "source": src})
        for p in sorted((OUTPUT / src).glob("*/transcript.json")):
            files.append({"path": str(p.relative_to(OUTPUT)), "source": src,
                          "name": titles.get(p.parent, p.parent.name)})
    return files


def safe_json(rel: str) -> Path | None:
    p = (OUTPUT / rel).resolve()
    if p.suffix == ".json" and OUTPUT.resolve() in p.parents and p.is_file():
        return p
    return None


def find_media(rel: str) -> Path | None:
    p = OUTPUT / rel
    if p.name == "transcript.json":
        stem, run = None, p.parent
    else:
        stem = p.stem
        run = run_titles().get(stem)
    if run:
        for name in ("media.mp4", "source.mp4"):
            if (run / name).is_file():
                return run / name
    if stem:
        for q in sorted(INPUT.rglob("*")):
            if q.stem == stem and q.suffix.lower() in VIDEO_EXT:
                return q
    return None


def run_meta(rel: str) -> dict:
    p = OUTPUT / rel
    if p.name != "transcript.json":
        return {}
    try:
        return json.loads((p.parent / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def gold_name(rel: str) -> str:
    p = OUTPUT / rel
    if p.name != "transcript.json":
        return p.name
    meta = run_meta(rel)
    if "title" not in meta:
        return p.parent.name + ".json"
    title = meta["title"].translate(UNSAFE)
    return f"{title} [{meta['id']}].json" if meta.get("youtube_id") else title + ".json"


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/files":
            self.send_json(list_files())
        elif url.path == "/api/json":
            p = safe_json(q.get("path", ""))
            if not p:
                return self.send_json({"error": "not found"}, 404)
            media = find_media(q["path"])
            gold = GOLDEN / gold_name(q["path"])
            self.send_json({
                "data": json.loads(p.read_text(encoding="utf-8")),
                "media": bool(media),
                "youtube_id": run_meta(q["path"]).get("youtube_id"),
                "gold_name": gold.name,
                "gold_exists": gold.is_file() and gold.resolve() != p,
            })
        elif url.path == "/api/media":
            media = find_media(unquote(q.get("path", "")))
            if not media:
                return self.send_error(404)
            self.send_file(media)
        else:
            self.send_error(404)

    def send_file(self, path: Path):
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            if a:
                start, end = int(a), int(b) if b else size - 1
            else:
                start = size - int(b)
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = Path(body["name"]).name
        if not name.endswith(".json"):
            name += ".json"
        GOLDEN.mkdir(parents=True, exist_ok=True)
        target = GOLDEN / name
        if target.exists():
            shutil.copy2(target, target.with_suffix(".json.bak"))
        target.write_text(json.dumps(body["data"], ensure_ascii=False, indent=2), encoding="utf-8")
        self.send_json({"saved": str(target.relative_to(ROOT))})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"gold editor: http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
