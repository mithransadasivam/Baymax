"""Shared plumbing for the knowledge fetchers: downloading only from an allowlist of official
hosts, cleaning HTML to plain text, cutting text into passages, and writing entries in the format
baymax/research.py reads -- while recording where every passage came from.

Vetting policy, enforced here rather than promised in a README:

* Only hosts in ALLOWED_HOSTS can be fetched at all. Each is a US government health authority whose
  content is public domain: NLM (MedlinePlus), FDA (drug labels), NIH (Office of Dietary Supplements).
* Every entry carries a ``Source:`` line naming the authority, plus ``Retrieved:`` and ``URL:``
  lines, and a manifest records a SHA-256 of the raw download. ``vet.py`` re-checks all of it.
* Text is passed through verbatim apart from whitespace and HTML removal: nothing is paraphrased,
  so nothing can drift from what the authority published.
"""

from __future__ import annotations

import datetime
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VETTED_DIR = ROOT / "baymax" / "knowledge" / "vetted"
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = VETTED_DIR / "MANIFEST.json"

ALLOWED_HOSTS = {
    "medlineplus.gov": "MedlinePlus (NLM/NIH)",
    "api.fda.gov": "FDA drug label",
    "ods.od.nih.gov": "NIH Office of Dietary Supplements",
}
USER_AGENT = "BaymaxKnowledgeFetcher/1.0 (personal offline health assistant)"
PASSAGE_CHARS = 1300


def today() -> str:
    return datetime.date.today().isoformat()


def fetch(url: str, *, retries: int = 3, pause: float = 0.0) -> bytes:
    """GET ``url``, refusing any host that isn't on the allowlist."""
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"refusing to fetch from {host!r}: not an allowlisted authority")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if pause:
                time.sleep(pause)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except Exception as exc:  # noqa: BLE001 - network flakiness: retry, then report the last failure
            last = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def fetch_cached(url: str, name: str) -> bytes:
    """Like fetch, but kept under data/raw so a re-run doesn't re-download a large file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    if path.exists():
        return path.read_bytes()
    data = fetch(url)
    path.write_bytes(data)
    return data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _TextExtractor(HTMLParser):
    """HTML to plain text: paragraphs and list items become their own lines, links keep their text."""

    _BLOCKS = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "li":
            self.parts.append("\n- ")
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCKS or tag == "li":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup)
    text = "".join(parser.parts)
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line and line != "-")


def clean_label_text(text: str) -> str:
    """FDA label sections arrive as one run-on string with section numbers and stray symbols."""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_SENTENCE = re.compile(r"(?<=[.!?:;])\s+(?=[A-Z0-9(\-])")


def passages(text: str, size: int = PASSAGE_CHARS) -> list[str]:
    """Cut ``text`` into passages of about ``size`` characters, on line then sentence boundaries."""
    units: list[str] = []
    for line in text.splitlines():
        if len(line) <= size:
            units.append(line)
        else:
            units.extend(_SENTENCE.split(line))
    out: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > size:  # an unbroken run: split on words as a last resort
            words, piece = unit.split(" "), ""
            for word in words:
                if len(piece) + len(word) + 1 > size and piece:
                    out.append(piece)
                    piece = ""
                piece = f"{piece} {word}".strip()
            unit = piece
        if current and len(current) + len(unit) + 1 > size:
            out.append(current)
            current = unit
        else:
            current = f"{current}\n{unit}".strip() if current else unit
    if current:
        # A runt tail ("NIH: National Cancer Institute") is folded into the passage before it.
        if out and len(current) < 120:
            out[-1] = f"{out[-1]}\n{current}"
        else:
            out.append(current)
    return out


def entry(title: str, source: str, url: str, retrieved: str, keywords: list[str], body: str) -> str:
    """One knowledge entry. Lines starting with '#' inside a body would split it, so they're neutralised."""
    body = "\n".join(line.lstrip("#").strip() if line.startswith("#") else line for line in body.splitlines())
    kw = ", ".join(dict.fromkeys(k.strip().lower() for k in keywords if k and k.strip()))
    return f"## {title}\nSource: {source}\nURL: {url}\nRetrieved: {retrieved}\nKeywords: {kw}\n\n{body}\n"


def write_topic_file(name: str, header: str, entries: list[str]) -> Path:
    VETTED_DIR.mkdir(parents=True, exist_ok=True)
    path = VETTED_DIR / name
    path.write_text(header.rstrip() + "\n\n" + "\n".join(entries), encoding="utf-8")
    return path


def update_manifest(key: str, record: dict) -> None:
    VETTED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest[key] = record
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
