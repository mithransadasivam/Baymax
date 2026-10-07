"""Turning the user's own files into passages small enough to retrieve one at a time.

A whole textbook can't be handed to a model that reads a few thousand tokens at once, and a
passage that's too long buries the one sentence that mattered. So files are read into plain text
and cut into chunks of a few sentences, each remembering which file (and page) it came from, so a
reply can say where it learned something.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

SUPPORTED = frozenset({".txt", ".md", ".pdf"})
CHUNK_CHARS = 800
OVERLAP_CHARS = 120

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_HYPHENATED_LINE_BREAK = re.compile(r"(\w)-\n(\w)")


class Chunk(NamedTuple):
    source: str
    """Where it came from, as it should be read aloud: ``paper.pdf, p. 12`` or ``notes.md``."""
    text: str


def find_files(root: Path) -> list[Path]:
    """Every readable file under ``root``, in a stable order. A missing folder is just empty."""
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        # Skips hidden files and the lock/temp files editors leave beside a document being edited.
        if path.is_file() and path.suffix.lower() in SUPPORTED and not path.name.startswith((".", "~"))
    )


def read_pages(path: Path) -> list[tuple[str, str]]:
    """``(source label, text)`` for each page of a PDF, or for the whole of a text file."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = _HYPHENATED_LINE_BREAK.sub(r"\1\2", page.extract_text() or "")
            pages.append((f"{path.name}, p. {number}", text))
        return pages
    return [(path.name, path.read_text(encoding="utf-8", errors="replace"))]


def chunk_file(path: Path) -> list[Chunk]:
    """All of a file's chunks. An unreadable or empty file yields none rather than an error: one
    corrupt PDF in a folder shouldn't stop every other document from being learned."""
    try:
        pages = read_pages(path)
    except Exception:
        return []
    return [Chunk(label, piece) for label, text in pages for piece in chunk_text(text)]


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Cut ``text`` into chunks of at most about ``size`` characters, on sentence boundaries where
    it can, with the tail of each one repeated at the start of the next so an idea that straddles
    a cut still appears whole in at least one of them."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in _units(text, size):
        if current and length + len(unit) + 1 > size:
            chunks.append(" ".join(current))
            current, length = _tail(current, overlap)
        current.append(unit)
        length += len(unit) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail(units: list[str], overlap: int) -> tuple[list[str], int]:
    kept: list[str] = []
    length = 0
    for unit in reversed(units):
        if length + len(unit) + 1 > overlap:
            break
        kept.insert(0, unit)
        length += len(unit) + 1
    return kept, length


def _units(text: str, size: int) -> Iterator[str]:
    """Paragraphs where they fit, else sentences, else (for a wall of text with no punctuation) words."""
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= size:
            yield paragraph
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            if len(sentence) <= size:
                yield sentence
            else:
                yield from _by_words(sentence, size)


def _by_words(text: str, size: int) -> Iterator[str]:
    line: list[str] = []
    length = 0
    for word in text.split():
        # A single "word" longer than the limit (a URL, a base64 blob) is cut where it stands.
        while len(word) > size:
            if line:
                yield " ".join(line)
                line, length = [], 0
            yield word[:size]
            word = word[size:]
        if line and length + len(word) + 1 > size:
            yield " ".join(line)
            line, length = [], 0
        line.append(word)
        length += len(word) + 1
    if line:
        yield " ".join(line)
