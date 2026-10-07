"""Retrieval over a small local library of research notes, so domain questions (robotics,
medicine, CRISPR and gene editing) get answered from real papers instead of just whatever the
model half-remembers from training.

Not full papers, and not verbatim excerpts: each entry in baymax/knowledge/*.md is a short,
Baymax-authored summary of one paper's key finding, with a citation for provenance -- readable in
a spoken reply, and clear of the copyright question a wholesale copy of the source text would
raise.

No embeddings, no vector database: with a few dozen entries, plain keyword overlap between the
question and each entry's keywords, title, and body is both fast enough and precise enough, and
it keeps this free of a multi-gigabyte embedding-model dependency for a personal voice assistant.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_MIN_SCORE = 2  # below this, an overlap is coincidental rather than the question actually being about it
_MAX_RESULTS = 2

_ENTRY_HEADING = re.compile(r"(?m)^##\s+")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been being with without this that "
    "these those it its as at by from into about over under how what why when where who "
    "which does do did can could would should will you your".split()
)


class Entry(NamedTuple):
    """One research summary: ``topic`` is the file it came from (e.g. "crispr"), ``source`` its citation."""

    topic: str
    title: str
    source: str
    keywords: frozenset[str]
    body: str


class ResearchCheck(NamedTuple):
    """The outcome of checking the research library for one question -- shaped just like WebCheck."""

    results: str
    record: str


# Returned when nothing in the library is relevant enough to add to the prompt.
NOT_CHECKED = ResearchCheck("", "")


def _tokenize(text: str) -> set[str]:
    """The distinctive lowercase words in ``text``: no stopwords, nothing shorter than 3 letters."""
    return {word for word in _WORD.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS}


def load_entries(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[Entry]:
    """Every entry across every topic file in ``knowledge_dir``, or [] if it doesn't exist yet."""
    if not knowledge_dir.is_dir():
        return []
    entries: list[Entry] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        entries.extend(parse_entries(path.stem, path.read_text(encoding="utf-8")))
    return entries


def parse_entries(topic: str, text: str) -> list[Entry]:
    """Split a topic file into entries. Each starts at a ``## Title`` heading, followed by optional
    ``Source:`` and ``Keywords:`` lines, with everything else forming the body."""
    entries = []
    # [1:] skips whatever precedes the first heading (the file's own title and intro).
    for block in _ENTRY_HEADING.split(text)[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        source = ""
        keywords: frozenset[str] = frozenset()
        body_lines = []
        for line in lines[1:]:
            lowered = line.lower()
            if lowered.startswith("source:"):
                source = line.split(":", 1)[1].strip()
            elif lowered.startswith("keywords:"):
                keywords = frozenset(_tokenize(line.split(":", 1)[1]))
            else:
                body_lines.append(line)
        entries.append(Entry(topic=topic, title=title, source=source, keywords=keywords, body="\n".join(body_lines).strip()))
    return entries


def _score(question_tokens: set[str], entry: Entry) -> int:
    """How strongly a question's words overlap an entry; higher means more relevant."""
    # A keyword match is a deliberate signal the entry is about that concept; a body/title word
    # match might just be incidental phrasing, so it counts for less.
    keyword_hits = len(question_tokens & entry.keywords) * 2
    body_hits = len(question_tokens & _tokenize(entry.title + " " + entry.body))
    return keyword_hits + body_hits


class Library:
    """Loads baymax/knowledge/*.md once and answers lookups against it for the life of the process."""

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR) -> None:
        self._entries = load_entries(knowledge_dir)

    def lookup(self, text: str) -> ResearchCheck:
        """The best-matching entries for ``text`` as prompt context, or NOT_CHECKED if none match well."""
        if not self._entries:
            return NOT_CHECKED
        question_tokens = _tokenize(text)
        scored = [(_score(question_tokens, entry), entry) for entry in self._entries]
        scored = [pair for pair in scored if pair[0] >= _MIN_SCORE]
        if not scored:
            return NOT_CHECKED
        scored.sort(key=lambda pair: -pair[0])
        top = [entry for _, entry in scored[:_MAX_RESULTS]]
        notes = "\n\n".join(f"{entry.title} ({entry.source}):\n{entry.body}" for entry in top)
        # dict.fromkeys de-duplicates the citations while keeping their order.
        sources = ", ".join(dict.fromkeys(entry.source for entry in top))
        return ResearchCheck(
            f"Relevant research, from Baymax's own notes library:\n\n{notes}",
            f"You answered that using research notes on {sources}.",
        )
