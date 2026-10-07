"""Retrieval by meaning, over Baymax's curated research notes and whatever the user drops into
their own library folder.

Keyword matching (baymax/research.py) can't tell that "heart attack" and "myocardial infarction"
are the same thing; embeddings can. Each passage is turned into a vector by a small free model
running in Ollama (nomic-embed-text), the question gets the same treatment, and the passages
whose vectors point the most nearly the same way are the ones handed to the model.

The expensive part -- embedding every passage -- happens once, in the background, and is cached on
disk keyed by each file's size and modification time, so restarting Baymax costs nothing and only
files that are new or changed are ever embedded again. Until the first index exists, or whenever
Ollama can't produce embeddings, questions fall back to the keyword search instead of failing.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import ollama

from baymax.documents import Chunk, chunk_file, find_files
from baymax.memory import KEEP_ALIVE
from baymax.research import KNOWLEDGE_DIR, NOT_CHECKED, ResearchCheck, parse_entries

EMBED_MODEL = "nomic-embed-text"

MIN_SIMILARITY = 0.70
"""Cosine similarity below this isn't "about" the question, however high it ranks: without a
floor, the three least-bad passages would be handed to the model for every message, "hello" included.
Measured against the real model: questions genuinely about the curated notes scored 0.73-0.86,
while everything unrelated -- jokes, weather, coffee -- topped out at 0.57. The vague middle is
where it errs towards silence: "how do airplanes fly" scored 0.69 against a paper on drone flocks,
and a question the notes miss just gets answered from what the model already knows."""
CLOSE_TO_BEST = 0.08
"""A passage must also be this near the best match's score to be included, so one clear answer
isn't padded out with two merely-on-topic ones that dilute it."""
TOP_K = 3
MAX_CONTEXT_CHARS = 2400
"""Ollama gives the model a 4096-token window by default, shared with the persona, the notes about
the user and the conversation so far -- a few thousand characters of passages is all that fits
before something older is silently pushed out of it."""

_BATCH = 32
# This model is trained to expect these, and ranks noticeably better with them than without.
_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "
_POLL_SECONDS = 5
_RETRY_SECONDS = 30
_MIN_QUERY_WORDS = 2

Embed = Callable[[list[str]], np.ndarray]


class Embedder:
    """Turns text into unit-length vectors using an embedding model served by Ollama."""

    def __init__(self, host: str, model: str = EMBED_MODEL) -> None:
        self._client = ollama.Client(host=host)
        self._model = model

    def __call__(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            # keep_alive on every request, as everywhere else Baymax talks to Ollama: each one
            # resets the unload timer to whatever it asked for, and the default is 5 minutes.
            response = self._client.embed(model=self._model, input=texts[start : start + _BATCH], keep_alive=KEEP_ALIVE)
            vectors.extend(response.embeddings)
        return _unit_length(np.asarray(vectors, dtype=np.float32))


def _unit_length(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.reshape(0, 0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class _Source(NamedTuple):
    key: str
    stamp: str
    load: Callable[[], list[Chunk]]


class _Record(NamedTuple):
    stamp: str
    chunks: list[Chunk]
    vectors: np.ndarray


class Index:
    """Every passage and its vector, grouped by the file they came from."""

    def __init__(self, model: str, records: dict[str, _Record]) -> None:
        self.model = model
        self.records = records
        self._chunks = [chunk for record in records.values() for chunk in record.chunks]
        self._matrix = np.vstack([record.vectors for record in records.values()]) if records else np.zeros((0, 0), np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: np.ndarray, k: int) -> list[tuple[float, Chunk]]:
        """The ``k`` best matches, best first. All vectors are unit length, so a dot product is
        exactly the cosine of the angle between them."""
        if not self._chunks:
            return []
        scores = self._matrix @ query
        return [(float(scores[i]), self._chunks[i]) for i in np.argsort(-scores)[:k]]


def _curated_chunks(path: Path) -> list[Chunk]:
    # One research summary is one passage: each is already a self-contained few sentences, and
    # cutting one in two would separate a finding from the caveat that goes with it.
    entries = parse_entries(path.stem, path.read_text(encoding="utf-8"))
    return [Chunk(entry.source or entry.title, f"{entry.title}. {entry.body}") for entry in entries]


def _stamp(path: Path) -> str:
    info = path.stat()
    return f"{info.st_size}:{info.st_mtime_ns}"


def scan(library_dir: Path, knowledge_dir: Path = KNOWLEDGE_DIR) -> list[_Source]:
    """What there is to index right now, without reading any of it."""
    sources = []
    if knowledge_dir.is_dir():
        for path in sorted(knowledge_dir.glob("*.md")):
            sources.append(_Source(f"notes:{path.name}", _stamp(path), lambda path=path: _curated_chunks(path)))
    for path in find_files(library_dir):
        key = f"library:{path.relative_to(library_dir).as_posix()}"
        sources.append(_Source(key, _stamp(path), lambda path=path: chunk_file(path)))
    return sources


def update_index(previous: Index | None, sources: list[_Source], embed: Embed, model: str) -> Index:
    """A new index for ``sources``, re-embedding only what's new or changed since ``previous``.
    Files that have gone are simply not carried over."""
    kept = previous.records if previous is not None and previous.model == model else {}
    records: dict[str, _Record] = {}
    pending: list[tuple[_Source, list[Chunk]]] = []
    for source in sources:
        old = kept.get(source.key)
        if old is not None and old.stamp == source.stamp:
            records[source.key] = old
        elif chunks := source.load():
            pending.append((source, chunks))
    if pending:
        vectors = embed([_DOCUMENT_PREFIX + chunk.text for _, chunks in pending for chunk in chunks])
        offset = 0
        for source, chunks in pending:
            records[source.key] = _Record(source.stamp, chunks, vectors[offset : offset + len(chunks)])
            offset += len(chunks)
    return Index(model, {source.key: records[source.key] for source in sources if source.key in records})


def save_index(index: Index, path: Path) -> None:
    """Writes ``path`` (passages and their sources) and its ``.npz`` twin (the vectors) -- each to
    a temporary file first, so Baymax being closed mid-save can't leave half of either behind."""
    meta = {
        "model": index.model,
        "files": {
            key: {"stamp": record.stamp, "chunks": [[chunk.source, chunk.text] for chunk in record.chunks]}
            for key, record in index.records.items()
        },
    }
    matrix = np.vstack([record.vectors for record in index.records.values()]) if index.records else np.zeros((0, 0), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors_path = path.with_suffix(".npz")
    temp_vectors, temp_meta = vectors_path.with_name(vectors_path.name + ".tmp"), path.with_name(path.name + ".tmp")
    with open(temp_vectors, "wb") as handle:
        np.savez(handle, vectors=matrix)
    temp_meta.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(temp_vectors, vectors_path)
    os.replace(temp_meta, path)


def load_index(path: Path, model: str) -> Index | None:
    """The saved index, or None if there isn't a usable one -- missing, corrupt, from a different
    embedding model (whose vectors mean something else entirely), or out of step with its twin."""
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["model"] != model:
            return None
        with np.load(path.with_suffix(".npz")) as data:
            matrix = data["vectors"]
        records: dict[str, _Record] = {}
        offset = 0
        for key, entry in meta["files"].items():
            chunks = [Chunk(source, text) for source, text in entry["chunks"]]
            records[key] = _Record(entry["stamp"], chunks, matrix[offset : offset + len(chunks)])
            offset += len(chunks)
        if offset != matrix.shape[0]:
            return None
        return Index(model, records)
    except Exception:
        return None


class Retriever:
    """The ``research_lookup`` Brain is given: ``lookup(text)`` answers questions from the index,
    while a background thread (``start()``) keeps that index in step with the files on disk."""

    def __init__(
        self,
        embed: Embed,
        model: str,
        library_dir: Path,
        index_path: Path,
        knowledge_dir: Path = KNOWLEDGE_DIR,
        fallback: Callable[[str], ResearchCheck] | None = None,
    ) -> None:
        self._embed = embed
        self._model = model
        self._library_dir = library_dir
        self._knowledge_dir = knowledge_dir
        self._index_path = index_path
        self._fallback = fallback
        # A saved index is usable immediately, before anything has been checked against the disk.
        self._index = load_index(index_path, model)
        self._seen: tuple | None = None
        self._failed_at: float | None = None
        self._complained = False
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        """Keep the index current for the rest of the process: indexing once, then noticing files
        added, changed or removed while Baymax is running, without being asked."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._watch, daemon=True, name="baymax-library")
            self._thread.start()

    def stop(self) -> None:
        self._stopping.set()

    def refresh(self) -> bool:
        """Bring the index up to date now. False if there was nothing to do, or embedding failed."""
        sources = scan(self._library_dir, self._knowledge_dir)
        signature = tuple((source.key, source.stamp) for source in sources)
        if signature == self._seen:
            return False
        if self._failed_at is not None and time.monotonic() - self._failed_at < _RETRY_SECONDS:
            return False
        try:
            index = update_index(self._index, sources, self._embed, self._model)
        except Exception as exc:
            self._failed_at = time.monotonic()
            if not self._complained:
                self._complained = True
                print(
                    f"Library: couldn't build the semantic index ({type(exc).__name__}); using keyword search for now. "
                    f"Is the embedding model installed? ollama pull {self._model}",
                    flush=True,
                )
            return False
        self._failed_at = None
        self._complained = False
        self._seen = signature
        self._index = index
        try:
            save_index(index, self._index_path)
        except OSError:
            pass  # still usable this session; it just gets rebuilt next time
        print(f"Library: {len(index)} passages from {len(index.records)} files indexed.", flush=True)
        return True

    def lookup(self, text: str) -> ResearchCheck:
        if len(text.split()) < _MIN_QUERY_WORDS:
            return NOT_CHECKED
        index = self._index
        if index is None or not len(index):
            return self._keywords(text)
        try:
            query = self._embed([_QUERY_PREFIX + text])[0]
        except Exception:
            return self._keywords(text)
        results = index.search(query, TOP_K)
        floor = max(MIN_SIMILARITY, results[0][0] - CLOSE_TO_BEST)
        hits = [chunk for score, chunk in results if score >= floor]
        return _format(hits) if hits else NOT_CHECKED

    def _keywords(self, text: str) -> ResearchCheck:
        return self._fallback(text) if self._fallback is not None else NOT_CHECKED

    def _watch(self) -> None:
        while not self._stopping.is_set():
            try:
                self.refresh()
            except Exception:
                pass  # a bad file or a full disk mustn't end the watcher for the rest of the session
            self._stopping.wait(_POLL_SECONDS)


def _format(hits: list[Chunk]) -> ResearchCheck:
    chosen: list[Chunk] = []
    used = 0
    for chunk in hits:
        if chosen and used + len(chunk.text) > MAX_CONTEXT_CHARS:
            break
        chosen.append(chunk)
        used += len(chunk.text)
    notes = "\n\n".join(f"{chunk.source}:\n{chunk.text}" for chunk in chosen)
    sources = ", ".join(dict.fromkeys(chunk.source for chunk in chosen))
    return ResearchCheck(
        f"Relevant notes from Baymax's library:\n\n{notes}",
        f"You answered that using notes from {sources}.",
    )
