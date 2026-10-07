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
import math
import os
import re
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
TOP_K = 4
MAX_CONTEXT_CHARS = 3600
"""Ollama gives the model a 4096-token window by default, shared with the persona, the notes about
the user and the conversation so far -- a few thousand characters of passages is all that fits
before something older is silently pushed out of it."""

# Hybrid retrieval: a passage must be semantically close (cosine) or a strong exact-term match
# (BM25), then the two rankings are fused by rank. Calibrated against tests/eval/retrieval_cases.json
# with tools/eval_retrieval.py.
MIN_COSINE = 0.62
STRONG_BM25 = 12.0
_RRF_K = 60
BM25_WEIGHT = 0.5  # how much the keyword ranking counts relative to the embedding ranking
_CANDIDATES = 40
_VETTED_BONUS = 0.003
# Added to a vetted official source's fused score -- about a fifth of one rank position -- so that
# between two otherwise equal passages the FDA / NLM / NIH one wins, but never over a clearly better one.
_INDEX_VERSION = "v2"  # bump to force every file to be re-read and re-embedded

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
            if len(texts) > 500 and (start // _BATCH) % 25 == 24:
                print(f"Library: embedded {len(vectors)} of {len(texts)} passages...", flush=True)
        return _unit_length(np.asarray(vectors, dtype=np.float32))


def _unit_length(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.reshape(0, 0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


_WORDS = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset(
    "a an the of to in on for and or is are was were be been being with without this that these those it its as at "
    "by from into about over under how what why when where who which does do did can could would should will you your "
    "i me my we our they them their if so not no any some there then than also may might".split()
)
_VETTED_PREFIXES = ("MedlinePlus (NLM/NIH)", "FDA drug label", "NIH Office of Dietary Supplements")


def _terms(text: str) -> list[str]:
    """Lowercase words with a light plural stemmer, so "symptoms" finds "symptom"."""
    out = []
    for word in _WORDS.findall(text.lower()):
        if word in _STOP or len(word) < 2:
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        out.append(word)
    return out


def is_vetted(source: str) -> bool:
    """Whether a passage came from one of the official authorities the fetchers are limited to."""
    return source.startswith(_VETTED_PREFIXES)


class _BM25:
    """Okapi BM25 over every passage, for exact-term matches embeddings blur: a drug name, a unit, a lab."""

    K1, B = 1.5, 0.75

    def __init__(self, documents: list[list[str]]) -> None:
        self.n = len(documents)
        lengths = np.asarray([len(d) for d in documents], dtype=np.float32)
        self.avg = float(lengths.mean()) if self.n else 1.0
        self.lengths = lengths
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for index, document in enumerate(documents):
            counts: dict[str, int] = {}
            for term in document:
                counts[term] = counts.get(term, 0) + 1
            for term, count in counts.items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(index)
                tfs.append(count)
        self.postings = {t: (np.asarray(i, dtype=np.int32), np.asarray(f, dtype=np.float32)) for t, (i, f) in postings.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(query):
            if term not in self.postings:
                continue
            ids, tfs = self.postings[term]
            idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            norm = tfs + self.K1 * (1 - self.B + self.B * self.lengths[ids] / self.avg)
            out[ids] += idf * tfs * (self.K1 + 1) / norm
        return out


class _Hit(NamedTuple):
    fused: float
    cosine: float
    bm25: float
    chunk: Chunk


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
        # The source name is counted twice: it names the drug or topic the passage is about.
        self._bm25 = _BM25([_terms(c.source) * 2 + _terms(c.text) for c in self._chunks])
        self._vetted = np.asarray([is_vetted(c.source) for c in self._chunks], dtype=bool)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: np.ndarray, k: int) -> list[tuple[float, Chunk]]:
        """The ``k`` best matches, best first. All vectors are unit length, so a dot product is
        exactly the cosine of the angle between them."""
        if not self._chunks:
            return []
        scores = self._matrix @ query
        return [(float(scores[i]), self._chunks[i]) for i in np.argsort(-scores)[:k]]


def _hybrid(index: Index, vector: np.ndarray, text: str, k: int) -> list[_Hit]:
    """Fuse the embedding ranking and the keyword ranking (reciprocal rank fusion), then keep the
    ``k`` best that clear the relevance gate: close in meaning, or an unmistakable exact match."""
    cosine = index._matrix @ vector
    bm25 = index._bm25.scores(_terms(text))
    fused = np.zeros(len(index._chunks), dtype=np.float32)
    for scores, weight in ((cosine, 1.0), (bm25, BM25_WEIGHT)):
        for rank, i in enumerate(np.argsort(-scores)[:_CANDIDATES]):
            if scores[i] > 0:
                fused[i] += weight / (_RRF_K + rank + 1)
    fused += np.where(index._vetted, _VETTED_BONUS, 0.0)
    hits = []
    for i in np.argsort(-fused)[:_CANDIDATES]:
        if cosine[i] >= MIN_COSINE or bm25[i] >= STRONG_BM25:
            hits.append(_Hit(float(fused[i]), float(cosine[i]), float(bm25[i]), index._chunks[i]))
        if len(hits) == k:
            break
    return hits


# Words a label uses for a drug's class, so "ibuprofen with warfarin" finds warfarin's passage about
# "NSAIDs" even though it never says "ibuprofen".
_CLASS_WORDS = {
    **dict.fromkeys(["ibuprofen", "naproxen", "aspirin", "diclofenac", "meloxicam"], ["nsaid", "nonsteroidal", "salicylate"]),
    **dict.fromkeys(["sertraline", "fluoxetine", "citalopram", "escitalopram", "paroxetine"], ["ssri", "serotonin", "antidepressant"]),
    **dict.fromkeys(["venlafaxine", "duloxetine"], ["snri", "serotonin", "antidepressant"]),
    **dict.fromkeys(["tramadol", "oxycodone", "morphine", "codeine", "fentanyl"], ["opioid", "serotonin"]),
    **dict.fromkeys(["lisinopril", "enalapril", "ramipril", "benazepril"], ["ace inhibitor", "angiotensin"]),
    **dict.fromkeys(["warfarin", "apixaban", "rivaroxaban", "dabigatran"], ["anticoagulant", "blood thinner"]),
    **dict.fromkeys(["alprazolam", "lorazepam", "clonazepam", "diazepam"], ["benzodiazepine"]),
    **dict.fromkeys(["sildenafil", "tadalafil"], ["pde5", "phosphodiesterase"]),
    **dict.fromkeys(["isosorbide", "nitroglycerin"], ["nitrate"]),
}
_INTERACTION_HEADINGS = ("drug interactions", "warnings and precautions", "contraindications", "boxed warning")


def _drug_chunks(index: Index) -> dict[str, list[int]]:
    """For each medicine with an FDA label, the indices of its interaction / warning passages."""
    found = getattr(index, "_drug_map", None)
    if found is None:
        found = {}
        for i, chunk in enumerate(index._chunks):
            if not chunk.source.startswith("FDA drug label, "):
                continue
            title = chunk.text.split(":", 1)[0].strip()
            if not any(h in chunk.text[: len(title) + 60].lower() for h in _INTERACTION_HEADINGS):
                continue
            name = _terms(title.split()[0])
            if name:
                found.setdefault(name[0], []).append(i)
        index._drug_map = found
    return found


def _pair_hits(index: Index, text: str, limit: int = 2) -> list[_Hit]:
    """When a question names two or more medicines, the label passages of one that mention the other
    (or its drug class): the answer to "can I take X with Y?" lives there, in a form similar search misses."""
    drugs = _drug_chunks(index)
    named = [t for t in dict.fromkeys(_terms(text)) if t in drugs]
    if len(named) < 2:
        return []
    out: list[_Hit] = []
    for drug in named:
        others = [o for o in named if o != drug]
        needles = {w for o in others for w in [o, *[_terms(c)[0] if " " not in c else c for c in _CLASS_WORDS.get(o, [])]]}
        best: tuple[int, int] | None = None
        for i in drugs[drug]:
            body = index._chunks[i].text.lower()
            score = sum(body.count(n) for n in needles)
            if score and (best is None or score > best[0]):
                best = (score, i)
        if best:
            out.append(_Hit(1.0, 1.0, 0.0, index._chunks[best[1]]))
    return out[:limit]


def _curated_chunks(path: Path) -> list[Chunk]:
    # One research summary is one passage: each is already a self-contained few sentences, and
    # cutting one in two would separate a finding from the caveat that goes with it.
    entries = parse_entries(path.stem, path.read_text(encoding="utf-8"))
    if path.parent.name == "vetted":
        # Official sources keep their own citation (it names the authority, the topic or label, and the date).
        return [Chunk(entry.source or entry.title, f"{entry.title}. {entry.body}") for entry in entries]
    # Baymax's own writing is labelled as such, so it is never mistaken for an official statement.
    label = "Baymax notes, not clinically reviewed" if path.stem == "clinical" else "Baymax summary of published research"
    return [Chunk(f"{label}: {entry.source or entry.title}", f"{entry.title}. {entry.body}") for entry in entries]


def _stamp(path: Path) -> str:
    info = path.stat()
    return f"{_INDEX_VERSION}:{info.st_size}:{info.st_mtime_ns}"


def scan(library_dir: Path, knowledge_dir: Path = KNOWLEDGE_DIR) -> list[_Source]:
    """What there is to index right now, without reading any of it."""
    sources = []
    if knowledge_dir.is_dir():
        for path in sorted(knowledge_dir.rglob("*.md")):
            sources.append(_Source(f"notes:{path.relative_to(knowledge_dir).as_posix()}", _stamp(path), lambda path=path: _curated_chunks(path)))
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

    def retrieve(self, text: str, mode: str = "hybrid") -> list[_Hit]:
        """The passages that answer ``text``, best first, or [] if nothing in the library does.
        ``mode="dense"`` is the original embedding-only search, kept so it can be benchmarked."""
        index = self._index
        if index is None or not len(index) or len(text.split()) < _MIN_QUERY_WORDS:
            return []
        query = self._embed([_QUERY_PREFIX + text])[0]
        if mode == "hybrid":
            paired = _pair_hits(index, text)
            rest = _hybrid(index, query, text, TOP_K)
            seen = {h.chunk.text for h in paired}
            return (paired + [h for h in rest if h.chunk.text not in seen])[:TOP_K]
        results = index.search(query, TOP_K)
        floor = max(MIN_SIMILARITY, results[0][0] - CLOSE_TO_BEST)
        return [_Hit(sc, sc, 0.0, c) for sc, c in results if sc >= floor]

    def lookup(self, text: str) -> ResearchCheck:
        if len(text.split()) < _MIN_QUERY_WORDS:
            return NOT_CHECKED
        index = self._index
        if index is None or not len(index):
            return self._keywords(text)
        try:
            hits = self.retrieve(text)
        except Exception:
            return self._keywords(text)
        return _format([hit.chunk for hit in hits]) if hits else NOT_CHECKED

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
        "Reference passages from Baymax's medical library. Base your answer on these, name the "
        "authority in a few words (for example \"according to the FDA label\" or \"MedlinePlus says\"), "
        "and if a passage is labelled as Baymax's own notes, treat it as less authoritative. If they "
        f"don't answer the question, say so instead of guessing:\n\n{notes}",
        f"You answered that using notes from {sources}.",
    )
