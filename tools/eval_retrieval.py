"""Benchmark the retriever: does the right passage come back, and does nothing come back for
questions that aren't medical?

Each positive case lists a substring that must appear in a returned passage's source, plus optional
substrings that must appear in its text. A case scores a hit if any of the top results satisfies both.
Negative cases are everyday chat that must retrieve nothing.

    uv run python tools/eval_retrieval.py [--verbose]

Needs Ollama running with the embedding model. Builds (or reuses) an index in data/eval-index.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baymax.rag import EMBED_MODEL, Embedder, Retriever  # noqa: E402

CASES = ROOT / "tests" / "eval" / "retrieval_cases.json"
HOST = "http://127.0.0.1:11434"


def satisfied(chunk, case) -> bool:
    return case["source"].lower() in (chunk.source + " " + chunk.text[:120]).lower() and all(
        t.lower() in chunk.text.lower() for t in case["text"]
    )


def main() -> None:
    verbose = "--verbose" in sys.argv
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    retriever = Retriever(
        Embedder(HOST, EMBED_MODEL), EMBED_MODEL, library_dir=ROOT / "data" / "empty-library",
        index_path=ROOT / "data" / "eval-index.json",
    )
    started = time.time()
    retriever.refresh()
    print(f"index ready: {len(retriever._index)} passages ({time.time() - started:.0f}s)")

    for mode in ("dense", "hybrid"):
        hits = rr = 0.0
        misses = []
        for case in cases["positive"]:
            results = retriever.retrieve(case["q"], mode=mode)
            rank = next((i for i, h in enumerate(results, 1) if satisfied(h.chunk, case)), None)
            if rank:
                hits += 1
                rr += 1 / rank
            else:
                misses.append((case["q"], [(h.chunk.source[:50], round(h.cosine, 2), round(h.bm25, 1)) for h in results[:3]]))
            if verbose:
                print(f"  [{mode}] rank={rank} cos/bm25={[(round(h.cosine,2), round(h.bm25,1)) for h in results[:2]]}  {case['q']}")
        false_fires = []
        for question in cases["negative"]:
            results = retriever.retrieve(question, mode=mode)
            if results:
                false_fires.append((question, [(h.chunk.source[:40], round(h.cosine, 2), round(h.bm25, 1)) for h in results[:2]]))
            elif verbose:
                print(f"  [{mode}] ok (nothing) {question}")
        n = len(cases["positive"])
        print(f"\n{mode.upper():7} hit@{4}: {hits:.0f}/{n} ({hits / n:.0%})   MRR: {rr / n:.2f}   "
              f"false fires on chat: {len(false_fires)}/{len(cases['negative'])}")
        for question, got in misses:
            print("   MISS:", question, got)
        for question, got in false_fires:
            print("   FALSE FIRE:", question, got)


if __name__ == "__main__":
    main()
