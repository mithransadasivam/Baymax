import numpy as np

from baymax import rag
from baymax.documents import Chunk
from baymax.rag import Index, _BM25, _Record, _hybrid, _terms, is_vetted


def build(chunks: list[Chunk], vectors: list[list[float]]) -> Index:
    matrix = rag._unit_length(np.asarray(vectors, dtype=np.float32))
    return Index("m", {"f": _Record("s", chunks, matrix)})


def test_terms_stems_plurals_and_drops_filler():
    assert _terms("What are the symptoms of strep throat?") == ["symptom", "strep", "throat"]


def test_bm25_ranks_the_passage_with_the_rare_word_first():
    docs = [_terms("metformin kidney eGFR contraindicated"), _terms("kidney stones pain"), _terms("headache rest")]
    scores = _BM25(docs).scores(_terms("metformin kidney"))
    assert int(np.argmax(scores)) == 0 and scores[2] == 0


def test_an_exact_drug_name_match_is_found_even_when_the_embedding_is_far(monkeypatch):
    monkeypatch.setattr(rag, "STRONG_BM25", 1.0)  # a two-passage corpus can't score as high as a real one
    chunks = [Chunk("FDA drug label, Warfarin (X), effective 2026-01-01", "Warfarin: Drug interactions. NSAIDs increase bleeding risk with warfarin."),
              Chunk("MedlinePlus (NLM/NIH), Headache", "Headache. Most headaches are harmless.")]
    index = build(chunks, [[0, 1], [1, 0]])
    hits = _hybrid(index, rag._unit_length(np.asarray([[1, 0.01]], dtype=np.float32))[0],
                   "warfarin warfarin warfarin warfarin interactions bleeding nsaids", 2)
    assert hits and "Warfarin" in hits[0].chunk.source


def test_small_talk_retrieves_nothing():
    chunks = [Chunk("MedlinePlus (NLM/NIH), Asthma", "Asthma. Asthma is a chronic lung disease.")]
    index = build(chunks, [[1, 0]])
    far = rag._unit_length(np.asarray([[0, 1]], dtype=np.float32))[0]
    assert _hybrid(index, far, "hello how are you", 4) == []


def test_official_sources_win_a_tie_over_the_assistants_own_notes():
    text = "Aspirin should not be given to children with viral illness because of Reye syndrome."
    chunks = [Chunk("Baymax notes, not clinically reviewed: x", text), Chunk("FDA drug label, Aspirin (Y), effective 2026-01-01", text)]
    index = build(chunks, [[1, 0], [1, 0]])
    hits = _hybrid(index, np.asarray([1, 0], dtype=np.float32), "aspirin reye syndrome children", 2)
    assert "FDA" in hits[0].chunk.source


def test_vetted_means_an_allowlisted_authority_prefix():
    assert is_vetted("FDA drug label, Ibuprofen (Z), effective 2025-01-01")
    assert is_vetted("MedlinePlus (NLM/NIH), Asthma")
    assert not is_vetted("Baymax notes, not clinically reviewed: AHA")
    assert not is_vetted("WebMD")


def test_a_question_naming_two_medicines_finds_each_ones_passage_about_the_other():
    chunks = [
        Chunk("FDA drug label, Warfarin Sodium (X), effective 2026-01-01",
              "Warfarin Sodium: Drug interactions. NSAIDs and aspirin increase the risk of bleeding with warfarin."),
        Chunk("FDA drug label, Warfarin Sodium (X), effective 2026-01-01",
              "Warfarin Sodium: Drug interactions (part 2 of 2). Vitamin K lowers the effect."),
        Chunk("FDA drug label, Ibuprofen (Y), effective 2026-01-01",
              "Ibuprofen: Warnings and precautions. Ask a doctor if you take a blood thinner such as warfarin."),
        Chunk("MedlinePlus (NLM/NIH), Headache", "Headache. Most headaches are harmless."),
    ]
    index = build(chunks, [[1, 0], [0, 1], [1, 1], [0, 0.1]])
    hits = rag._pair_hits(index, "Is it safe to take ibuprofen while on warfarin?")
    texts = " ".join(h.chunk.text for h in hits)
    assert "NSAIDs and aspirin" in texts and "blood thinner" in texts


def test_a_question_naming_one_medicine_is_not_routed():
    chunks = [Chunk("FDA drug label, Warfarin Sodium (X), effective 2026-01-01", "Warfarin Sodium: Drug interactions. Bleeding.")]
    assert rag._pair_hits(build(chunks, [[1, 0]]), "What is warfarin?") == []


class TestPacedAndResumableIndexing:
    def test_bulk_embedding_rests_between_batches_but_a_question_does_not(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(rag.time, "sleep", sleeps.append)
        monkeypatch.setattr(rag, "_PAUSE_SECONDS", 2.5)

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                pass

            def embed(self, model, input, keep_alive):
                return type("R", (), {"embeddings": [[1.0, 0.0] for _ in input]})()

        monkeypatch.setattr(rag.ollama, "Client", FakeClient)
        embedder = rag.Embedder("h")

        embedder(["a question"])
        assert sleeps == [], "a single question must never be paced"

        embedder([f"passage {i}" for i in range(100)])  # 4 batches of 32, so 3 rests (none after the last)
        assert sleeps == [2.5, 2.5, 2.5]

    def test_pausing_can_be_turned_off(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(rag.time, "sleep", sleeps.append)
        monkeypatch.setattr(rag, "_PAUSE_SECONDS", 0)

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                pass

            def embed(self, model, input, keep_alive):
                return type("R", (), {"embeddings": [[1.0, 0.0] for _ in input]})()

        monkeypatch.setattr(rag.ollama, "Client", FakeClient)
        rag.Embedder("h")([f"p{i}" for i in range(100)])

        assert sleeps == []

    def test_progress_is_reported_after_each_file_with_a_usable_partial_index(self):
        def load(label):
            return lambda: [Chunk(f"{label} src", f"{label} text one"), Chunk(f"{label} src", f"{label} text two")]

        sources = [rag._Source("a", "1", load("a")), rag._Source("b", "1", load("b"))]
        seen = []
        embed = lambda texts: rag._unit_length(np.ones((len(texts), 2), dtype=np.float32))  # noqa: E731

        final = rag.update_index(None, sources, embed, "m", on_progress=lambda idx: seen.append(len(idx)))

        assert seen == [2, 4], "after file a there are 2 passages, after file b there are 4"
        assert len(final) == 4

    def test_unchanged_files_are_not_embedded_again(self):
        sources = [rag._Source("a", "1", lambda: [Chunk("s", "text")])]
        calls = []

        def embed(texts):
            calls.append(len(texts))
            return rag._unit_length(np.ones((len(texts), 2), dtype=np.float32))

        first = rag.update_index(None, sources, embed, "m")
        rag.update_index(first, sources, embed, "m")

        assert calls == [1], "the second pass should reuse the saved vectors"
