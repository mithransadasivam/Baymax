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
