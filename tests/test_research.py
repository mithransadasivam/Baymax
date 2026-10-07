from baymax.research import NOT_CHECKED, Library, load_entries


_ROBOTICS_MD = """\
## Sim-to-real transfer for legged locomotion
Source: Example et al., 2021 -- arXiv:2100.00001
Keywords: locomotion, sim-to-real, reinforcement learning, quadruped

Policies trained in simulation with randomized dynamics transferred to a real quadruped robot
without further tuning, walking over uneven terrain. The gap between simulated and real-world
friction remained the largest source of failure, and the result hasn't been replicated on
bipedal platforms.

## Swarm coordination without central control
Source: Example & Someone, 2019 -- arXiv:1900.00002
Keywords: swarm robotics, decentralized control, emergent behavior

A colony of simple robots reached consensus on a shared task using only local communication,
with no central coordinator. Scaling past a few hundred units introduces communication delays
that haven't yet been solved.
"""

_MEDICINE_MD = """\
## CRISPR base editing without double-strand breaks
Source: Researcher et al., 2022 -- PMID:1234567
Keywords: crispr, base editing, gene editing, dna repair

Base editors convert one DNA base to another directly, without cutting both strands of the
double helix. This avoids the unpredictable insertions and deletions double-strand breaks can
cause, though off-target editing at similar sequences elsewhere in the genome is still measurable
and not fully solved.
"""


def _write_knowledge(tmp_path):
    (tmp_path / "robotics.md").write_text(_ROBOTICS_MD, encoding="utf-8")
    (tmp_path / "medicine.md").write_text(_MEDICINE_MD, encoding="utf-8")
    return tmp_path


class TestLoadEntries:
    def test_parses_every_entry_across_every_topic_file(self, tmp_path):
        entries = load_entries(_write_knowledge(tmp_path))

        assert len(entries) == 3
        assert {entry.topic for entry in entries} == {"robotics", "medicine"}

    def test_a_missing_knowledge_directory_yields_no_entries_rather_than_crashing(self, tmp_path):
        assert load_entries(tmp_path / "does-not-exist") == []

    def test_parses_the_title_source_and_keywords(self, tmp_path):
        entries = load_entries(_write_knowledge(tmp_path))

        crispr_entry = next(entry for entry in entries if "base editing" in entry.title.lower())
        assert crispr_entry.source == "Researcher et al., 2022 -- PMID:1234567"
        assert "crispr" in crispr_entry.keywords
        assert "double-strand breaks" in crispr_entry.body


class TestLibraryLookup:
    def test_a_matching_question_returns_the_relevant_entry(self, tmp_path):
        library = Library(_write_knowledge(tmp_path))

        check = library.lookup("How does CRISPR base editing avoid double-strand breaks?")

        assert "base editing" in check.results.lower()
        assert "PMID:1234567" in check.results
        assert "research notes" in check.record.lower()

    def test_an_unrelated_question_is_not_checked(self, tmp_path):
        library = Library(_write_knowledge(tmp_path))

        assert library.lookup("What's the weather today?") == NOT_CHECKED

    def test_an_empty_library_never_matches_anything(self, tmp_path):
        library = Library(tmp_path / "empty")

        assert library.lookup("Tell me about CRISPR base editing.") == NOT_CHECKED

    def test_at_most_two_entries_come_back_even_when_more_match(self, tmp_path):
        knowledge = _write_knowledge(tmp_path)
        library = Library(knowledge)

        check = library.lookup("Tell me about robotics research: locomotion and swarm robotics.")

        # Both robotics entries are plausible matches; the cap keeps a reply from turning into a
        # wall of citations no one asked to hear read aloud.
        assert check.results.count("arXiv:") <= 2
