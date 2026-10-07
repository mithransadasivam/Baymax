from types import SimpleNamespace

import pytest

from baymax import memory as memory_module
from baymax.memory import MAX_NOTES_IN_PROMPT, Memory, parse_notes, worth_noting


@pytest.fixture
def model(monkeypatch):
    """Replaces Ollama with a stub whose note-taking reply each test sets, recording what it was asked."""
    state = {"reply": "NONE", "asked": [], "error": None}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def chat(self, model, messages, options, **kwargs):
            state["asked"].append(messages[-1]["content"])
            if state["error"]:
                raise state["error"]
            return SimpleNamespace(message=SimpleNamespace(content=state["reply"]))

    monkeypatch.setattr(memory_module.ollama, "Client", FakeClient)
    return state


def open_memory(path, **kwargs) -> Memory:
    return Memory(path, host="h", model="m", **kwargs)


class TestLearn:
    def test_a_lasting_fact_is_written_to_the_file(self, model, tmp_path):
        model["reply"] = "The user's sister's birthday is on March 3rd."
        memory = open_memory(tmp_path / "memory.txt")

        assert memory.learn("Remember that my sister's birthday is on March 3rd.") == [
            "The user's sister's birthday is on March 3rd."
        ]
        assert (tmp_path / "memory.txt").read_text(encoding="utf-8") == "The user's sister's birthday is on March 3rd.\n"

    def test_notes_are_still_there_in_tomorrow_s_session(self, model, tmp_path):
        model["reply"] = "The user's name is Mithran.\nThe user lives in Las Vegas."
        open_memory(tmp_path / "memory.txt").learn("My name is Mithran and I live in Las Vegas.")

        tomorrow = open_memory(tmp_path / "memory.txt")

        assert tomorrow.notes == ["The user's name is Mithran.", "The user lives in Las Vegas."]
        assert "The user lives in Las Vegas." in tomorrow.prompt()

    def test_questions_not_about_the_user_never_reach_the_model(self, model, tmp_path):
        memory = open_memory(tmp_path / "memory.txt")

        assert memory.learn("What is the temp in Chennai?") == []
        assert model["asked"] == []

    def test_nothing_is_saved_when_the_model_finds_nothing_lasting(self, model, tmp_path):
        model["reply"] = "NONE"
        memory = open_memory(tmp_path / "memory.txt")

        assert memory.learn("I'm a bit tired today.") == []
        assert not (tmp_path / "memory.txt").exists()

    def test_a_fact_already_known_is_not_saved_twice(self, model, tmp_path):
        (tmp_path / "memory.txt").write_text("The user lives in Las Vegas.\n", encoding="utf-8")
        model["reply"] = "The user lives in las vegas\nThe user owns a dog named Rex."
        memory = open_memory(tmp_path / "memory.txt")

        assert memory.learn("I live in Las Vegas with my dog Rex.") == ["The user owns a dog named Rex."]
        assert memory.notes == ["The user lives in Las Vegas.", "The user owns a dog named Rex."]

    def test_new_notes_are_announced(self, model, tmp_path):
        model["reply"] = "The user is allergic to peanuts."
        announced = []
        memory = open_memory(tmp_path / "memory.txt", on_noted=announced.append)

        memory.learn("I'm allergic to peanuts.")

        assert announced == [["The user is allergic to peanuts."]]

    def test_a_hand_edited_file_without_a_final_newline_stays_one_note_per_line(self, model, tmp_path):
        (tmp_path / "memory.txt").write_text("The user prefers Celsius.", encoding="utf-8")
        model["reply"] = "The user works as a nurse."
        memory = open_memory(tmp_path / "memory.txt")

        memory.learn("I work as a nurse.")

        assert (tmp_path / "memory.txt").read_text(encoding="utf-8").splitlines() == [
            "The user prefers Celsius.",
            "The user works as a nurse.",
        ]

    def test_a_model_failure_costs_the_note_not_the_conversation(self, model, tmp_path):
        model["error"] = ConnectionError("Ollama went away")
        memory = open_memory(tmp_path / "memory.txt")

        assert memory.learn("My name is Mithran.") == []

    def test_the_memory_folder_is_created_on_first_note(self, model, tmp_path):
        model["reply"] = "The user's name is Mithran."
        memory = open_memory(tmp_path / "new" / "folder" / "memory.txt")

        memory.learn("My name is Mithran.")

        assert (tmp_path / "new" / "folder" / "memory.txt").exists()


class TestPrompt:
    def test_no_notes_means_nothing_is_added_to_the_prompt(self, model, tmp_path):
        assert open_memory(tmp_path / "memory.txt").prompt() == ""

    def test_only_the_most_recent_notes_fit_in_the_prompt(self, model, tmp_path):
        notes = [f"The user likes number {n}." for n in range(MAX_NOTES_IN_PROMPT + 10)]
        (tmp_path / "memory.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

        prompt = open_memory(tmp_path / "memory.txt").prompt()

        assert "number 9." not in prompt
        assert f"number {MAX_NOTES_IN_PROMPT + 9}." in prompt

    def test_blank_lines_in_the_file_are_ignored(self, model, tmp_path):
        (tmp_path / "memory.txt").write_text("\nThe user is Mithran.\n\n", encoding="utf-8")

        assert open_memory(tmp_path / "memory.txt").notes == ["The user is Mithran."]


class TestParseNotes:
    def test_none_means_no_notes(self):
        assert parse_notes("NONE") == []

    def test_chatter_and_bullets_around_the_notes_are_ignored(self):
        reply = "Here are the facts:\n- The user lives in Chennai\n* The user has a cat."
        assert parse_notes(reply) == ["The user lives in Chennai.", "The user has a cat."]


class TestWorthNoting:
    @pytest.mark.parametrize(
        "question",
        [
            "Where do I live?",
            "where do I live",
            "When is my sister's birthday?",
            "Do you remember my name?",
            "Tell me about my sister.",
        ],
    )
    def test_questions_about_the_user_are_never_shown_to_the_note_taker(self, question):
        # Shown "Where do I live?", the model once noted that the user lives in New York City.
        assert worth_noting(question) == ""

    def test_a_question_that_asks_for_something_to_be_remembered_still_counts(self):
        text = "Can you remember that my mom's birthday is June 9th?"

        assert worth_noting(text) == text

    def test_only_the_statements_in_a_mixed_message_are_kept(self):
        assert worth_noting("I live in Chennai. What is the weather like?") == "I live in Chennai."

    def test_a_question_about_the_user_never_reaches_the_model(self, model, tmp_path):
        assert open_memory(tmp_path / "memory.txt").learn("Where do I live?") == []
        assert model["asked"] == []


class TestNoteOrderAndContent:
    def test_the_newest_notes_come_first_so_an_update_wins(self, model, tmp_path):
        (tmp_path / "memory.txt").write_text("The user lives in Las Vegas.\nThe user lives in Chennai.\n", encoding="utf-8")

        prompt = open_memory(tmp_path / "memory.txt").prompt()

        assert prompt.index("Chennai") < prompt.index("Las Vegas")

    def test_the_file_itself_stays_oldest_first(self, model, tmp_path):
        model["reply"] = "The user lives in Chennai."
        (tmp_path / "memory.txt").write_text("The user lives in Las Vegas.\n", encoding="utf-8")

        open_memory(tmp_path / "memory.txt").learn("I moved to Chennai.")

        assert (tmp_path / "memory.txt").read_text(encoding="utf-8").splitlines() == [
            "The user lives in Las Vegas.",
            "The user lives in Chennai.",
        ]

    def test_a_note_that_something_is_unknown_is_dropped(self):
        # As the newest note, it would override the real birthday.
        assert parse_notes("The user's sister's birthday is unknown.\nThe user has a sister.") == ["The user has a sister."]
