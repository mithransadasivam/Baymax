"""Notes about the user that last between sessions, kept in a plain text file they can read and edit.

Statements about the user go to the model with a note-taking prompt, which carries worked examples:
without them, llama3.2:3b turned "remember my sister's birthday is on March 3rd" into "the user has a
sister". Questions never go to it, because it answers them with invented facts. Every startup puts the
notes back into the prompt, and because the file is plain text, a wrong note is one deleted line away.
"""

import re
from collections.abc import Callable
from pathlib import Path

import ollama

DEFAULT_FILE = Path.home() / ".baymax" / "memory.txt"

# How long Ollama keeps the model loaded after a request. Its own default is 5 minutes, and every
# request resets the clock to whatever *that* request asked for -- so a single one that leaves it
# out quietly undoes any longer setting, and the next question after a quiet spell waits for the
# whole model to load from disk again. It lives here only because this is the lowest-level module
# that talks to the model; every request Baymax makes to it has to carry this.
KEEP_ALIVE = "3h"
MAX_NOTES_IN_PROMPT = 50

# Facts about someone come in the first person, so anything else skips the model entirely.
_ABOUT_THEMSELVES = re.compile(r"\b(?:i|me|my|mine|myself|we|us|our|ours|remember)\b", re.IGNORECASE)

# Questions state no facts, but the model answers them anyway: shown "Where do I live?", it noted that
# the user lives in New York City. So it only ever sees statements, unless a question explicitly asks
# for something to be remembered.
_QUESTION = re.compile(
    r"^\s*(?:what|when|where|who|whose|which|why|how|is|are|am|was|were|do|does|did|can|could|will|would|"
    r"should|have|has|tell|remind|give|show|find|explain|describe)\b",
    re.IGNORECASE,
)
_ASKED_TO_REMEMBER = re.compile(r"\bremember\s+that\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# A note that something is unknown holds no fact, and as the newest note it would override the real one.
_NOT_A_FACT = re.compile(
    r"\b(?:unknown|not\s+known|not\s+sure|unsure|no\s+information|does\s*n[o']t\s+know)\b", re.IGNORECASE
)

_NOTE_TAKING_PROMPT = """\
You keep a notes file about the person using a voice assistant. Read their message and write \
down every lasting fact it reveals about them, keeping all the details such as names, dates, and places. \
Lasting facts include their name, where they live, family, pets, work, likes, dislikes, preferences, \
health details (allergies, medicines and doses, conditions, surgeries, pregnancy, blood type), plans, and anything they ask you to remember. Write each fact on its own line as a short sentence \
starting with "The user". Only write down what the message actually says. Ignore requests, small talk, \
passing moods, and temporary symptoms like a current cold or fever (keep only lasting conditions, allergies, and medicines). If there is nothing lasting worth keeping, reply with exactly NONE.

Examples:
Message: Remember that my dad's birthday is on June 9th.
The user's dad's birthday is on June 9th.

Message: I'd rather you use kilometres, not miles.
The user prefers kilometres to miles.

Message: I moved to Berlin in May.
The user lives in Berlin.

Message: I'm allergic to penicillin and I take metformin for type 2 diabetes.
The user is allergic to penicillin.
The user takes metformin for type 2 diabetes.

Message: I've had a sore throat and a fever of 101 for two days.
NONE

Message: I'm so sleepy right now.
NONE

Message: I told you my name earlier.
NONE"""


class Memory:
    """The user's remembered notes: loaded from a text file, extended by asking the model to pull
    lasting facts out of what the user says, and rendered into the system prompt."""

    def __init__(
        self,
        path: Path,
        host: str,
        model: str,
        on_noted: Callable[[list[str]], None] | None = None,
    ) -> None:
        self._path = path
        self._client = ollama.Client(host=host)
        self._model = model
        self._on_noted = on_noted  # optional callback, e.g. so the UI can show what was just noted
        try:
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        except OSError as exc:
            raise RuntimeError(f"can't read the memory file {path}: {exc}") from None
        self.notes = [line.strip() for line in lines if line.strip()]

    def prompt(self) -> str:
        """The notes, ready to add to the system prompt, or '' when there aren't any."""
        if not self.notes:
            return ""
        # Newest first: listed oldest first, llama3.2:3b kept answering "Las Vegas" after the user
        # said they'd moved to Chennai, even when told that later notes win.
        recent = reversed(self.notes[-MAX_NOTES_IN_PROMPT:])
        return (
            "\n\nNotes about the user from earlier conversations, newest first. When two notes disagree, the one "
            "nearer the top is true and the other is out of date. Use them when they're relevant, and don't "
            "recite them unprompted:\n" + "\n".join(f"- {note}" for note in recent)
        )

    def learn(self, text: str) -> list[str]:
        """Note anything lasting that ``text`` reveals about the user, and return the notes that are new."""
        statements = worth_noting(text)
        if not statements:
            return []
        try:
            reply = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _NOTE_TAKING_PROMPT},
                    {"role": "user", "content": f"Message: {statements}"},
                ],
                options={"temperature": 0},
                keep_alive=KEEP_ALIVE,
            ).message.content
        except Exception:
            # A missed note costs little; a conversation that dies mid-sentence costs a lot.
            return []
        # Drop notes already known, and duplicates within this reply, comparing on normalized text.
        known = {_normalized(note) for note in self.notes}
        new = list({_normalized(note): note for note in parse_notes(reply) if _normalized(note) not in known}.values())
        if new:
            self._save(new)
            self.notes.extend(new)
            if self._on_noted:
                self._on_noted(new)
        return new

    def _save(self, new: list[str]) -> None:
        """Append ``new`` notes to the file, one per line, creating it (and its folder) if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        # Someone editing the file by hand may not leave a newline at the end.
        separator = "\n" if existing and not existing.endswith("\n") else ""
        with self._path.open("a", encoding="utf-8") as file:
            file.write(separator + "".join(f"{note}\n" for note in new))


def worth_noting(text: str) -> str:
    """The sentences in ``text`` that might say something lasting about the user, or '' if none do."""
    kept = []
    for sentence in _SENTENCE_END.split(text.strip()):
        # Keep first-person statements; skip questions unless they explicitly say "remember that".
        asking =sentence.rstrip().endswith("?") or _QUESTION.match(sentence)
        if _ASKED_TO_REMEMBER.search(sentence) or (not asking and _ABOUT_THEMSELVES.search(sentence)):
            kept.append(sentence.strip())
    return " ".join(kept)


def parse_notes(reply: str) -> list[str]:
    """Pull the notes out of the model's reply, ignoring NONE, chatter, and notes that hold no fact."""
    notes = []
    for line in reply.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if line.lower().startswith("the user") and not _NOT_A_FACT.search(line):
            notes.append(line if line.endswith((".", "!", "?")) else f"{line}.")
    return notes


def _normalized(note: str) -> str:
    """A note reduced to lowercase words with no end punctuation, for duplicate detection."""
    return " ".join(note.lower().rstrip(".!?").split())
