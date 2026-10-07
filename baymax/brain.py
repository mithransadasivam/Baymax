"""Conversation with a local model served by Ollama."""

import datetime
from collections.abc import Callable, Iterator

import ollama

from baymax import emergency
from baymax.calc import lookup as calc_lookup
from baymax.memory import KEEP_ALIVE, Memory
from baymax.persona import RESEARCH_AWARE, SEARCH_SYSTEM_PROMPT, SYSTEM_PROMPT
from baymax.research import NOT_CHECKED as RESEARCH_NOT_CHECKED
from baymax.research import ResearchCheck
from baymax.search import NOT_CHECKED, WebCheck
from baymax.text import speak_moment

_FOLLOW_UP_WORDS = 6


class Brain:
    """Holds the conversation and builds each prompt: persona, date, optional web / calendar /
    research / calculator context, and the user's remembered notes, then streams the model's reply."""

    def __init__(
        self,
        model: str,
        host: str,
        max_turns: int = 8,
        lookup: Callable[[str], WebCheck] | None = None,
        research_lookup: Callable[[str], ResearchCheck] | None = None,
        memory: Memory | None = None,
    ) -> None:
        self._client = ollama.Client(host=host)
        self._model = model
        self._host = host
        self._history: list[dict[str, str]] = []
        self._max_messages = max_turns * 2  # a turn is one user message plus one assistant reply
        # Each lookup is optional; when one is None, that source is simply never consulted.
        self._lookup = lookup
        self._research_lookup = research_lookup
        self._system_prompt = SEARCH_SYSTEM_PROMPT if lookup else SYSTEM_PROMPT
        self._memory = memory
        # What each source did on the previous turn, replayed once so "where did that come from?"
        # gets an honest answer (see reply()).
        self._last_web_record = ""
        self._last_research_record = ""

    def check(self) -> None:
        """Raise RuntimeError with a fix-it hint if the server or model isn't available."""
        try:
            installed = {m.model for m in self._client.list().models}
        except ConnectionError:
            raise RuntimeError(f"can't reach Ollama at {self._host}. Start it with: ollama serve") from None
        if self._model not in installed and f"{self._model}:latest" not in installed:
            raise RuntimeError(f"model {self._model!r} isn't installed. Get it with: ollama pull {self._model}")

    def load(self) -> None:
        """Load the model into memory now, rather than stalling the first reply for several seconds."""
        self._client.generate(model=self._model, prompt="", keep_alive=KEEP_ALIVE)

    def reply(self, text: str) -> Iterator[str]:
        """Stream a reply token by token, keeping recent turns as conversational context."""
        self._history.append({"role": "user", "content": text})
        if alert := emergency.check(text):
            # Decided by plain text rules, never the model, and the model is never consulted for
            # this turn: a fixed, vetted script is the only thing worth saying here.
            self._history.append({"role": "assistant", "content": alert.script})
            del self._history[: -self._max_messages]
            yield alert.script
            return
        if self._memory:
            # Before replying, so a fact told just now is already known, and taking notes can never
            # overlap the moment the user might press Enter to cut Baymax off.
            self._memory.learn(text)
        messages = [{"role": "system", "content": self._prompt()}, *self._history]
        # Extra context is inserted at -1, i.e. just before the user's latest message, because
        # models weight the most recent text most heavily.
        if self._last_web_record:
            # Whether the previous reply came from the web, so "where did you get that?" is answered
            # honestly. Kept for one turn only: left in permanently, the model started citing sources
            # for questions it never actually searched, having seen the pattern established earlier
            # in the same conversation.
            messages.insert(-1, {"role": "system", "content": self._last_web_record})
        if self._last_research_record:
            messages.insert(-1, {"role": "system", "content": self._last_research_record})
        web = self._look_up(text)
        if web.results:
            messages.insert(-1, {"role": "system", "content": web.results})
        research = self._look_up_research(text)
        if research.results:
            messages.insert(-1, {"role": "system", "content": research.results})
        calc = calc_lookup(text)
        if calc.results:
            # No one-turn "record" tracking needed here, unlike web/calendar/research: a plain
            # arithmetic question is self-contained, and there's nothing stateful to be honest
            # about afterwards the way "which site did that come from" is.
            messages.insert(-1, {"role": "system", "content": calc.results})
        parts: list[str] = []
        try:
            for chunk in self._client.chat(model=self._model, messages=messages, stream=True, keep_alive=KEEP_ALIVE):
                if token := chunk.message.content:
                    parts.append(token)
                    yield token
        finally:
            # In a finally so history stays consistent even if the caller stops iterating early
            # (the user cutting Baymax off mid-reply).
            self._last_web_record = web.record
            self._last_research_record = research.record
            self._history.append({"role": "assistant", "content": "".join(parts)})
            del self._history[: -self._max_messages]  # trim to the most recent max_turns turns

    def _prompt(self) -> str:
        """Assemble the system prompt: persona, current time, then each enabled capability's notes."""
        prompt = self._system_prompt + self._now_line()
        if self._research_lookup:
            prompt += RESEARCH_AWARE
        if self._memory:
            prompt += self._memory.prompt()
        return prompt

    def _now_line(self) -> str:
        # A model this small has no reliable sense of "today" -- its notion of the date comes from
        # whenever its training data was collected, which is why it guessed a day of the week that
        # was already wrong. Telling it the real date and time, computed here rather than asked of
        # the model, is the only way it can ever get this right.
        return f"\nRight now it's {speak_moment(datetime.datetime.now(), year=True)}.\n"

    def _look_up(self, text: str) -> WebCheck:
        return self._lookup(text) if self._lookup else NOT_CHECKED

    def _look_up_research(self, text: str) -> ResearchCheck:
        if not self._research_lookup:
            return RESEARCH_NOT_CHECKED
        # A short follow-up ("what about for kids?") means nothing on its own, so it's searched
        # together with the question before it. Only the search sees this; the model gets the
        # real conversation.
        previous = [m["content"] for m in self._history[:-1] if m["role"] == "user"]
        query = f"{previous[-1]} {text}" if previous and len(text.split()) <= _FOLLOW_UP_WORDS else text
        return self._research_lookup(query)
