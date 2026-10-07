"""Web search through DuckDuckGo, for the few questions Baymax can't answer from memory.

Small models are hopeless at deciding when to search. Offered a search tool, llama3.2:3b
reaches for it even for "the capital of Australia" — a network round trip in front of every
reply, and the end of the one property Baymax is built on. So a plain text gate decides
instead, and questions that don't depend on live information never touch the network.

Two things send a question to the web: it depends on live information (weather, news, prices),
or it's technical enough that the model tends to answer it vaguely, or with an invented detail
in among the true ones — a Wheatstone bridge's balance condition, a welding hazard. Everyday
questions the model already answers well, general knowledge included, still cost nothing.
"""

import re
from collections.abc import Callable
from typing import NamedTuple
from urllib.parse import urlparse

MAX_RESULTS = 4  # few results keep the prompt short, which matters for a small local model
TIMEOUT_SECONDS = 5  # give up quickly: a slow search is worse than a "couldn't check" answer
_SNIPPET_CHARS = 300  # each result's body is cut to this many characters

# An explicit order always wins, and only what follows it gets searched.
_ASKED_TO_SEARCH = re.compile(
    r"^\s*(?:baymax[,\s]+)?(?:please\s+)?(?:could\s+you\s+|can\s+you\s+)?"
    r"(?:search(?:\s+the\s+web)?(?:\s+for)?|look\s+up|google)\s+(.+?)[?.!]*\s*$",
    re.IGNORECASE,
)

# Anything whose answer can change between now and next week. Erring towards searching is
# the cheap mistake: a needless search costs a few seconds, a missed one costs a wrong answer.
_NEEDS_LIVE_INFORMATION = re.compile(
    r"""\b(
        today | tonight | tomorrow | yesterday | right\s+now | currently | current |
        this\s+(?:week|weekend|month|year|morning|afternoon|evening) | latest | recent(?:ly)? |
        news | headlines? | weather | forecast | temps? | temperature | humidity |
        how\s+(?:hot|cold|warm|humid) | raining | rainy | snowing | will\s+it\s+(?:rain|snow) |
        what\s+time\s+is\s+it | open\s+(?:right\s+)?now | traffic |
        prices? | cost\s+of | stocks? | shares? | exchange\s+rate |
        scores? | who\s+won | winner | election | standings
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Technical questions the model tends to answer vaguely or with an invented detail: asked why a
# Wheatstone bridge balances, it never named the actual mechanism; asked about welding a fuel tank,
# it listed "toxic fumes like methanol" as a hazard, which isn't the real one. Two signals catch
# these without over-firing on everyday questions the model already answers well: an unambiguous
# request for a formula or derivation, or an explanatory question paired with domain vocabulary —
# "why does X work" alone would also catch "why does my dog bark", so it isn't enough by itself.
_QUANTITATIVE = re.compile(
    r"""\b(
        formula\s+for | derive | derivation | equation\s+for |
        second\s+moment\s+of\s+area | moment\s+of\s+inertia
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
_EXPLAIN = re.compile(
    r"""\b(
        why\s+(?:does|is|do|are) | what\s+causes | how\s+do(?:es)?\s+.+\s+work |
        what(?:'s|\s+is)\s+the\s+difference\s+between | is\s+it\s+safe\s+to |
        how\s+(?:do\s+you|to)\s+calculate
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
# Domain vocabulary that, combined with an _EXPLAIN phrase above, marks a question as technical.
_ENGINEERING_VOCAB = re.compile(
    r"""\b(
        bridge | circuit | voltage | current | resistance | resistor | capacitor | inductor |
        beam | cantilever | torque | stress | strain | yield | tensile | shear | bending |
        moment | load | pressure | velocity | acceleration | thermodynamic | efficiency |
        frequency | wavelength | bandwidth | pump | cavitation | weld(?:ing)? | alloy |
        corrosion | fatigue | fuel\s+tank | vapor | reaction | molecule | atom | valence |
        catalyst | gear | bearing | motor | engine | turbine | rpm | horsepower | structural |
        mechanical | electrical | chemical | aerodynamic | hydraulic | pneumatic |
        semiconductor | transistor | amplifier
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)


class WebCheck(NamedTuple):
    """The outcome of checking the web for one question."""

    results: str
    """Handed to the model alongside the question, then dropped: it's bulky and goes stale."""
    record: str
    """Kept in the history, so "where did you get that?" has a true answer to give."""


# Returned when a question doesn't need the web: nothing to add to the prompt or remember.
NOT_CHECKED = WebCheck("", "")


def search_query(text: str) -> str | None:
    """What to look up for ``text``, or None when Baymax should answer from what it knows."""
    if asked := _ASKED_TO_SEARCH.match(text):
        return asked.group(1).strip()
    if _NEEDS_LIVE_INFORMATION.search(text) or _is_technical(text):
        return text.strip()
    return None


def _is_technical(text: str) -> bool:
    """Whether the question asks for a formula/derivation, or explains something in engineering terms."""
    if _QUANTITATIVE.search(text):
        return True
    return bool(_EXPLAIN.search(text) and _ENGINEERING_VOCAB.search(text))


def lookup(text: str, announce: Callable[[], None] | None = None) -> WebCheck:
    """Check the web if the question needs live information.

    ``announce`` is called just before searching, so the few-second wait is visibly a search.
    """
    query = search_query(text)
    if not query:
        return NOT_CHECKED
    if announce:
        announce()
    results = _fetch(query)
    if not results:
        # Without this, a failed search looks exactly like no search, and the model guesses.
        return WebCheck(
            "You tried to check the web for this just now, but the search failed. Say you couldn't check.",
            "You tried to check the web to answer that, but the search failed.",
        )
    return WebCheck(
        f"Web results for {query!r}, fetched just now:\n" + "\n".join(map(_as_line, results)),
        "You checked the web to answer that, using " + ", ".join(dict.fromkeys(map(_site, results))) + ".",
    )


def search(query: str, max_results: int = MAX_RESULTS) -> str:
    """Return the top results as plain text, or '' if the search found nothing or failed."""
    return "\n".join(map(_as_line, _fetch(query, max_results)))


def _fetch(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Raw DuckDuckGo results for ``query``, or [] on any failure."""
    # Imported here because it pulls in lxml, and most runs never search.
    from ddgs import DDGS

    try:
        # Pinned: left on its default ("auto"), ddgs shuffles through every engine it knows --
        # Google included -- and scrapes each from this machine's IP. Google answers automated
        # queries like that with a captcha for anyone on the same connection, browsing included.
        found = DDGS(timeout=TIMEOUT_SECONDS).text(query, max_results=max_results, backend="duckduckgo")
    except Exception:
        # Offline first: a failed search should cost one answer, not the conversation.
        return []
    return [result for result in found if result.get("title")]  # untitled entries are unusable


def _as_line(result: dict) -> str:
    """One bullet line for the prompt: title, site, and a whitespace-collapsed, truncated snippet."""
    body = " ".join(str(result.get("body", "")).split())[:_SNIPPET_CHARS]
    return f"- {str(result['title']).strip()} ({_site(result)}): {body}"


def _site(result: dict) -> str:
    """The site a result came from, which is what a person means by "where did you get that?"."""
    return urlparse(str(result.get("href", ""))).netloc.removeprefix("www.") or "an unnamed site"
