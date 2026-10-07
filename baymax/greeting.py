"""Deciding whether a transcribed utterance was meant for Baymax, and separating the greeting
from the actual question.

No trained wake word means anyone starting to speak gets recorded and transcribed by
baymax.app._voice_watcher, well before there's any way to know if they were talking to Baymax at
all. This is what tells "hey Baymax, what's the capital of France" apart from someone across the
room saying "hey, did you catch the game last night" -- a text check on what Whisper heard, not an
acoustic model. The trade-off going in: common words like "hey" and "hi" will occasionally catch
nearby conversation that a trained phrase like "hey jarvis" wouldn't have.
"""

import re

GREETINGS = [
    "hey baymax", "hi baymax", "hello baymax",
    "good morning", "good afternoon", "good evening",
    "baymax", "hey", "hi", "hello", "yo",
]

# Longest first, so "hey baymax" is tried before the "hey" it also starts with -- otherwise "hey"
# would match alone and leave "baymax, what's..." sitting unstripped in front of the question.
_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(re.escape(g) for g in sorted(GREETINGS, key=len, reverse=True)) + r")\b[,.!\s]*",
    re.IGNORECASE,
)


def strip_greeting(text: str) -> str | None:
    """What's left after the greeting Baymax was addressed by, or None if ``text`` doesn't open
    with one at all -- in which case it almost certainly wasn't addressed to Baymax.

    An empty string means the greeting was the whole utterance ("Hey Baymax." and nothing else):
    addressed to Baymax, but no question in it yet.
    """
    match = _PATTERN.match(text)
    if not match:
        return None
    return text[match.end() :].strip()
