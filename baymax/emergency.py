"""Spotting a medical emergency, deliberately without the language model.

A small local model can't be trusted to notice that "my left arm is numb and my chest feels tight"
means call an ambulance now, and a missed emergency is the one failure this assistant can't have.
So, like the web gate in baymax/search.py and the calculator in baymax/calc.py, a plain text check
decides, and a fixed script is spoken instead of whatever the model might have improvised. It errs
on the side of escalating: a needless "call 911" costs a minute, a missed one can cost far more.

Numbers are for the United States (911, 988, Poison Control).
"""

from __future__ import annotations

import re
from typing import NamedTuple

EMERGENCY = "emergency"
CRISIS = "crisis"


class Alert(NamedTuple):
    """A detected emergency: ``kind`` is a short label, ``script`` is what Baymax says."""

    kind: str
    script: str


_CALL = "Please call 911 now"
_FLAGS = re.IGNORECASE | re.VERBOSE

# (kind, pattern, script). The first match wins, so the most time-critical rules come first.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "suicide-or-self-harm",
        re.compile(
            r"""\b(
                kill(?:ing)?\s+(?:my\s*self|me) | suicid\w* | end(?:ing)?\s+(?:my\s+(?:own\s+)?life|it\s+all) |
                want\s+to\s+die | wanna\s+die | don'?t\s+want\s+to\s+(?:live|be\s+alive|be\s+here) |
                (?:hurt|harm|cut)(?:ting)?\s+my\s*self | better\s+off\s+(?:dead|without\s+me) |
                take\s+my\s+own\s+life | no\s+reason\s+to\s+(?:live|go\s+on)
            )\b""",
            _FLAGS,
        ),
        "I'm really glad you told me, and I'm concerned about you. You don't have to carry this alone. "
        "Please call or text 9 8 8, the Suicide and Crisis Lifeline, right now. It's free and someone "
        "answers any time. If you are in immediate danger or have already hurt yourself, call 911. "
        "Please stay with another person if you can. I'm here, and I'll stay with you while you reach out.",
    ),
    (
        "heart-attack",
        re.compile(
            r"""\b(
                chest\s+(?:pain|pains|pressure|tightness|discomfort|heaviness) |
                (?:pain|pressure|tightness|crushing)\s+(?:in|on|across)\s+(?:my\s+|the\s+)?chest |
                (?:my\s+)?chest\s+(?:hurts|is\s+(?:tight|crushing|squeezing)|feels\s+(?:tight|heavy|crushed)) |
                heart\s+attack |
                pain\s+(?:going|spreading|radiating)\s+(?:down|into|to)\s+(?:my\s+)?(?:left\s+)?(?:arm|jaw)
            )\b""",
            _FLAGS,
        ),
        f"{_CALL}. Chest pain can be a heart attack, and minutes matter. Stop what you're doing and sit "
        "down. Unlock your front door and stay on the line with the dispatcher. If you are not allergic "
        "to aspirin and have no doctor's warning against it, the dispatcher may tell you to chew one "
        "regular aspirin. Don't drive yourself. I'm staying right here.",
    ),
    (
        "stroke",
        re.compile(
            r"""\b(
                stroke | (?:face|smile|mouth)\s+(?:is\s+)?(?:droop\w*|numb|crooked|uneven) |
                droop\w*\s+(?:face|smile|mouth) | slurr\w*\s+(?:speech|words) |
                (?:speech|words)\s+(?:are|is)\s+slurr\w* |
                can'?t\s+(?:speak|talk|find\s+(?:my|the)\s+words) |
                sudden(?:ly)?\s+(?:weak\w*|numb\w*|confus\w*|can'?t\s+see|lost\s+(?:my\s+)?vision|dizz\w*) |
                (?:one|left|right)\s+(?:side|arm|leg)\s+(?:is\s+)?(?:weak|numb|limp|paraly\w+) |
                worst\s+headache\s+(?:of\s+my\s+life|ever)
            )\b""",
            _FLAGS,
        ),
        f"{_CALL}. These can be signs of a stroke, and treatment works best in the first hour. Note the "
        "exact time the symptoms started and tell the dispatcher. Don't give food, drink, or pills. Lie "
        "down with your head slightly raised, and don't drive yourself.",
    ),
    (
        "cannot-breathe",
        re.compile(
            r"""\b(
                can'?t\s+(?:breathe|breath|catch\s+my\s+breath|get\s+(?:any\s+)?(?:air|breath)) |
                (?:struggling|gasping|fighting)\s+(?:to|for)\s+(?:breathe|breath|air) |
                (?:not|stopped|stops|isn'?t)\s+breathing |
                (?:lips|face|fingers)\s+(?:are\s+|is\s+)?(?:turning\s+)?(?:blue|purple|gray|grey) |
                chok(?:ing|ed) | something\s+(?:is\s+)?stuck\s+in\s+(?:my|his|her|their|the)\s+throat
            )\b""",
            _FLAGS,
        ),
        f"{_CALL}. Trouble breathing is an emergency. Sit upright and stay as calm as you can. If you "
        "have a rescue inhaler or an EpiPen prescribed to you, use it now. If someone is choking and "
        "can't cough, speak, or breathe, give five firm back blows between the shoulder blades, then five "
        "abdominal thrusts, and repeat until it clears or help arrives. If anyone has stopped breathing, "
        "the dispatcher will coach you through CPR.",
    ),
    (
        "anaphylaxis",
        re.compile(
            r"""\b(
                anaphyla\w+ |
                (?:throat|tongue|lips?|face)\s+(?:is\s+|are\s+|feels?\s+)?(?:swelling|swollen|closing|tight) |
                swelling\s+(?:of|in)\s+(?:my\s+|the\s+)?(?:throat|tongue|lips|face) |
                (?:hives|rash)\s+and\s+(?:trouble|difficulty)\s+breathing
            )\b""",
            _FLAGS,
        ),
        f"{_CALL}. This could be a severe allergic reaction. If you have an epinephrine auto-injector, "
        "use it right now in the outer thigh, then still call 911 even if you feel better, because the "
        "reaction can come back. Lie down with your legs raised unless that makes it harder to breathe.",
    ),
    (
        "severe-bleeding",
        re.compile(
            r"""\b(
                (?:won'?t|can'?t|doesn'?t|isn'?t)\s+(?:stop|stopping)\s+bleeding |
                bleeding\s+(?:won'?t|will\s+not|doesn'?t|isn'?t)\s+stop\w* |
                bleeding\s+(?:heavily|badly|a\s+lot|profusely|out) | (?:heavy|severe|massive)\s+bleeding |
                (?:coughing|vomiting|throwing)\s+up\s+(?:up\s+)?blood |
                blood\s+(?:is\s+)?(?:pouring|gushing|spurting) | soaking\s+through
            )\b""",
            _FLAGS,
        ),
        f"{_CALL}. Press firmly on the wound with a clean cloth or your hand and don't lift it to check. "
        "If blood soaks through, add more cloth on top and keep pressing. If it's an arm or leg and "
        "the bleeding is life-threatening, a tight tourniquet above the wound can save a life. Lie down "
        "and keep warm until help arrives.",
    ),
    (
        "overdose-or-poisoning",
        re.compile(
            r"""\b(
                overdos\w+ | poison\w* |
                (?:took|taken|swallowed|ate|drank|had)\s+(?:too\s+many|too\s+much|a\s+whole\s+bottle|an?\s+entire) |
                (?:swallowed|drank|ate)\s+(?:bleach|antifreeze|cleaning|detergent|pesticide|a\s+battery)
            )\b""",
            _FLAGS,
        ),
        "Please act now. If the person is unconscious, having a seizure, or struggling to breathe, call "
        "911. Otherwise call Poison Control at 1, 8 0 0, 2 2 2, 1 2 2 2. It's free, private, and open all "
        "day and night. Have the container with you and know roughly how much and when. Don't make them "
        "vomit unless they tell you to.",
    ),
    (
        "unresponsive-or-seizure",
        re.compile(
            r"""\b(
                unconscious | unresponsive | (?:won'?t|can'?t|not)\s+wak\w+ | passed\s+out\s+and |
                (?:having|had|is\s+having|started\s+having|in)\s+(?:a\s+)?seizure | seizing | convuls\w+ |
                no\s+pulse
            )\b""",
            _FLAGS,
        ),
        f"{_CALL}. If someone is having a seizure, clear the space around them, cushion their head, turn "
        "them onto their side, and don't put anything in their mouth. Time it. If someone is "
        "unresponsive and not breathing normally, start chest compressions in the center of the chest, "
        "hard and fast, and the dispatcher will guide you.",
    ),
]

# "I don't have chest pain" must not trigger. Checked in the few words before a match.
_NEGATED = re.compile(
    r"\b(?:no|not|never|without|don'?t|doesn'?t|didn'?t|haven'?t|hasn'?t|isn'?t|wasn'?t|nothing|neither|nor)\s+"
    r"(?:\w+\s+){0,3}$",
    re.IGNORECASE,
)
# "my dad had a stroke ten years ago" is history, not an emergency, unless something says it's happening now.
_PAST = re.compile(
    r"\b(?:years?|decades?|months?)\s+ago|last\s+year|history\s+of|used\s+to|when\s+i\s+was\s+(?:a\s+)?(?:kid|child|young)",
    re.IGNORECASE,
)
_NOW = re.compile(
    r"\b(?:now|currently|today|tonight|just\s+(?:now|started|happened)|this\s+(?:minute|morning)|started|help)\b",
    re.IGNORECASE,
)
# Asking about a condition in general terms is not having one.
_GENERAL = re.compile(
    r"^\s*(?:what\s+(?:is|are|causes|happens)|how\s+(?:do|does|can|to)|why\s+(?:do|does|is)|explain|tell\s+me\s+about|"
    r"what'?s\s+the\s+difference|signs\s+of|symptoms\s+of|how\s+(?:common|dangerous))\b",
    re.IGNORECASE,
)
_PERSONAL = re.compile(
    r"\b(?:i|i'?m|i'?ve|my|me|he|she|they|his|her|their|someone|somebody|we|our|mom|dad|mother|father|"
    r"baby|child|kid|friend|wife|husband)\b",
    re.IGNORECASE,
)


def check(text: str) -> Alert | None:
    """The emergency script for ``text``, or None when nothing in it looks urgent."""
    general = _GENERAL.match(text)
    for kind, pattern, script in _RULES:
        for match in pattern.finditer(text):
            if _NEGATED.search(text[: match.start()][-40:]):
                continue
            # Thoughts of suicide are never filtered by tense or phrasing: a missed one is the worst error.
            if not kind.startswith("suicide"):
                if _PAST.search(text) and not _NOW.search(text):
                    continue
                # "What causes a stroke?" is a question about the topic, not a person in trouble.
                if general and not _PERSONAL.search(text[general.end() :]):
                    continue
            return Alert(CRISIS if kind.startswith("suicide") else EMERGENCY, script)
    return None
