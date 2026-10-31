"""Heart-rate readings from the user's Apple Watch, kept on this machine and explained on request.

A Shortcut on the iPhone/Watch reads Health data and posts it to the phone server (``/vitals`` in
baymax/phone.py) over the home network or Tailscale. Readings go into a plain-text file under
~/.baymax/ and never anywhere else.

Interpretation is deliberately rule-based, like emergency.py: which band a number falls in, and
whether a number plus a symptom is urgent, is decided here, not guessed by the language model. The
model is only handed the facts and told to explain them calmly. These are wellness readings from a
consumer watch, not a medical measurement, and nothing here diagnoses a rhythm problem.

Reference ranges are general adult resting figures (American Heart Association). Like the notes in
knowledge/clinical.md, they have not been clinically reviewed.
"""

from __future__ import annotations

import hmac
import json
import re
import secrets
import threading
import time
from pathlib import Path
from typing import NamedTuple

DEFAULT_DIR = Path.home() / ".baymax"
MIN_BPM, MAX_BPM = 25, 250  # outside this is a sensor glitch or a typo, not a heartbeat
FRESH_SECONDS = 30 * 60  # a reading older than this says nothing about how someone feels right now
MAX_KEPT = 5000


class Reading(NamedTuple):
    ts: float
    bpm: int
    kind: str  # "heart_rate" (any moment) or "resting_heart_rate" (the watch's daily resting value)
    source: str


class VitalsStore:
    """Readings in ``vitals.jsonl``: one JSON object per line, so the file is easy to read, edit, or delete."""

    def __init__(self, directory: Path = DEFAULT_DIR) -> None:
        self._dir = directory
        self._path = directory / "vitals.jsonl"
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def add(self, bpm: float, kind: str = "heart_rate", ts: float | None = None, source: str = "apple-watch") -> Reading:
        """Record one reading. Raises ValueError for a value no heart could have."""
        if kind not in ("heart_rate", "resting_heart_rate"):
            raise ValueError(f"unknown reading type {kind!r}")
        value = round(float(bpm))
        if not MIN_BPM <= value <= MAX_BPM:
            raise ValueError(f"{value} bpm is outside the believable range {MIN_BPM}-{MAX_BPM}")
        now = time.time()
        stamp = now if ts is None else float(ts)
        if stamp > now + 300:  # a clock a few minutes fast is fine; the future isn't
            stamp = now
        reading = Reading(stamp, value, kind, source)
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            lines = self._lines()
            lines.append(json.dumps(reading._asdict()))
            self._path.write_text("\n".join(lines[-MAX_KEPT:]) + "\n", encoding="utf-8")
        return reading

    def _lines(self) -> list[str]:
        try:
            return [line for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return []

    def readings(self) -> list[Reading]:
        out = []
        for line in self._lines():
            try:
                out.append(Reading(**json.loads(line)))
            except (ValueError, TypeError):
                continue  # a hand-edited or damaged line is skipped, never fatal
        return sorted(out, key=lambda r: r.ts)

    def latest(self, kind: str | None = None) -> Reading | None:
        found = [r for r in self.readings() if kind is None or r.kind == kind]
        return found[-1] if found else None

    def since(self, seconds: float) -> list[Reading]:
        cutoff = time.time() - seconds
        return [r for r in self.readings() if r.ts >= cutoff]

    # -- what Brain uses --------------------------------------------------------------------

    def context_for(self, text: str) -> str:
        """Facts to hand the model alongside ``text``, or "" when it isn't about the user's heart rate."""
        if not asks_about_heart(text):
            return ""
        return describe(self.readings())

    def urgent_alert(self, text: str) -> str | None:
        """A fixed 911 script when a fresh, extreme reading comes with a worrying symptom. Never the model."""
        latest = self.latest()
        if latest is None or time.time() - latest.ts > FRESH_SECONDS or not _WORRYING_SYMPTOM.search(text):
            return None
        # "no chest pain and I'm not dizzy" must not count: strip every negated symptom, then look again.
        if not _WORRYING_SYMPTOM.search(_NEGATED_SYMPTOM.sub(" ", text)):
            return None
        if latest.bpm >= 150 or latest.bpm <= 40:
            word = "very fast" if latest.bpm >= 150 else "very slow"
            return (
                f"Please call 911 now. Your watch just read {latest.bpm} beats per minute, which is {word}, and "
                "you're describing symptoms that go with it. Sit or lie down, stay still, and don't drive "
                "yourself. If you have someone nearby, ask them to stay with you. I'm right here."
            )
        return None


# Questions about the user's own heart rate / pulse, as opposed to heart disease in general.
_HEART_TOPIC = re.compile(
    r"\b(heart\s*rate|pulse|bpm|beats?\s+per\s+minute|heart\s*beat|heartbeat|palpitations?|"
    r"heart\s+(?:is\s+|has\s+been\s+|feels?\s+)?(?:racing|pounding|fluttering|skipping|beating)|"
    r"racing\s+heart|resting\s+heart)\b",
    re.IGNORECASE,
)
_ABOUT_ME = re.compile(
    r"\b(?:my|me|i|i'?m|mine|check|scan|read|measure|what'?s|how'?s|is\s+it)\b", re.IGNORECASE
)
_WORRYING_SYMPTOM = re.compile(
    r"\b(?:chest\s+(?:pain|pressure|tightness|hurts?)|faint\w*|pass(?:ed|ing)\s+out|dizz\w+|lighthead\w*|"
    r"short\s+of\s+breath|can'?t\s+breathe|trouble\s+breathing|shortness\s+of\s+breath|confus\w+)\b",
    re.IGNORECASE,
)
_NEGATED_SYMPTOM = re.compile(
    r"\b(?:no|not|without|never|don'?t\s+have|haven'?t\s+had)\s+(?:\w+\s+){0,2}"
    r"(?:chest\s+pain|dizz\w+|faint\w*|breath\w*)\b",
    re.IGNORECASE,
)


def asks_about_heart(text: str) -> bool:
    """Whether ``text`` is about the user's own pulse. "What causes a fast heartbeat?" counts too: the
    model answers it better knowing the user's actual reading."""
    return bool(_HEART_TOPIC.search(text) and _ABOUT_ME.search(text))


def band(bpm: int) -> tuple[str, str]:
    """(label, plain explanation) for an adult resting heart rate."""
    if bpm < 40:
        return "very low", (
            "Under 40 is very low. It can be normal in elite athletes, but with dizziness, fainting, or "
            "chest discomfort it needs prompt medical attention."
        )
    if bpm < 60:
        return "below the usual range", (
            "Under 60 is below the typical 60 to 100, but it's common and healthy in fit people and during "
            "sleep. It matters mainly if it comes with dizziness, fainting, or unusual tiredness."
        )
    if bpm <= 100:
        return "normal", "60 to 100 beats per minute is the usual range for an adult at rest."
    if bpm <= 120:
        return "a little high", (
            "101 to 120 is mildly above the usual resting range. Everyday causes are the most common: "
            "caffeine, stress or anxiety, poor sleep, dehydration, a recent climb of stairs, a fever, or "
            "some medicines. It usually settles after resting."
        )
    return "high", (
        "Over 120 at true rest is high. Recent activity, fever, dehydration, or caffeine can do this, but "
        "if it stays up after resting for several minutes, or comes with chest pain, trouble breathing, "
        "fainting, or dizziness, it's worth getting checked promptly."
    )


def _ago(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 48:
        return f"about {hours} hour{'s' if hours != 1 else ''} ago"
    return f"{hours // 24} days ago"


def describe(readings: list[Reading]) -> str:
    """The facts about the user's readings, as a note for the model. Never claims a number it doesn't have."""
    if not readings:
        return (
            "The user asked about their heart rate, but no reading has reached Baymax from their Apple Watch "
            "yet. Say so plainly and kindly, and tell them to run their Baymax Shortcut on the watch or "
            "iPhone. Do not guess a number."
        )
    now = time.time()
    latest = readings[-1]
    label, why = band(latest.bpm)
    if latest.kind == "resting_heart_rate":
        kind_note = "the watch's resting value"
    else:
        kind_note = "taken at any moment, so it may include activity"
    parts = [
        "Heart-rate readings from the user's Apple Watch, received on this machine (facts, not guesses):",
        f"- Latest: {latest.bpm} beats per minute, {_ago(now - latest.ts)} ({kind_note}).",
        f"- For an adult at rest that is {label}. {why}",
    ]
    day = [r for r in readings if now - r.ts <= 86400 and r.kind == "heart_rate"]
    if len(day) >= 3:
        low, high = min(r.bpm for r in day), max(r.bpm for r in day)
        parts.append(f"- Last 24 hours: {len(day)} readings, from {low} to {high}.")
    rest = [r for r in readings if r.kind == "resting_heart_rate"][-7:]
    if len(rest) >= 3:
        parts.append(
            "- Recent resting values: " + ", ".join(str(r.bpm) for r in rest) + " (oldest to newest)."
        )
    if now - latest.ts > FRESH_SECONDS:
        parts.append(
            "- This reading is not recent, so say it may not match how they feel right now, and offer to "
            "look at a fresh one from the watch."
        )
    parts.append(
        "Explain it calmly in two or three sentences, leading with the ordinary explanation. A consumer "
        "watch is not a medical device, so don't diagnose a rhythm problem. Mention when to see a doctor "
        "in one short sentence, and ask whether they were resting if that matters."
    )
    return "\n".join(parts)


# -- the token the Shortcut must send --------------------------------------------------------


def token(directory: Path = DEFAULT_DIR) -> str:
    """The shared secret the Shortcut sends with each reading, created on first use. It keeps a stranger
    on the same WiFi from filling the user's health history with invented numbers."""
    path = directory / "vitals-token"
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    directory.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(18)
    path.write_text(value + "\n", encoding="utf-8")
    return value


def token_matches(expected: str, given: str | None) -> bool:
    return bool(given) and hmac.compare_digest(expected.encode(), str(given).encode())
