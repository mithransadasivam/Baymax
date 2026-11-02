"""Make audition clips of candidate Baymax voices, so you can listen and pick one.

Each voice says the same Baymax lines, once at Piper's own defaults and once with Baymax's softer,
slower settings. Voices not yet on this machine are downloaded from the Piper voices repository
(about 60 MB each) the first time they're used.

    uv run python tools/voice_samples.py [voice ...]

Clips land in voice-samples/ next to this project.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baymax.tts import Synthesizer  # noqa: E402

LINES = (
    "Hello. I am Baymax, your personal healthcare companion. "
    "On a scale of one to ten, how would you rate your pain? "
    "I am sorry you are hurting. Let me help. "
    "I am satisfied with my care."
)

# American male voices that are warm and even; the first is already on this machine.
CANDIDATES = [
    "en_US-ryan-medium",
    "en_US-joe-medium",
    "en_US-john-medium",
    "en_US-norman-medium",
    "en_US-danny-low",
    "en_US-hfc_male-medium",
]


def main() -> None:
    voices = sys.argv[1:] or CANDIDATES
    out = ROOT / "voice-samples"
    out.mkdir(exist_ok=True)
    for voice in voices:
        try:
            plain = Synthesizer(voice, speed=1.0, variation=0.667, rhythm=0.8)
            baymax = Synthesizer(voice)  # the Baymax defaults from baymax/tts.py
        except Exception as exc:  # noqa: BLE001 - a missing voice shouldn't stop the others
            print(f"{voice}: skipped ({type(exc).__name__}: {exc})")
            continue
        plain.save_wav(LINES, out / f"{voice}__default.wav")
        baymax.save_wav(LINES, out / f"{voice}__baymax.wav")
        print(f"{voice}: done")
    print(f"\nClips are in {out}")


if __name__ == "__main__":
    main()
