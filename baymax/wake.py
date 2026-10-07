"""Hands-free activation: wait for speech, then record until the user stops talking.

Two ways to decide someone's actually talking to Baymax. A trained model from openWakeWord listens
for one specific phrase before recording starts at all -- it ships a ready-made "hey jarvis", and a
custom phrase such as "hey baymax" is a single .onnx file, trained with its own notebook and passed
in place of the name. Or, with no model at all (``model=None``), anyone starting to speak is recorded
and handed to baymax.greeting to check afterwards -- see that module for why. Either way, with no
second key press to mark the end of a command, the Silero voice activity detector that ships
alongside openWakeWord decides when the user has stopped talking.
"""

import re
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # Hz; what openWakeWord and the Silero VAD both work at
FRAME = 1280  # 80 ms: the chunk size openWakeWord expects
# Used to turn the second-based settings of Endpointer into frame counts.
_FRAMES_PER_SECOND = SAMPLE_RATE / FRAME


def phrase(model: str) -> str:
    """What to say for a model: "hey_jarvis" is "hey jarvis", and "models/hey_baymax_v0.1.onnx" is "hey baymax"."""
    name = Path(model).stem if model.endswith((".onnx", ".tflite")) else model
    return re.sub(r"_v\d[\w.]*$", "", name).replace("_", " ")


class Endpointer:
    """Decides when a spoken command is over, from one speech probability per frame."""

    def __init__(
        self,
        speech_threshold: float = 0.5,  # probability at or above which a frame counts as speech
        silence_to_finish: float = 1.0,  # seconds of quiet after speaking that end the command
        wait_for_speech: float = 5.0,  # seconds to wait for speech to begin before giving up
        longest: float = 15.0,  # hard cap, in seconds, on any one command
    ) -> None:
        self._threshold = speech_threshold
        self._silence_frames = round(silence_to_finish * _FRAMES_PER_SECOND)
        self._waiting_frames = round(wait_for_speech * _FRAMES_PER_SECOND)
        self._longest_frames = round(longest * _FRAMES_PER_SECOND)
        self._frames = 0
        self._quiet_frames = 0  # consecutive non-speech frames; reset whenever speech returns
        self.heard_speech = False  # public: callers use it to tell a command from a timeout

    def finished(self, speech_probability: float) -> bool:
        """Count one more frame, and say whether the user has stopped talking, or never started."""
        self._frames += 1
        if speech_probability >= self._threshold:
            self.heard_speech = True
            self._quiet_frames = 0
        else:
            self._quiet_frames += 1
        if self._frames >= self._longest_frames:
            return True
        # Before any speech, only the waiting limit applies; after it, a long enough pause ends it.
        if not self.heard_speech:
            return self._frames >= self._waiting_frames
        return self._quiet_frames >= self._silence_frames


class WakeWord:
    """Listens on the microphone for a wake phrase (or any speech), then captures the command."""

    def __init__(self, model: str | None, threshold: float = 0.5) -> None:
        """``model=None`` skips loading a keyword detector entirely, for VAD-only listening --
        only ``wake_first=False`` calls are valid then; ``hear()`` raises otherwise."""
        # Imported here, because push-to-talk never needs them.
        import openwakeword.utils
        from openwakeword.vad import VAD

        self._detector = None
        self.phrase = ""
        if model is not None:
            from openwakeword.model import Model

            try:
                # Fetches only files that are missing, so after the first run this never touches the network.
                openwakeword.utils.download_models([model])
                self._detector = Model(wakeword_models=[model], inference_framework="onnx")
            except Exception as exc:
                raise RuntimeError(
                    f"can't load the wake word model {model!r}: {exc}. "
                    "Use a built-in name such as hey_jarvis, or the path to a .onnx file"
                ) from None
            self.phrase = phrase(model)
        else:
            # download_models([]) means "download every official model", not "download none" --
            # a name that matches none of them is the only way to fetch just the feature/VAD
            # models it always fetches regardless, without every pretrained keyword model too.
            openwakeword.utils.download_models(["__no_keyword_model__"])
        self._vad = VAD()  # Silero voice activity detector, used to find the end of the command
        self._threshold = threshold  # wake-word score at or above which the phrase counts as heard

    def listen(
        self,
        device: int | str | None = None,
        on_wake: Callable[[], None] | None = None,
        wake_first: bool = True,
        on_level: Callable[[float], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
        wait_for_speech: float = 5.0,
    ) -> np.ndarray:
        """Wait for the wake word, unless ``wake_first`` is False, then return the command that follows it."""
        import sounddevice as sd

        # int16 matches what openWakeWord's detector takes; hear() converts to float afterwards.
        with sd.InputStream(device=device, samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME) as stream:
            command = self.hear(_frames(stream), on_wake, wake_first, on_level, stop_check, wait_for_speech)
        return command if command is not None else np.zeros(0, dtype=np.float32)

    def hear(
        self,
        frames: Iterator[np.ndarray],
        on_wake: Callable[[], None] | None = None,
        wake_first: bool = True,
        on_level: Callable[[float], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
        wait_for_speech: float = 5.0,
    ) -> np.ndarray | None:
        """The command spoken after the wake word, as 16 kHz float audio.

        Empty if the user woke Baymax and then said nothing within ``wait_for_speech`` seconds --
        worth raising a little for a wake-word-free follow-up, since deciding whether you have one
        takes a beat longer than a command you were already about to say. None if the frames ran
        out before it woke, or ``stop_check`` returned True first. ``on_level`` is called with a
        0..1 loudness estimate for every frame after the wake word, for driving a live meter — not
        before it, since a HUD showing "idle" has nothing to meter yet. ``stop_check`` is polled
        once per frame, but only while still waiting for the wake word: once the user is actually
        talking, cutting them off mid-command would be worse than letting this call finish.
        """
        # Imported here, not at module scope, so importing wake.py never pulls in Piper's chain
        # (baymax.audio imports Synthesizer) just to reach a small pure function.
        from baymax.audio import audio_level

        if wake_first and self._detector is None:
            raise ValueError("this WakeWord has no keyword model (model=None); only wake_first=False is valid")
        # Clear state left over from any earlier call so it can't leak into this one.
        if self._detector is not None:
            self._detector.reset()
        self._vad.reset_states()
        # Phase 1: consume frames until the wake word scores high enough.
        if wake_first:
            for frame in frames:
                if stop_check and stop_check():
                    return None
                if max(self._detector.predict(frame).values()) >= self._threshold:
                    break
            else:
                return None
        if on_wake:
            on_wake()
        # Phase 2: record frames until the endpointer decides the command is over.
        endpointer = Endpointer(wait_for_speech=wait_for_speech)
        command = []
        for frame in frames:
            command.append(frame)
            if on_level:
                on_level(audio_level(frame.astype(np.float32) / 32768))
            if endpointer.finished(self._vad.predict(frame, frame_size=FRAME // 2)):
                break
        if not endpointer.heard_speech:
            return np.zeros(0, dtype=np.float32)
        # int16 -> float in [-1, 1], the format the transcriber expects.
        return np.concatenate(command).astype(np.float32) / 32768


def _frames(stream) -> Iterator[np.ndarray]:
    """Endless generator of FRAME-sized mono int16 blocks read from an open input stream."""
    while True:
        frame, _overflowed = stream.read(FRAME)
        yield frame[:, 0]
