"""Microphone capture and speaker playback."""

import queue
import sys
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from baymax.tts import Synthesizer

_PLAYBACK_BLOCK_SECONDS = 0.1
_LEVEL_GAIN = 4.0  # raw RMS reads as barely-there for anything short of shouting


def audio_level(samples: np.ndarray) -> float:
    """A 0..1 loudness estimate for a block of float32 audio in [-1, 1], for driving a meter."""
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return min(1.0, rms * _LEVEL_GAIN)


def microphone_name(device: int | str | None, sample_rate: int) -> str:
    """Confirm the microphone can record at ``sample_rate`` and return its name."""
    try:
        sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=sample_rate)
        return sd.query_devices(device, kind="input")["name"]
    except (ValueError, sd.PortAudioError) as exc:
        raise RuntimeError(
            f"can't use microphone {device if device is not None else '(default)'}: {exc}. "
            "List devices with: python -m sounddevice"
        ) from None


def find_input_device(name: str, sample_rate: int) -> int | None:
    """The device number of the MME input device whose name contains ``name``, or None if nothing
    matching is connected right now.

    Built for reconnecting a Bluetooth headset mid-session: its device number can change on
    reconnect, but its name doesn't, so polling this instead of a fixed number survives that. MME
    specifically, because the same physical microphone shows up once per Windows host API (MME,
    DirectSound, WASAPI, WDM-KS) -- matching by name alone finds several and raises an ambiguity
    error rather than picking one -- and MME has been the one that reliably resamples to Baymax's
    16 kHz; WASAPI and WDM-KS have both rejected some devices outright.
    """
    needle = name.lower()
    host_apis = sd.query_hostapis()
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] < 1 or needle not in device["name"].lower():
            continue
        if host_apis[device["hostapi"]]["name"] != "MME":
            continue
        try:
            sd.check_input_settings(device=index, channels=1, dtype="float32", samplerate=sample_rate)
        except (ValueError, sd.PortAudioError):
            continue
        return index
    return None


def record_until_enter(sample_rate: int, device: int | str | None = None) -> np.ndarray:
    """Record mono audio until the user presses Enter."""
    blocks: list[np.ndarray] = []
    with sd.InputStream(
        device=device,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=lambda data, frames, time, status: blocks.append(data.copy()),
    ):
        input()
    return np.concatenate(blocks).ravel() if blocks else np.zeros(0, dtype=np.float32)


class SpeechQueue:
    """Speaks sentences in order on a background thread, and can be cut off mid-sentence."""

    def __init__(self, synthesizer: Synthesizer, on_level: Callable[[float], None] | None = None) -> None:
        self._synthesizer = synthesizer
        self._pending: queue.Queue[tuple[int, str]] = queue.Queue()
        # Bumped by interrupt(); anything queued or playing under an older value is abandoned.
        self._generation = 0
        self._outstanding = 0
        self._idle = threading.Condition()
        self._on_level = on_level
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str) -> None:
        """Queue a sentence to be spoken after everything already queued."""
        if not text:
            return
        # Counted before queueing, so wait() can never see zero while this is still in flight.
        with self._idle:
            self._outstanding += 1
        # Tagged with the current generation, so interrupt() can invalidate it later.
        self._pending.put((self._generation, text))

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been spoken. Returns False if the timeout expired first."""
        with self._idle:
            return self._idle.wait_for(lambda: self._outstanding == 0, timeout)

    def interrupt(self) -> None:
        """Stop talking within a fraction of a second and drop whatever is still queued."""
        self._generation += 1

    def _run(self) -> None:
        """Worker loop: take one queued sentence at a time, synthesize it, and play it."""
        while True:
            generation, text = self._pending.get()
            try:
                if generation == self._generation:
                    self._play(self._synthesizer.synthesize(text), generation)
            except Exception as exc:
                # Keep the worker alive: if it died, wait() would block forever.
                print(f"\n[speech error: {exc}]", file=sys.stderr)
            finally:
                with self._idle:
                    self._outstanding -= 1
                    self._idle.notify_all()

    def _play(self, audio: np.ndarray, generation: int) -> None:
        """Play audio in short blocks, checking between blocks whether it has been interrupted."""
        rate = self._synthesizer.sample_rate
        # Small blocks are what let interrupt() take effect within ~0.1s instead of at sentence end.
        block = int(rate * _PLAYBACK_BLOCK_SECONDS)
        audio = audio.astype(np.float32, copy=False)
        with sd.OutputStream(samplerate=rate, channels=1, dtype="float32") as stream:
            for start in range(0, len(audio), block):
                if generation != self._generation:
                    return
                chunk = audio[start : start + block]
                stream.write(chunk.reshape(-1, 1))
                if self._on_level:
                    self._on_level(audio_level(chunk))
