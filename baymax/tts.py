"""Speech synthesis with Piper."""

import wave
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from piper import PiperVoice

PIPER_VOICES_REPO = "rhasspy/piper-voices"


class Synthesizer:
    """Wraps a loaded Piper voice, producing audio as arrays or WAV files."""

    def __init__(self, voice: str) -> None:
        self._voice = PiperVoice.load(_download_voice(voice))
        # Callers need this to open an output stream at the right rate.
        self.sample_rate: int = self._voice.config.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        """Mono float32 audio for ``text`` (empty if the voice produced nothing)."""
        chunks = [chunk.audio_float_array for chunk in self._voice.synthesize(text)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def save_wav(self, text: str, path: Path) -> None:
        """Speak ``text`` straight into a WAV file at ``path`` instead of the speakers."""
        with wave.open(str(path), "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)


def _download_voice(voice: str) -> Path:
    """Fetch a voice and its config into the Hugging Face cache, returning the model path."""
    try:
        language, name, quality = voice.split("-")
    except ValueError:
        raise ValueError(f"Piper voice names look like 'en_GB-alan-medium', got {voice!r}") from None
    # Mirrors the repo layout: en_GB-alan-medium lives at en/en_GB/alan/medium/.
    folder = f"{language.split('_')[0]}/{language}/{name}/{quality}"
    _fetch(f"{folder}/{voice}.onnx.json")
    return Path(_fetch(f"{folder}/{voice}.onnx"))


def _fetch(filename: str) -> str:
    # Cache first, so a normal startup never touches the network.
    try:
        return hf_hub_download(PIPER_VOICES_REPO, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(PIPER_VOICES_REPO, filename)
