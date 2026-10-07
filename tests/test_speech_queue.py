import time

import numpy as np
import pytest

from baymax import audio
from baymax.audio import SpeechQueue

SAMPLE_RATE = 1_000


class FakeSynthesizer:
    sample_rate = SAMPLE_RATE

    def __init__(self, seconds_per_sentence: float) -> None:
        self.seconds = seconds_per_sentence
        self.synthesized: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.synthesized.append(text)
        return np.zeros(int(SAMPLE_RATE * self.seconds), dtype=np.float32)


@pytest.fixture
def played(monkeypatch) -> list[int]:
    """Replaces the speaker with one that plays in real time and records how many samples it played."""
    samples: list[int] = []

    class FakeOutputStream:
        def __init__(self, *, samplerate, **kwargs):
            self.samplerate = samplerate

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, data):
            time.sleep(len(data) / self.samplerate)
            samples.append(len(data))

    monkeypatch.setattr(audio.sd, "OutputStream", FakeOutputStream)
    return samples


def test_speaks_every_sentence_in_order(played):
    synth = FakeSynthesizer(seconds_per_sentence=0.1)
    speech = SpeechQueue(synth)
    for sentence in ["one", "two", "three"]:
        speech.say(sentence)

    assert speech.wait(timeout=5)
    assert synth.synthesized == ["one", "two", "three"]
    assert sum(played) == 3 * 100


def test_wait_with_timeout_reports_speech_still_in_progress(played):
    speech = SpeechQueue(FakeSynthesizer(seconds_per_sentence=1.0))
    speech.say("a long sentence")

    assert speech.wait(timeout=0.1) is False
    speech.interrupt()
    assert speech.wait(timeout=2)


def test_interrupt_stops_mid_sentence_and_drops_the_queue(played):
    synth = FakeSynthesizer(seconds_per_sentence=1.0)
    speech = SpeechQueue(synth)
    for sentence in ["first", "second", "third"]:
        speech.say(sentence)
    time.sleep(0.25)

    started = time.monotonic()
    speech.interrupt()
    assert speech.wait(timeout=2)

    assert time.monotonic() - started < 0.5
    assert sum(played) < SAMPLE_RATE, "the first sentence should have been cut off partway"
    assert synth.synthesized == ["first"], "queued sentences should be dropped without being synthesized"


def test_sentences_queued_after_an_interrupt_are_still_spoken(played):
    synth = FakeSynthesizer(seconds_per_sentence=0.1)
    speech = SpeechQueue(synth)
    speech.say("before")
    speech.interrupt()
    speech.wait(timeout=2)

    speech.say("after")

    assert speech.wait(timeout=2)
    assert synth.synthesized[-1] == "after"


def test_empty_text_is_ignored(played):
    speech = SpeechQueue(FakeSynthesizer(seconds_per_sentence=0.1))
    speech.say("")
    assert speech.wait(timeout=0.5)


class TestAudioLevel:
    def test_silence_reads_as_zero(self):
        assert audio.audio_level(np.zeros(100, dtype=np.float32)) == 0.0

    def test_a_full_scale_tone_reads_near_one(self):
        loud = np.ones(100, dtype=np.float32)
        assert audio.audio_level(loud) == 1.0

    def test_louder_audio_reads_higher(self):
        quiet = np.full(100, 0.05, dtype=np.float32)
        loud = np.full(100, 0.2, dtype=np.float32)
        assert audio.audio_level(quiet) < audio.audio_level(loud)

    def test_an_empty_block_reads_as_zero_instead_of_crashing(self):
        assert audio.audio_level(np.zeros(0, dtype=np.float32)) == 0.0


class FakeLoudSynthesizer(FakeSynthesizer):
    def synthesize(self, text: str) -> np.ndarray:
        self.synthesized.append(text)
        return np.full(int(SAMPLE_RATE * self.seconds), 0.5, dtype=np.float32)


def test_on_level_is_called_with_the_level_of_each_block_played(played):
    levels: list[float] = []
    speech = SpeechQueue(FakeLoudSynthesizer(seconds_per_sentence=0.2), on_level=levels.append)

    speech.say("a phrase loud enough to measure")
    assert speech.wait(timeout=2)

    assert levels, "on_level should have been called at least once"
    assert all(level > 0 for level in levels), "a non-silent block should never read as zero"


class TestFindInputDevice:
    @pytest.fixture
    def devices(self, monkeypatch):
        """A small fake device table: the same physical mic listed once per host API, matching
        what Windows actually reports -- only the MME entry should ever be picked."""
        table = [
            {"name": "Speakers", "max_input_channels": 0, "hostapi": 0},
            {"name": "Headset (WH-1000XM4 Hands-Free)", "max_input_channels": 1, "hostapi": 0},  # MME
            {"name": "Headset (WH-1000XM4 Hands-Free AG Audio)", "max_input_channels": 1, "hostapi": 1},  # DirectSound
            {"name": "Headset (WH-1000XM4 Hands-Free AG Audio)", "max_input_channels": 1, "hostapi": 2},  # WASAPI
            {"name": "Microphone (Realtek)", "max_input_channels": 2, "hostapi": 0},  # MME, wrong device
        ]
        apis = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]
        rejects_16k = {2}  # WASAPI entry rejects Baymax's sample rate, matching real hardware this week

        def check(device, channels, dtype, samplerate):
            if device in rejects_16k:
                raise audio.sd.PortAudioError("rejected")

        monkeypatch.setattr(audio.sd, "query_devices", lambda: table)
        monkeypatch.setattr(audio.sd, "query_hostapis", lambda: apis)
        monkeypatch.setattr(audio.sd, "check_input_settings", check)
        return table

    def test_only_the_mme_entry_is_returned_even_though_the_name_matches_three_devices(self, devices):
        assert audio.find_input_device("XM4", 16000) == 1

    def test_matching_is_case_insensitive(self, devices):
        assert audio.find_input_device("xm4", 16000) == 1

    def test_a_name_that_matches_nothing_returns_none(self, devices):
        assert audio.find_input_device("nonexistent headset", 16000) is None

    def test_a_device_that_rejects_the_sample_rate_is_skipped(self, devices, monkeypatch):
        # Move the MME match to the one that rejects 16 kHz, and confirm it's passed over rather
        # than returned anyway.
        monkeypatch.setattr(
            audio.sd, "check_input_settings",
            lambda device, channels, dtype, samplerate: (_ for _ in ()).throw(audio.sd.PortAudioError("no"))
            if device == 1 else None,
        )
        assert audio.find_input_device("XM4", 16000) is None

    def test_output_only_devices_are_never_matched(self, devices):
        assert audio.find_input_device("Speakers", 16000) is None
