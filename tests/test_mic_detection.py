from baymax import audio


class FakeSd:
    """Stands in for sounddevice with a chosen list of devices, all on the MME host API."""

    class PortAudioError(Exception):
        pass

    def __init__(self, names, unusable=()):
        self._devices = [
            {"name": name, "max_input_channels": channels, "hostapi": 0} for name, channels in names
        ]
        self._unusable = set(unusable)

    def query_hostapis(self):
        return [{"name": "MME"}]

    def query_devices(self, device=None, kind=None):
        return self._devices if device is None else self._devices[device]

    def check_input_settings(self, device, channels, dtype, samplerate):
        if device in self._unusable:
            raise self.PortAudioError("can't record at that rate")


def use(monkeypatch, names, unusable=()):
    monkeypatch.setattr(audio, "sd", FakeSd(names, unusable))


def test_a_real_headset_beats_a_virtual_device_that_sorts_first(monkeypatch):
    use(monkeypatch, [("Microphone (Steam Streaming Mic", 8), ("Headset (WH-1000XM4 Hands-Free", 1)])

    assert audio.find_best_input_device(16000) == 1


def test_virtual_and_loopback_inputs_are_never_chosen(monkeypatch):
    use(monkeypatch, [("Microsoft Sound Mapper - Input", 2), ("Stereo Mix (Realtek)", 2), ("Microphone (Steam Streaming Mic", 8)])

    assert audio.find_best_input_device(16000) is None


def test_a_microphone_that_cannot_record_at_16k_is_skipped(monkeypatch):
    use(monkeypatch, [("Headset (Broken)", 1), ("USB Microphone", 1)], unusable=[0])

    assert audio.find_best_input_device(16000) == 1


def test_outputs_are_ignored(monkeypatch):
    use(monkeypatch, [("Headphones (WH-1000XM4 Stereo)", 0), ("Microphone (Realtek)", 2)])

    assert audio.find_best_input_device(16000) == 1


def test_a_headset_is_preferred_over_a_plain_microphone(monkeypatch):
    use(monkeypatch, [("Microphone (Realtek)", 2), ("Headset Earphone (USB)", 1)])

    assert audio.find_best_input_device(16000) == 1


def test_nothing_connected_means_none(monkeypatch):
    use(monkeypatch, [("Speakers", 0)])

    assert audio.find_best_input_device(16000) is None


def test_a_named_microphone_is_still_found_by_name(monkeypatch):
    use(monkeypatch, [("Microphone (Steam Streaming Mic", 8), ("Headset (WH-1000XM4 Hands-Free", 1)])

    assert audio.find_input_device("xm4", 16000) == 1
    assert audio.find_input_device("nonexistent", 16000) is None


def test_the_app_picks_the_real_mic_when_none_is_named(monkeypatch):
    from baymax import app

    use(monkeypatch, [("Microphone (Steam Streaming Mic", 8), ("Headset (WH-1000XM4 Hands-Free", 1)])

    assert app._resolve_mic(None) == (1, "Headset (WH-1000XM4 Hands-Free")


def test_the_app_still_honours_a_named_mic(monkeypatch):
    from baymax import app

    use(monkeypatch, [("Microphone (Steam Streaming Mic", 8), ("Headset (WH-1000XM4 Hands-Free", 1)])

    assert app._resolve_mic("xm4")[0] == 1


def test_the_app_reports_no_mic_instead_of_crashing_when_none_is_connected(monkeypatch):
    from baymax import app

    use(monkeypatch, [("Speakers", 0)])

    assert app._resolve_mic(None) == (None, None)
    assert app._resolve_mic("XM4") == (None, None)
