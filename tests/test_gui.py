import sys
import types

import pytest

from baymax.gui import Hud


class FakeWindow:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate_js(self, script: str) -> None:
        self.calls.append(script)


class TestHud:
    def test_calls_before_the_window_is_attached_are_dropped_silently(self):
        Hud().set_state("listening")  # must not raise

    def test_set_state_calls_the_page_s_setstate(self):
        hud, window = Hud(), FakeWindow()
        hud.attach(window)

        hud.set_state("listening")

        assert window.calls == ['setState("listening")']

    def test_text_arguments_are_safely_quoted_for_javascript(self):
        hud, window = Hud(), FakeWindow()
        hud.attach(window)

        hud.set_readout("HEARD", 'He said "stop", then left.')

        assert window.calls == ['setReadout("HEARD", "He said \\"stop\\", then left.")']

    def test_set_config_passes_every_field_in_order(self):
        hud, window = Hud(), FakeWindow()
        hud.attach(window)

        hud.set_config(model="llama3.2:3b", host="http://127.0.0.1:11434", voice="en_GB-alan-medium", wake_phrase="hey jarvis")

        assert window.calls == ['setConfig("llama3.2:3b", "http://127.0.0.1:11434", "en_GB-alan-medium", "hey jarvis")']

    def test_set_mic_calls_the_page_s_setmic(self):
        hud, window = Hud(), FakeWindow()
        hud.attach(window)

        hud.set_mic("Headset (WH-1000XM4)")

        assert window.calls == ['setMic("Headset (WH-1000XM4)")']

    def test_a_dead_window_does_not_crash_the_caller(self):
        class DeadWindow:
            def evaluate_js(self, script: str) -> None:
                raise RuntimeError("window was closed")

        hud = Hud()
        hud.attach(DeadWindow())

        hud.set_state("idle")  # must not raise


class TestTypedInput:
    def test_a_typed_submission_is_tagged_as_typed(self):
        hud = Hud()

        hud.submit("what's the weather like")

        assert hud.wait_for_input(timeout=1) == ("typed", "what's the weather like")

    def test_a_voice_submission_is_tagged_as_voice(self):
        hud = Hud()

        hud.submit_voice("what's the weather like")

        assert hud.wait_for_input(timeout=1) == ("voice", "what's the weather like")

    def test_leading_and_trailing_whitespace_is_stripped(self):
        hud = Hud()

        hud.submit("  hello there  \n")

        assert hud.wait_for_input(timeout=1) == ("typed", "hello there")

    def test_an_empty_submission_is_not_queued_at_all(self):
        hud = Hud()

        hud.submit("   ")

        assert hud.wait_for_input(timeout=0.2) is None

    def test_waiting_with_nothing_submitted_times_out_to_none(self):
        assert Hud().wait_for_input(timeout=0.2) is None

    def test_submissions_are_delivered_in_the_order_they_arrived_regardless_of_source(self):
        hud = Hud()

        hud.submit("first")
        hud.submit_voice("second")

        assert hud.wait_for_input(timeout=1) == ("typed", "first")
        assert hud.wait_for_input(timeout=1) == ("voice", "second")


class _FiringEvent:
    """Stands in for one pywebview event: fires the callback the moment it's added, since these
    tests never wait on a real page load."""

    def __iadd__(self, callback):
        callback()
        return self


class FakeWebviewWindow(FakeWindow):
    def __init__(self) -> None:
        super().__init__()
        self.events = types.SimpleNamespace(loaded=_FiringEvent())
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


def _fake_webview(window, create_window_calls=None):
    """A stand-in for the pywebview module: create_window returns ``window``, and start runs the
    given function immediately instead of opening a real GUI loop."""

    def create_window(*args, **kwargs):
        if create_window_calls is not None:
            create_window_calls.append(kwargs)
        return window

    return types.SimpleNamespace(create_window=create_window, start=lambda fn, **kwargs: fn())


class TestRun:
    def test_the_hud_itself_is_registered_as_the_page_s_js_api(self, monkeypatch):
        # Without this, the window's input box has nothing to call: pywebview.api would be undefined.
        window = FakeWebviewWindow()
        calls = []
        monkeypatch.setitem(sys.modules, "webview", _fake_webview(window, calls))
        from baymax.gui import Hud, run

        run(lambda hud: None)

        assert len(calls) == 1
        assert isinstance(calls[0]["js_api"], Hud)


    def test_the_window_is_destroyed_even_when_the_worker_crashes(self, monkeypatch):
        # This is the actual bug: --gui --text hung forever after typed input ran out, because
        # EOFError killed the background thread and nothing else was ever going to close the window.
        window = FakeWebviewWindow()
        monkeypatch.setitem(sys.modules, "webview", _fake_webview(window))
        from baymax.gui import run

        def worker(hud):
            raise EOFError("stdin closed")

        with pytest.raises(EOFError):
            run(worker)

        assert window.destroyed

    def test_the_window_is_also_destroyed_after_an_ordinary_return(self, monkeypatch):
        window = FakeWebviewWindow()
        monkeypatch.setitem(sys.modules, "webview", _fake_webview(window))
        from baymax.gui import run

        run(lambda hud: None)

        assert window.destroyed
