import numpy as np
import pytest

from baymax.wake import Endpointer, phrase

SPEECH, SILENCE = 0.9, 0.1
FRAMES_PER_SECOND = 12.5


def frames_until_finished(endpointer: Endpointer, probabilities) -> int | None:
    for count, probability in enumerate(probabilities, start=1):
        if endpointer.finished(probability):
            return count
    return None


def seconds(n: float) -> int:
    return round(n * FRAMES_PER_SECOND)


class TestEndpointer:
    def test_a_command_ends_after_a_second_of_silence(self):
        endpointer = Endpointer()
        talking, then_quiet = [SPEECH] * seconds(2), [SILENCE] * seconds(3)

        finished_at = frames_until_finished(endpointer, talking + then_quiet)

        assert finished_at == seconds(2) + seconds(1)
        assert endpointer.heard_speech

    def test_a_short_pause_mid_sentence_does_not_end_it(self):
        endpointer = Endpointer()
        with_a_pause = [SPEECH] * seconds(1) + [SILENCE] * seconds(0.5) + [SPEECH] * seconds(1)

        assert frames_until_finished(endpointer, with_a_pause) is None

    def test_waking_baymax_and_saying_nothing_gives_up_after_five_seconds(self):
        endpointer = Endpointer()

        assert frames_until_finished(endpointer, [SILENCE] * seconds(10)) == seconds(5)
        assert not endpointer.heard_speech

    def test_someone_who_never_stops_talking_is_cut_off_at_fifteen_seconds(self):
        endpointer = Endpointer()

        assert frames_until_finished(endpointer, [SPEECH] * seconds(30)) == seconds(15)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("hey_jarvis", "hey jarvis"),
        ("models/hey_baymax_v0.1.onnx", "hey baymax"),
        ("C:/wake/hey_baymax.onnx", "hey baymax"),
    ],
)
def test_the_phrase_to_say_comes_from_the_model_name(model, expected):
    assert phrase(model) == expected


class TestStopCheck:
    def test_stop_check_ends_the_wait_before_the_wake_word_fires(self):
        from baymax.wake import WakeWord

        wake = WakeWord.__new__(WakeWord)  # bypass __init__, which loads real models
        wake._detector = type("D", (), {"reset": lambda self: None, "predict": lambda self, f: {"m": 0.0}})()
        wake._vad = type("V", (), {"reset_states": lambda self: None})()
        wake._threshold = 0.5

        calls = []
        def stop_after_three():
            calls.append(1)
            return len(calls) > 3

        result = wake.hear(iter([object()] * 1000), stop_check=stop_after_three)

        assert result is None
        assert len(calls) == 4, "should stop checking (and iterating frames) the moment it returns True"

    def test_stop_check_is_never_consulted_once_the_command_recording_has_started(self):
        # Cutting the user off mid-command would be worse than letting this call finish.
        # wake_first=False skips straight to command-recording, as a cut-in already does.
        from baymax.wake import WakeWord

        wake = WakeWord.__new__(WakeWord)
        wake._detector = type("D", (), {"reset": lambda self: None})()
        wake._vad = type("V", (), {"reset_states": lambda self: None, "predict": lambda self, f, frame_size: 0.0})()
        wake._threshold = 0.5

        stop_check = lambda: True  # would end the call immediately if it were still being checked

        result = wake.hear(iter([np.zeros(1280, dtype=np.int16)] * 3), wake_first=False, stop_check=stop_check)

        assert result is not None, "the command-recording phase should not be cut short by stop_check"
