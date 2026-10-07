"""Command-line entry point: wires speech recognition, the model, and speech output together."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from baymax import emergency
from baymax.brain import Brain
from baymax.memory import DEFAULT_FILE, Memory
from baymax.rag import EMBED_MODEL, Embedder, Retriever
from baymax.research import Library
from baymax.search import lookup
from baymax.text import SentenceBuffer, clean_for_speech

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from baymax.audio import SpeechQueue
    from baymax.gui import Hud
    from baymax.stt import Transcriber
    from baymax.tts import Synthesizer
    from baymax.wake import WakeWord

# Set only while the GUI window is running, so the search/memory announce hooks below -- shared
# with every other mode -- can also mirror themselves onto the HUD without threading a parameter
# through Brain, which is built once in main() before the mode is even chosen.
_active_hud: Hud | None = None


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build the enabled lookups, then run the chosen mode. Returns the exit code."""
    args = _parse_args(argv)
    # Each optional capability is a ``text -> check`` callable, or None when switched off.
    web = None if args.no_search else partial(lookup, announce=_announce_search)
    # Indexing runs in the background, so nothing here waits on it -- and if Ollama has no embedding model,
    # questions quietly fall back to the keyword search over the same curated notes.
    args.library.mkdir(parents=True, exist_ok=True)
    retriever = Retriever(
        Embedder(args.host, args.embed_model),
        args.embed_model,
        library_dir=args.library,
        index_path=DEFAULT_FILE.parent / "library-index.json",
        fallback=Library().lookup,
    )
    retriever.start()
    research = retriever.lookup

    if args.gui:
        # Checking Ollama and loading the model can take a good while right after a reboot, and
        # this is launched with no console to show it in -- so the window opens first and reports
        # its own progress, instead of all of that happening silently before anyone can see it.
        _gui_loop(args, web, research)
        return 0

    # Terminal modes. RuntimeError carries a user-readable fix-it hint; Ctrl+C / end of input is a normal exit.
    try:
        memory = None if args.no_memory else Memory(args.memory, args.host, args.model, on_noted=_announce_notes)
        brain = Brain(model=args.model, host=args.host, lookup=web, research_lookup=research, memory=memory)
        brain.check()  # fail early, before loading anything heavy
        print("Loading model...", flush=True)
        brain.load()
        if memory:
            print(f"Memory: {len(memory.notes)} notes in {args.memory}")
        if args.ask:
            return _answer_recording(args, brain)
        elif args.text:
            _text_loop(brain, None if args.quiet else _start_speech(args.voice))
        else:
            _voice_loop(args, brain)
    except RuntimeError as exc:
        print(f"baymax: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nStanding down.")
    return 0


def _announce_search() -> None:
    """Callback for web.search: signal that a search is about to run."""
    # Printed, never spoken: it explains the pause, and shows which answers came from the web.
    print("(checking the web) ", end="", flush=True)
    if _active_hud:
        _active_hud.set_readout("PROCESSING", "checking the web...")


def _announce_notes(notes: list[str]) -> None:
    """Callback for Memory: show which notes were just taken."""
    # Printed, never spoken, so it's always clear what Baymax is keeping about you.
    print(f"(noted: {' '.join(notes)}) ", end="", flush=True)
    if _active_hud:
        _active_hud.set_readout("NOTED", " ".join(notes))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Define and validate the command line. Every option can also be set by a BAYMAX_* environment
    variable, which is what the defaults below read, so a launcher can configure Baymax without flags."""
    parser = argparse.ArgumentParser(prog="baymax", description="A local, offline voice assistant.")
    parser.add_argument(
        "--model",
        default=os.environ.get("BAYMAX_MODEL", "llama3.1:8b"),
        help="Ollama model to talk to (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("BAYMAX_OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama server URL; point it at another machine to borrow its GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--whisper",
        default=os.environ.get("BAYMAX_WHISPER_MODEL", "base.en"),
        help="faster-whisper model size (default: %(default)s)",
    )
    parser.add_argument(
        "--voice",
        default=os.environ.get("BAYMAX_VOICE", "en_GB-alan-medium"),
        help="Piper voice (default: %(default)s)",
    )
    parser.add_argument(
        "--mic",
        type=lambda value: int(value) if value.isdigit() else value,
        default=os.environ.get("BAYMAX_MIC"),
        help="microphone name (or part of it) or device number; list them with: python -m sounddevice",
    )
    # How input arrives: typed, from a recording, or by wake word. At most one of these.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--text", action="store_true", help="type instead of talking")
    mode.add_argument("--ask", type=Path, metavar="WAV", help="answer a recorded question, then exit")
    mode.add_argument(
        "--wake",
        action="store_true",
        default=os.environ.get("BAYMAX_WAKE") == "1",
        help="hands-free: say the wake word to talk, instead of pressing Enter",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=os.environ.get("BAYMAX_GUI") == "1",
        help="open a HUD window alongside the terminal; hands-free via the wake word, or combine "
        "with --text to type without a microphone",
    )
    parser.add_argument(
        "--wake-model",
        default=os.environ.get("BAYMAX_WAKE_MODEL", "hey_jarvis"),
        help="wake word: a built-in name, or the path to a custom .onnx model (default: %(default)s)",
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=float(os.environ.get("BAYMAX_WAKE_THRESHOLD", "0.5")),
        help="how sure the wake word model must be, from 0 to 1; raise it if Baymax wakes by mistake (default: %(default)s)",
    )
    parser.add_argument("--save", type=Path, metavar="WAV", help="with --ask: write the spoken reply to a file")
    parser.add_argument("--quiet", action="store_true", help="print replies without speaking them")
    parser.add_argument(
        "--no-search",
        action="store_true",
        default=os.environ.get("BAYMAX_SEARCH", "1") == "0",
        help="stay fully offline: never look anything up, even for questions about live information",
    )
    parser.add_argument(
        "--memory",
        type=Path,
        metavar="FILE",
        default=Path(os.environ.get("BAYMAX_MEMORY_FILE", DEFAULT_FILE)),
        help="where Baymax keeps its notes about you (default: %(default)s)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        default=os.environ.get("BAYMAX_MEMORY", "1") == "0",
        help="don't remember anything between sessions, or take new notes",
    )
    parser.add_argument(
        "--library",
        type=Path,
        metavar="DIR",
        default=Path(os.environ.get("BAYMAX_LIBRARY", DEFAULT_FILE.parent / "library")),
        help="a folder of your own documents (.pdf, .txt, .md) for Baymax to learn from; add or change "
        "files while it runs and they're picked up on their own (default: %(default)s)",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("BAYMAX_EMBED_MODEL", EMBED_MODEL),
        help="Ollama model used to search the library by meaning (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    # Combinations argparse can't express itself.
    if args.save and not args.ask:
        parser.error("--save only works with --ask")
    if args.save and args.quiet:
        parser.error("--save and --quiet contradict each other")
    if args.gui and args.ask:
        parser.error("--gui doesn't work with --ask: --ask answers one recording and exits, --gui keeps a window open")
    return args


def _start_speech(voice: str) -> SpeechQueue:
    """Load the voice and start the background speaker. Imported lazily: --quiet never needs Piper."""
    from baymax.audio import SpeechQueue
    from baymax.tts import Synthesizer

    print("Loading voice...", flush=True)
    return SpeechQueue(Synthesizer(voice))


def _text_loop(brain: Brain, speech: SpeechQueue | None) -> None:
    """Terminal chat: read a typed line, answer it, repeat until Ctrl+C or end of input."""
    print("Baymax is online. Type a message, Ctrl+C to quit.")
    while True:
        text = input("\nYou: ").strip()
        if text:
            _respond(brain, text, speech)
            if speech:
                speech.wait()


def _voice_loop(args: argparse.Namespace, brain: Brain) -> None:
    """Terminal voice chat: listen (Enter-to-talk or wake word), transcribe, answer, repeat."""
    from baymax.audio import microphone_name
    from baymax.stt import SAMPLE_RATE, Transcriber

    print(f"Microphone: {microphone_name(args.mic, SAMPLE_RATE)}")
    speech = None if args.quiet else _start_speech(args.voice)
    print("Loading speech recognition...", flush=True)
    transcriber = Transcriber(args.whisper)
    if args.wake:
        from baymax.wake import WakeWord

        print("Loading wake word...", flush=True)
        wake = WakeWord(args.wake_model, args.wake_threshold)
        print(f'Baymax is online. Say "{wake.phrase}", then your question; it stops listening when you do.')
    else:
        print("Baymax is online. Press Enter to talk, Enter again to stop.")
    print("Press Enter while Baymax is talking to cut in. Ctrl+C to quit.")
    # True when the user interrupted the last reply: they're already talking, so the next
    # recording starts immediately instead of waiting for Enter or the wake word.
    cut_in = False
    while True:
        audio = _hear_after_wake_word(wake, args.mic, cut_in) if args.wake else _record_after_enter(args.mic, cut_in)
        text = transcriber.transcribe(audio) if audio.size else ""
        if not text:
            print("(Didn't catch that.)")
            cut_in = False
            continue
        print(f"You: {text}")
        _respond(brain, text, speech)
        cut_in = speech is not None and _wait_unless_cut_in(speech)


def _record_after_enter(mic: int | str | None, cut_in: bool) -> np.ndarray:
    """Push-to-talk: wait for Enter (unless cutting in), then record until Enter again."""
    from baymax.audio import record_until_enter
    from baymax.keys import discard_pending_keys
    from baymax.stt import SAMPLE_RATE

    if not cut_in:
        discard_pending_keys()  # stray keys from earlier would otherwise answer this prompt
        input("\n[Enter] to talk ")
    discard_pending_keys()  # likewise, so only a fresh Enter stops the recording
    print("● Listening... [Enter] to stop", flush=True)
    return record_until_enter(SAMPLE_RATE, mic)


def _hear_after_wake_word(wake: WakeWord, mic: int | str | None, cut_in: bool) -> np.ndarray:
    # After a cut-in the user is already talking, so there's no wake word to wait for.
    if not cut_in:
        print(f'\n(say "{wake.phrase}")', flush=True)
    return wake.listen(mic, on_wake=lambda: print("● Listening...", flush=True), wake_first=not cut_in)


class _StartUp(NamedTuple):
    """What _start_up hands back to the GUI loop. A field is None when that part wasn't needed."""

    brain: Brain
    transcriber: Transcriber | None
    synthesizer: Synthesizer | None
    wake: WakeWord | None


def _start_up(
    args: argparse.Namespace,
    web: Callable[[str], object] | None,
    research: Callable[[str], object] | None,
    hud: Hud,
) -> _StartUp:
    """Everything that has to happen before the GUI can answer its first question, reported to
    the HUD as it goes rather than left to print() -- see _gui_loop for why.

    The model, speech recognition, and the voice don't depend on each other, so they load on
    parallel threads instead of one after another: on a cold start right after a reboot, that's
    the difference between waiting for the sum of every load and waiting for just the slowest
    one. Transcriber and Synthesizer are also built once here and handed to the phone server when
    --phone is combined with --gui, instead of it loading its own separate copy of each.
    """
    from baymax.stt import Transcriber
    from baymax.tts import Synthesizer
    from baymax.wake import WakeWord

    memory = None if args.no_memory else Memory(args.memory, args.host, args.model, on_noted=_announce_notes)
    brain = Brain(model=args.model, host=args.host, lookup=web, research_lookup=research, memory=memory)
    hud.set_readout("STARTING", "checking Ollama...")
    brain.check()  # fails fast, before spinning up threads that would all need this to work

    # The phone always needs a microphone path even under --text (which only turns off the GUI's
    # own mic), so a transcriber is loaded whenever either side of the app will use one.
    need_input = not args.text
    need_output = not args.quiet
    built: dict[str, object] = {}  # results from the loader threads below, keyed by name

    def _load_model() -> None:
        hud.set_readout("STARTING", "loading the model...")
        print("Loading model...", flush=True)
        brain.load()

    def _load_input() -> None:
        if not need_input:
            return
        hud.set_readout("STARTING", "loading speech recognition...")
        print("Loading speech recognition...", flush=True)
        built["transcriber"] = Transcriber(args.whisper)
        # model=None: no trained wake word, just the voice activity detector openWakeWord ships
        # alongside one -- and not needed at all when the GUI itself is never going to listen.
        built["wake"] = None if args.text else WakeWord(model=None)

    def _load_output() -> None:
        if not need_output:
            return
        hud.set_readout("STARTING", "loading the voice...")
        print("Loading voice...", flush=True)
        built["synthesizer"] = Synthesizer(args.voice)

    # Run the three loads in parallel and wait for all of them before continuing.
    jobs = [threading.Thread(target=job) for job in (_load_model, _load_input, _load_output)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()

    if memory:
        print(f"Memory: {len(memory.notes)} notes in {args.memory}")

    transcriber = built.get("transcriber")
    synthesizer = built.get("synthesizer")
    wake = built.get("wake")
    return _StartUp(brain, transcriber, synthesizer, wake)


def _gui_loop(
    args: argparse.Namespace,
    web: Callable[[str], object] | None,
    research: Callable[[str], object] | None,
) -> None:
    """The conversation shown in a HUD window instead of the terminal.

    Everything slow -- checking Ollama, loading the model, loading speech recognition -- happens
    only after the window is already open, reported through the HUD's own readout rather than
    print(): launched from the hidden desktop shortcut, there's no console for print() to reach,
    and a cold Ollama model load right after a reboot can take the better part of a minute with
    nothing to show it's happening otherwise.

    With --text, the window's own input box is the only way in: no microphone is ever touched.
    Otherwise both a spoken greeting and the input box work at once -- say "hey", "hi", "baymax",
    a time-of-day greeting, or type, whichever's easier at the time (see baymax/greeting.py for
    the full list and why a trained wake word isn't used here) -- and a background watcher
    re-detects the microphone every couple of seconds, so reconnecting a headset partway through a
    session picks it back up without restarting Baymax. Point --mic at a name rather than a number
    for that to survive a reconnect: Windows can hand a reconnected device a new number, but its
    name doesn't change.

    No cut-in yet: interrupting Baymax mid-reply isn't supported from either voice or the box.
    """
    from baymax.audio import SpeechQueue
    from baymax.gui import run

    global _active_hud

    def worker(hud: Hud) -> None:
        """Everything that runs inside the window: start-up, then the main answer loop."""
        global _active_hud
        _active_hud = hud
        hud.set_state("thinking")
        try:
            brain, transcriber, synthesizer, wake = _start_up(args, web, research, hud)
        except RuntimeError as exc:
            # Nothing else can show this: there's no console, and the window would otherwise just
            # vanish the instant this function returns, before anyone could read why.
            hud.set_readout("ERROR", str(exc))
            threading.Event().wait()
            return

        if args.text:
            print("Opening the Baymax window. Type into it; close the window to quit.")
        else:
            # Not needing a microphone to exist yet -- only actually listening does, and that's
            # handled by _voice_watcher, which tolerates one not being connected at all.
            print('Opening the Baymax window. Greet it ("hey", "hi", "baymax", ...) or type; close the window to quit.')

        speech = SpeechQueue(synthesizer, on_level=hud.set_level) if synthesizer else None
        hud.set_config(model=args.model, host=args.host, voice=args.voice, wake_phrase="a greeting" if wake else "(typing only)")
        # One-time, not inside the loop below: without this, whatever the last "loading..." line
        # happened to be stays on screen forever, since the loop itself no longer resets it after
        # each reply.
        hud.set_readout("STANDBY", "Ready.")

        # Set while a reply is in progress, so the voice watcher doesn't listen over it.
        busy = threading.Event()
        # Set right after a voice-originated reply finishes, so the watcher's next listen skips
        # straight to capture -- a follow-up shouldn't need a greeting repeated.
        follow_up = threading.Event()
        if wake:
            threading.Thread(target=_voice_watcher, args=(args.mic, wake, transcriber, hud, busy, follow_up), daemon=True).start()
        else:
            hud.set_mic("(typing only, no microphone)")

        try:
            while True:
                hud.set_state("idle")
                # No readout reset here on purpose: the last reply stays on screen until the next
                # question overwrites it, rather than being wiped back to a standby hint the moment
                # each answer finishes.
                heard = hud.wait_for_input()  # blocks until typed or spoken input arrives
                if not heard:
                    continue
                source, text = heard
                busy.set()
                hud.set_state("thinking")
                print(f"You: {text}")
                hud.set_readout("HEARD", text)
                _respond(brain, text, speech, hud=hud)
                if speech:
                    speech.wait()
                hud.set_level(0)
                busy.clear()  # let the voice watcher resume
                if source == "voice":  # typed input never triggers a spoken follow-up window
                    follow_up.set()
        except KeyboardInterrupt:
            # gui.run() always closes the window once this function returns, either way.
            print("\nStanding down.")

    run(worker)
    # run() returns once the window is closed -- but the worker thread pywebview starts for us
    # isn't a daemon, and sits in an endless loop waiting for input. Without this, closing the
    # window left Baymax running with no window, still holding the phone ports and the microphone,
    # and findable only in Task Manager -- and each later launch stacked another one on top.
    sys.stdout.flush()
    os._exit(0)


_MIC_POLL_SECONDS = 2.0  # how often to re-check for a microphone, and how long to back off after an error
_FOLLOW_UP_SECONDS = 6.0  # how long to wait for a follow-up after a spoken reply
_SPEECH_LEVEL_THRESHOLD = 0.08  # meter level (0..1) above which the HUD switches to "listening"


def _speech_reactive(hud: Hud) -> Callable[[float], None]:
    """Wraps hud.set_level so the HUD only shows "listening" once actual speech is heard, not the
    instant a capture cycle starts -- with no wake word, a cycle begins on a plain timer, and
    flashing the state every empty poll would just be visual noise."""
    woken = False  # per capture cycle: the state change fires once, then only the meter updates

    def on_level(level: float) -> None:
        nonlocal woken
        if not woken and level > _SPEECH_LEVEL_THRESHOLD:
            hud.set_state("listening")
            woken = True
        hud.set_level(level)

    return on_level


def _voice_watcher(
    mic_arg: int | str | None,
    wake: WakeWord,
    transcriber: Transcriber,
    hud: Hud,
    busy: threading.Event,
    follow_up: threading.Event,
) -> None:
    """Runs for the life of the window: re-detects the microphone every couple of seconds and, once
    one's connected, waits for speech and transcribes it -- pushing it onto the same queue the
    input box uses only if baymax.greeting recognizes it as addressed to Baymax (skipped for a
    follow-up, which is already known to be). Backs off and retries on any error, which is what
    actually happens when a Bluetooth headset disconnects mid-recording, and pauses around a reply
    already in progress rather than letting two conversations run at once.

    ``follow_up`` sends the next capture straight through with no greeting required, whether that's
    because Baymax just answered something spoken, or because the last thing heard was a greeting
    with nothing after it ("Baymax?") -- either way, a repeated greeting would be redundant.
    """
    from baymax.greeting import strip_greeting

    while True:
        # Step 1: stay out of the way of a reply, and make sure there's a microphone to use.
        if busy.is_set():
            time.sleep(_MIC_POLL_SECONDS)
            continue
        device, label = _resolve_mic(mic_arg)
        if device is None:
            hud.set_mic("(none detected)")
            time.sleep(_MIC_POLL_SECONDS)
            continue
        hud.set_mic(label)
        skip_greeting_check = follow_up.is_set()
        follow_up.clear()
        hud.set_state("idle")
        if skip_greeting_check:
            hud.set_readout("LISTENING", "Go ahead, or stay quiet to go back to standby.")
        # Step 2: capture one utterance.
        try:
            heard = wake.listen(
                device,
                wake_first=False,
                on_level=_speech_reactive(hud),
                stop_check=busy.is_set,
                wait_for_speech=_FOLLOW_UP_SECONDS if skip_greeting_check else 5.0,
            )
        except Exception:
            hud.set_level(0)
            time.sleep(_MIC_POLL_SECONDS)
            continue
        hud.set_level(0)
        if heard.size == 0:  # nothing said, or interrupted because a reply began
            continue
        # Step 3: transcribe, then decide whether it was meant for Baymax.
        hud.set_state("thinking")
        hud.set_readout("PROCESSING", "")
        text = transcriber.transcribe(heard)
        if not text:
            hud.set_state("idle")
            continue

        if skip_greeting_check:
            question = text
        else:
            question = strip_greeting(text)
            if question is None and emergency.check(text):
                # Never wait for a greeting when someone describes an emergency: "I can't breathe"
                # isn't addressed to anyone by name, and it must not be ignored as background chatter.
                question = text
            if question is None:
                # Not addressed to Baymax -- background chatter, someone else's name, the TV.
                # Say nothing and just keep listening.
                continue
            if question == "":
                # "Baymax?" and nothing else: addressed to Baymax, but no question in it yet.
                follow_up.set()
                continue
        hud.submit_voice(question)


def _resolve_mic(mic_arg: int | str | None) -> tuple[int | str | None, str] | tuple[None, None]:
    """The current device for ``mic_arg`` and its display name, or (None, None) if it's not there
    right now. A name is re-searched by name each time, so a reconnect landing on a new device
    number is still found; a number or the system default is just tried as given."""
    from baymax.audio import find_input_device, microphone_name
    from baymax.stt import SAMPLE_RATE

    if isinstance(mic_arg, str):
        device = find_input_device(mic_arg, SAMPLE_RATE)
        if device is None:
            return None, None
        return device, microphone_name(device, SAMPLE_RATE)
    try:
        return mic_arg, microphone_name(mic_arg, SAMPLE_RATE)
    except RuntimeError:
        return None, None


def _wait_unless_cut_in(speech: SpeechQueue) -> bool:
    """Let Baymax finish talking, unless the user presses Enter first. Returns True if they cut in."""
    from baymax.keys import discard_pending_keys, enter_pressed

    # A press made while the model was still thinking would silence Baymax before it said a word.
    discard_pending_keys()
    while not speech.wait(timeout=0.05):
        if enter_pressed():
            speech.interrupt()
            speech.wait()
            print("\n(Cut in.)")
            return True
    return False


def _answer_recording(args: argparse.Namespace, brain: Brain) -> int:
    """--ask mode: transcribe one WAV file, answer it (spoken, or saved with --save), and return an exit code."""
    from baymax.stt import Transcriber

    question = Transcriber(args.whisper).transcribe(str(args.ask))
    if not question:
        print("baymax: no speech found in that recording", file=sys.stderr)
        return 1
    print(f"You: {question}")

    if args.save:
        from baymax.tts import Synthesizer

        reply = _respond(brain, question, speech=None)
        Synthesizer(args.voice).save_wav(clean_for_speech(reply), args.save)
        print(f"Reply saved to {args.save}")
    else:
        speech = None if args.quiet else _start_speech(args.voice)
        _respond(brain, question, speech)
        if speech:
            speech.wait()
    return 0


def _respond(brain: Brain, text: str, speech: SpeechQueue | None, hud: Hud | None = None) -> str:
    """Print the reply as it streams and queue each sentence for speech; returns before speech finishes."""
    print("Baymax: ", end="", flush=True)
    sentences = SentenceBuffer()  # groups streamed tokens into sentences so speech can start early
    reply: list[str] = []
    speaking = False
    for token in brain.reply(text):
        print(token, end="", flush=True)
        reply.append(token)
        if hud:
            if not speaking:
                # The first token is the moment generation actually starts, after any search --
                # exactly when "thinking" should give way to "speaking" in the HUD.
                hud.set_state("speaking")
                speaking = True
            hud.set_readout("REPLY", "".join(reply))
        if speech:
            for sentence in sentences.feed(token):
                speech.say(clean_for_speech(sentence))
    print()
    if speech:
        # Whatever never reached a sentence boundary (the reply's last sentence).
        for sentence in sentences.flush():
            speech.say(clean_for_speech(sentence))
    return "".join(reply)
