"""Non-blocking Enter detection, so the user can cut in while Baymax is talking.

A terminal holds on to keys pressed while a program isn't reading, and hands them
to the next prompt. Left alone, one stray Enter during a reply shifts every
later press out of step: "start talking" stops the recording, and "stop" starts it.
"""

import sys

if sys.platform == "win32":
    import msvcrt

    def enter_pressed() -> bool:
        """True if Enter was among the keys pressed since the last call; never blocks."""
        pressed = False
        # Drain the whole buffer rather than stopping at the first key, so none carry over.
        while msvcrt.kbhit():
            pressed |= msvcrt.getwch() in "\r\n"
        return pressed

    def discard_pending_keys() -> None:
        """Throw away any keys typed while nothing was reading them."""
        while msvcrt.kbhit():
            msvcrt.getwch()

else:
    # POSIX equivalents of the two functions above, built on select() and termios.
    import select
    import termios

    def enter_pressed() -> bool:
        """True if a line was entered since the last call; never blocks."""
        # A zero timeout makes select() a pure poll.
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return False
        # Readable but empty means stdin was closed, not that Enter was pressed.
        if not sys.stdin.readline():
            raise EOFError
        return True

    def discard_pending_keys() -> None:
        """Throw away any keys typed while nothing was reading them."""
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
