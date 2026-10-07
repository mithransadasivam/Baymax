# Entry point for `python -m baymax`: run the app and use its return value as the exit code.
from baymax.app import main

raise SystemExit(main())
