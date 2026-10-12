"""A real HTTPS certificate for reaching Baymax from outside the house, via Tailscale.

The phone page needs a secure context for the microphone. At home that meant a self-signed
certificate the phone had to be told to trust; over Tailscale the machine has a proper name
(``pc.tailnet.ts.net``) and Tailscale will fetch a genuine, publicly trusted certificate for it,
so a phone accepts it with no profile to install.

Everything here degrades to "no Tailscale" rather than failing: not installed, not signed in,
HTTPS certificates not switched on for the account, or offline. Baymax starts the same either way.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

_KNOWN_LOCATIONS = (
    r"C:\Program Files\Tailscale\tailscale.exe",
    "/usr/bin/tailscale",
    "/usr/local/bin/tailscale",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
)
_TIMEOUT_SECONDS = 60
# Without this, a Baymax launched with no console flashes one up each time it shells out.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


class Tailnet(NamedTuple):
    """Where this machine is reachable on the tailnet, and the TLS certificate files for that name."""

    name: str
    """This machine's name on the tailnet, e.g. ``desktop.tail1234.ts.net``."""
    cert: Path
    key: Path


def find_cli() -> str | None:
    """Path to the tailscale executable: on PATH if possible, else a standard install location."""
    return shutil.which("tailscale") or next((path for path in _KNOWN_LOCATIONS if Path(path).exists()), None)


def certificate(cert_dir: Path) -> Tailnet | None:
    """This machine's tailnet name and a certificate for it, or None if that isn't available."""
    cli = find_cli()
    if cli is None:
        return None
    try:
        name = _certifiable_name(cli)
        if name is None:
            return None
        cert_dir.mkdir(parents=True, exist_ok=True)
        cert, key = cert_dir / "tailscale-cert.pem", cert_dir / "tailscale-key.pem"
        # Cheap when the cached certificate is still good, and renews it as it nears expiry --
        # which is why this runs on every start instead of once.
        done = _run(cli, "cert", "--cert-file", str(cert), "--key-file", str(key), name)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if done.returncode != 0 or not (cert.exists() and key.exists()):
        return None
    return Tailnet(name, cert, key)


def _certifiable_name(cli: str) -> str | None:
    """This machine's tailnet DNS name if Tailscale is running and allowed to issue a cert for it."""
    done = _run(cli, "status", "--json")
    if done.returncode != 0:
        return None
    status = json.loads(done.stdout)
    if status.get("BackendState") != "Running":
        return None
    # DNS names come back with a trailing dot, which certificates and URLs don't want.
    name = str(status.get("Self", {}).get("DNSName", "")).rstrip(".")
    # CertDomains is empty until HTTPS certificates are enabled in the admin console.
    return name if name and name in (status.get("CertDomains") or []) else None


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    """Run a command to completion with captured text output, a timeout, and no console window."""
    return subprocess.run(
        command, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, creationflags=_NO_WINDOW
    )
