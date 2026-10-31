"""Baymax over the home network: a mobile page that talks to this same machine's Ollama, Whisper,
and Piper, so a phone becomes a remote control rather than a second brain -- nothing runs on the
phone itself, and it only works while this machine is on and reachable on the same network.

Tap-to-talk, not a wake word: recording starts and stops on a deliberate tap, the same way
push-to-talk does on the desktop, so there's no VAD or greeting-word gate needed here at all.

Two servers, one app: browsers refuse to expose the microphone at all -- ``navigator.mediaDevices``
is simply undefined -- on a page that isn't a secure context, and a plain LAN address never is.
The plain HTTP server (``run_http``) exists to serve the page that explains this and hands over a
certificate to trust; once that's done, the same page over HTTPS (``run_https``) is where the
microphone actually works.
"""

from __future__ import annotations

import base64
import datetime
import ipaddress
import ssl
import sys
import tempfile
import threading
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import TYPE_CHECKING
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import bottle

from baymax.text import clean_for_speech
from baymax.vitals import VitalsStore, token_matches

if TYPE_CHECKING:
    from baymax.brain import Brain
    from baymax.stt import Transcriber
    from baymax.tts import Synthesizer

_PAGE = Path(__file__).parent / "phone" / "index.html"  # the tap-to-talk page
_WATCH_FACE = Path(__file__).parent / "assets" / "baymax.png"  # image served for the watch Shortcut
_CERT_VALID_DAYS = 3650  # ten years: a phone only has to be told to trust it once


def lan_address() -> str:
    """This machine's address on the local network, for showing the phone what to open.

    Deliberately doesn't send anything: connecting a UDP socket just asks the OS to pick the
    outbound interface, without a packet ever leaving.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"  # no network at all; still usable from this machine itself


def ensure_certificate(cert_dir: Path, lan_ip: str) -> tuple[Path, Path]:
    """A self-signed certificate covering ``lan_ip``, kept on disk and reused across restarts so a
    phone that has already trusted it doesn't need to trust it again next time Baymax runs.

    Regenerated only when there's no cert yet, it's expired, or it doesn't cover the current
    address -- a DHCP lease can hand this machine a different one than last time.
    """
    cert_path, key_path = cert_dir / "phone-cert.pem", cert_dir / "phone-key.pem"
    if cert_path.exists() and key_path.exists() and _covers(cert_path, lan_ip):
        return cert_path, key_path
    cert_dir.mkdir(parents=True, exist_ok=True)
    _generate_certificate(cert_path, key_path, lan_ip)
    return cert_path, key_path


def _covers(cert_path: Path, lan_ip: str) -> bool:
    """Whether the certificate on disk is unexpired and lists ``lan_ip`` among its addresses."""
    from cryptography import x509

    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        if cert.not_valid_after_utc < datetime.datetime.now(datetime.timezone.utc):
            return False
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        addresses = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
        return lan_ip in addresses
    except Exception:
        return False  # unreadable or malformed counts as "doesn't cover it": just regenerate


def _generate_certificate(cert_path: Path, key_path: Path, lan_ip: str) -> None:
    """Create a new self-signed certificate and private key (PEM) at the given paths, valid for the
    LAN address, 127.0.0.1 and localhost."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Baymax")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))  # backdated a day to absorb clock skew
        .not_valid_after(now + datetime.timedelta(days=_CERT_VALID_DAYS))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.IPAddress(ipaddress.ip_address(lan_ip)),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.DNSName("localhost"),
                ]
            ),
            critical=False,
        )
        # Marked as a CA so a phone can install it as a trusted root, which is how it's "trusted".
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def build_app(
    brain: Brain,
    transcriber: Transcriber,
    synthesizer: Synthesizer | None,
    model: str,
    host: str,
    https_url: str,
    cert_path: Path,
    vitals: VitalsStore | None = None,
    vitals_token: str | None = None,
) -> bottle.Bottle:
    """The routes, wired to one Brain -- kept separate from any other running Brain (the GUI's,
    say), so a phone request can never mutate conversation history another thread is using.

    Shared between the HTTP and HTTPS servers: the page it serves checks which one it's actually
    on and shows the right thing either way.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    # Phones want the certificate in DER form when downloading it to install.
    cert_der = x509.load_pem_x509_certificate(cert_path.read_bytes()).public_bytes(serialization.Encoding.DER)

    app = bottle.Bottle()
    # Requests now run concurrently, but a Brain's conversation history isn't safe to mutate from
    # two threads at once -- so replies take turns, while everything else (uploads, transcription,
    # serving the page) no longer waits on them.
    brain_lock = threading.Lock()

    @app.get("/")
    def page():
        """The tap-to-talk web page itself."""
        return bottle.static_file(_PAGE.name, root=str(_PAGE.parent))

    @app.get("/config")
    def config():
        """Settings the page displays, including where to find the HTTPS version."""
        return {"model": model, "host": host, "httpsUrl": https_url}

    @app.get("/watchface.png")
    def watch_face():
        """Static image for the watch face."""
        return bottle.static_file(_WATCH_FACE.name, root=str(_WATCH_FACE.parent), mimetype="image/png")

    @app.get("/baymax-cert.cer")
    def download_cert():
        """The certificate the phone installs to trust the HTTPS server."""
        bottle.response.content_type = "application/x-x509-ca-cert"
        return cert_der

    @app.post("/ask")
    def ask():
        """Voice question: audio in; the transcript, the reply text and spoken reply audio out."""
        upload = bottle.request.files.get("audio")
        if upload is None:
            bottle.response.status = 400
            return {"error": "no audio uploaded"}

        # Whisper reads from a file path, so the upload is saved to a temp file and always cleaned up.
        # delete=False because the file must be closed before another reader can open it on Windows.
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp:
            upload.save(temp)
            temp_path = temp.name
        try:
            heard = transcriber.transcribe(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        # Nothing intelligible: reply with empties so the page can say it didn't catch that.
        if not heard:
            return {"heard": "", "reply": "", "audio": None}

        print(f"Phone: {heard}", flush=True)
        with brain_lock:
            reply = "".join(brain.reply(heard))
        print(f"Baymax: {reply}", flush=True)
        # The on-screen reply keeps its formatting; only the spoken version is cleaned. No audio
        # when the server was started without a voice (--quiet).
        spoken = clean_for_speech(reply)
        audio_base64 = _synthesize_to_base64(synthesizer, spoken) if synthesizer and spoken else None
        return {"heard": heard, "reply": reply, "audio": audio_base64}

    @app.post("/say")
    def say():
        # For clients that already have text and can only speak it themselves -- an Apple Watch
        # Shortcut dictates on the watch, posts the words here, and reads the reply aloud with the
        # watch's own voice. Plain text back, not JSON, so the Shortcut needs no parsing step.
        text = bottle.request.forms.getunicode("text")
        if text is None and (bottle.request.content_type or "").startswith("application/json"):
            text = (bottle.request.json or {}).get("text")
        text = (text or "").strip()
        if not text:
            bottle.response.status = 400
            bottle.response.content_type = "text/plain; charset=utf-8"
            return "no text sent"
        print(f"Watch: {text}", flush=True)
        with brain_lock:
            reply = "".join(brain.reply(text))
        print(f"Baymax: {reply}", flush=True)
        bottle.response.content_type = "text/plain; charset=utf-8"
        return clean_for_speech(reply)

    @app.post("/vitals")
    def vitals_reading():
        """A heart-rate reading from the Apple Watch Shortcut. Needs the shared token; stores the
        reading locally; with ``explain`` set, also answers "how is my heart rate?" as plain text for
        the watch to read aloud, so one Shortcut does the whole thing."""
        bottle.response.content_type = "text/plain; charset=utf-8"
        if vitals is None or not vitals_token:
            bottle.response.status = 404
            return "heart-rate readings aren't switched on"
        data = bottle.request.json if (bottle.request.content_type or "").startswith("application/json") else None
        data = data if isinstance(data, dict) else {}

        def field(name: str):
            return data.get(name) if name in data else bottle.request.forms.getunicode(name)

        if not token_matches(vitals_token, field("token") or bottle.request.headers.get("X-Baymax-Token")):
            bottle.response.status = 401
            return "wrong or missing token"
        stored = []
        try:
            for name, kind in (("heart_rate", "heart_rate"), ("resting_heart_rate", "resting_heart_rate")):
                value = field(name)
                if value not in (None, ""):
                    stored.append(vitals.add(float(value), kind=kind, ts=_parse_time(field("time"))))
        except (TypeError, ValueError) as exc:
            bottle.response.status = 400
            return f"couldn't use that reading: {exc}"
        if not stored:
            bottle.response.status = 400
            return "send heart_rate and/or resting_heart_rate"
        latest = stored[0]
        print(f"Vitals: {', '.join(f'{r.kind} {r.bpm}' for r in stored)}", flush=True)
        explain = str(field("explain") or "").lower() in ("1", "true", "yes")
        if not explain:
            return f"Got it. {latest.bpm} beats per minute."
        with brain_lock:
            reply = "".join(brain.reply("How is my heart rate right now?"))
        return clean_for_speech(reply)

    return app


def _parse_time(value) -> float | None:
    """Seconds since the epoch from a number or an ISO-8601 date the Shortcut sends, or None for "now"."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.timestamp()


def _synthesize_to_base64(synthesizer: Synthesizer, text: str) -> str:
    """Speak ``text`` to a temporary WAV file and return its bytes base64-encoded, ready to embed in JSON."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        temp_path = temp.name
    try:
        synthesizer.save_wav(text, Path(temp_path))
        return base64.b64encode(Path(temp_path).read_bytes()).decode("ascii")
    finally:
        Path(temp_path).unlink(missing_ok=True)


class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """One thread per connection. wsgiref's own server handles them strictly one at a time, and a
    browser that opens a connection and then says nothing -- which Safari does routinely, to have
    one ready -- left every real request queued behind it, forever: the page sat on "thinking"."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:  # noqa: ANN001 -- matches socketserver
        # A silent connection timing out, or a phone dropping one mid-handshake, is routine here,
        # and there's no console for the traceback to go to anyway.
        if not isinstance(sys.exc_info()[1], (OSError, ssl.SSLError)):
            super().handle_error(request, client_address)


class _QuietHandler(WSGIRequestHandler):
    # Bounds how long any one connection can sit silent, now that each has a thread of its own to
    # tie up. Only counts time spent waiting on the socket, never the model's thinking time.
    timeout = 30

    def log_request(self, *args, **kwargs) -> None:
        pass  # suppress wsgiref's per-request access log lines


class _ThreadedWSGIRefServer(bottle.WSGIRefServer):
    """bottle's stock wsgiref adapter, swapped onto the threaded server and quiet handler above."""

    def run(self, app) -> None:  # noqa: ANN001 -- matches bottle.ServerAdapter's own signature
        self.srv = make_server(self.host, self.port, app, _ThreadingWSGIServer, _QuietHandler)
        self.port = self.srv.server_port  # records the real port if 0 ("any free port") was asked for
        self.srv.serve_forever()


class _SSLWSGIRefServer(_ThreadedWSGIRefServer):
    """bottle has no built-in HTTPS support; this wraps the same wsgiref server it already uses
    with a TLS socket, rather than pulling in a second, heavier server just for this."""

    def __init__(self, cert_path: Path, key_path: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cert_path = cert_path
        self._key_path = key_path

    def run(self, app) -> None:  # noqa: ANN001 -- matches bottle.ServerAdapter's own signature
        self.srv = make_server(self.host, self.port, app, _ThreadingWSGIServer, _QuietHandler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(self._cert_path), keyfile=str(self._key_path))
        # Handshake in each connection's own thread, not in accept(): otherwise a client that
        # connects and never completes one stalls the accept loop itself, threads or no threads.
        self.srv.socket = context.wrap_socket(self.srv.socket, server_side=True, do_handshake_on_connect=False)
        self.port = self.srv.server_port
        self.srv.serve_forever()


def run_http(app: bottle.Bottle, host: str = "0.0.0.0", port: int = 8420) -> None:
    """Blocks serving ``app`` over plain HTTP -- call on a background thread if the calling thread
    is needed for something else (the GUI's own event loop, or run_https on another thread)."""
    bottle.run(app, server=_ThreadedWSGIRefServer(host=host, port=port, quiet=True), quiet=True)


def run_https(app: bottle.Bottle, cert_path: Path, key_path: Path, host: str = "0.0.0.0", port: int = 8443) -> None:
    """Blocks serving ``app`` over HTTPS with a self-signed certificate -- the only way a phone's
    browser will expose the microphone at all. Same threading note as run_http."""
    server = _SSLWSGIRefServer(cert_path, key_path, host=host, port=port, quiet=True)
    bottle.run(app, server=server, quiet=True)
