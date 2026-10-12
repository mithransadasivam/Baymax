"""Unit coverage for the phone server's routes, using a real bottle server on an ephemeral port
and real HTTP requests -- bottle has no bundled test client, so this is plain urllib against a
genuine socket, the same shape of request curl made in manual testing."""

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

import bottle
import pytest

from baymax.phone import _ThreadedWSGIRefServer, build_app, ensure_certificate, lan_address


@pytest.fixture(scope="session")
def certificate(tmp_path_factory):
    """Generating a real RSA key pair isn't free; one certificate is plenty for every test here."""
    cert_dir = tmp_path_factory.mktemp("baymax-test-cert")
    return ensure_certificate(cert_dir, "127.0.0.1")


class FakeBrain:
    def __init__(self, reply: str = "The capital of France is Paris.") -> None:
        self.reply_text = reply
        self.asked: list[str] = []

    def reply(self, text: str):
        self.asked.append(text)
        yield self.reply_text


class FakeTranscriber:
    def __init__(self, text: str = "What is the capital of France?") -> None:
        self.text = text
        self.paths: list[str] = []

    def transcribe(self, path: str) -> str:
        self.paths.append(path)
        return self.text


class FakeSynthesizer:
    def __init__(self) -> None:
        self.synthesized: list[str] = []

    def save_wav(self, text: str, path) -> None:
        self.synthesized.append(text)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 100)


@pytest.fixture
def server(certificate):
    """Starts a real bottle server on a free port for one test, tearing it down afterward."""
    started = {}
    cert_path, _key_path = certificate

    def start(brain=None, transcriber=None, synthesizer="default"):
        brain = brain if brain is not None else FakeBrain()
        transcriber = transcriber if transcriber is not None else FakeTranscriber()
        synthesizer = FakeSynthesizer() if synthesizer == "default" else synthesizer
        app = build_app(
            brain, transcriber, synthesizer, model="llama3.2:3b", host="http://127.0.0.1:11434",
            https_url="https://127.0.0.1:8443", cert_path=cert_path,
        )
        wsgi_server = _ThreadedWSGIRefServer(host="127.0.0.1", port=0, quiet=True)
        thread = threading.Thread(target=wsgi_server.run, args=(app,), daemon=True)
        thread.start()
        for _ in range(100):
            if getattr(wsgi_server, "srv", None) is not None:
                break
            time.sleep(0.02)
        started["server"] = wsgi_server
        return f"http://127.0.0.1:{wsgi_server.port}", brain, transcriber, synthesizer

    yield start
    if "server" in started:
        started["server"].srv.shutdown()


def _post_multipart(url: str, field: str, filename: str, content: bytes) -> dict:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


class TestRoutes:
    def test_the_page_is_served_at_root(self, server):
        base_url, *_ = server()

        with urllib.request.urlopen(f"{base_url}/") as response:
            assert response.status == 200
            assert b"Baymax" in response.read()

    def test_config_reports_the_model_and_host(self, server):
        base_url, *_ = server()

        with urllib.request.urlopen(f"{base_url}/config") as response:
            assert json.loads(response.read()) == {
                "model": "llama3.2:3b", "host": "http://127.0.0.1:11434", "httpsUrl": "https://127.0.0.1:8443",
            }

    def test_a_real_upload_is_transcribed_answered_and_spoken(self, server):
        base_url, brain, transcriber, synthesizer = server()

        result = _post_multipart(f"{base_url}/ask", "audio", "question.webm", b"pretend audio bytes")

        assert result["heard"] == "What is the capital of France?"
        assert result["reply"] == "The capital of France is Paris."
        assert result["audio"], "a WAV should have come back, base64-encoded"
        assert brain.asked == ["What is the capital of France?"]
        assert synthesizer.synthesized == ["The capital of France is Paris."]

    def test_the_uploaded_audio_reaches_the_transcriber_as_a_real_file(self, server):
        base_url, _, transcriber, _ = server()

        _post_multipart(f"{base_url}/ask", "audio", "question.webm", b"pretend audio bytes")

        assert len(transcriber.paths) == 1
        assert transcriber.paths[0].endswith(".webm")

    def test_no_audio_field_is_a_client_error(self, server):
        base_url, *_ = server()
        boundary = uuid.uuid4().hex
        body = f"--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            f"{base_url}/ask", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request)
        assert exc_info.value.code == 400

    def test_silence_that_transcribes_to_nothing_never_reaches_the_model(self, server):
        base_url, brain, transcriber, synthesizer = server(transcriber=FakeTranscriber(text=""))

        result = _post_multipart(f"{base_url}/ask", "audio", "silence.webm", b"pretend silence")

        assert result == {"heard": "", "reply": "", "audio": None}
        assert brain.asked == []
        assert synthesizer.synthesized == []

    def test_quiet_mode_skips_synthesis_but_still_answers(self, server):
        base_url, brain, transcriber, _ = server(synthesizer=None)

        result = _post_multipart(f"{base_url}/ask", "audio", "question.webm", b"pretend audio")

        assert result["reply"] == "The capital of France is Paris."
        assert result["audio"] is None

    def test_the_watch_face_image_is_served_as_a_real_png(self, server):
        base_url, *_ = server()

        with urllib.request.urlopen(f"{base_url}/watchface.png") as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_say_answers_typed_text_with_plain_text_and_no_audio(self, server):
        import urllib.parse

        base_url, brain, transcriber, synthesizer = server()
        request = urllib.request.Request(
            f"{base_url}/say", data=urllib.parse.urlencode({"text": "What is the capital of France?"}).encode(),
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.headers["Content-Type"].startswith("text/plain")
            assert response.read().decode() == "The capital of France is Paris."
        assert brain.asked == ["What is the capital of France?"]
        assert transcriber.paths == [], "text needs no transcription"
        assert synthesizer.synthesized == [], "the watch speaks the reply itself"

    def test_say_also_accepts_a_json_body(self, server):
        base_url, brain, *_ = server()
        request = urllib.request.Request(
            f"{base_url}/say", data=json.dumps({"text": "Hello there"}).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200
        assert brain.asked == ["Hello there"]

    def test_say_with_no_text_is_a_client_error_and_never_reaches_the_model(self, server):
        base_url, brain, *_ = server()
        request = urllib.request.Request(f"{base_url}/say", data=b"text=+", method="POST")

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request)
        assert exc_info.value.code == 400
        assert brain.asked == []

    def test_the_certificate_download_is_a_real_x509_certificate(self, server):
        from cryptography import x509

        base_url, *_ = server()

        with urllib.request.urlopen(f"{base_url}/baymax-cert.cer") as response:
            assert response.headers["Content-Type"] == "application/x-x509-ca-cert"
            cert = x509.load_der_x509_certificate(response.read())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert "127.0.0.1" in {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}

    def test_the_page_works_the_same_way_served_over_https(self, server, certificate):
        # The certificate isn't a real CA as far as urllib is concerned, so this simulates a phone
        # that has already trusted it -- the point being tested is that HTTPS serves the same app.
        import ssl

        cert_path, key_path = certificate
        app = build_app(
            FakeBrain(), FakeTranscriber(), FakeSynthesizer(), model="m", host="h",
            https_url="https://127.0.0.1:0", cert_path=cert_path,
        )
        from baymax.phone import _SSLWSGIRefServer

        wsgi_server = _SSLWSGIRefServer(cert_path, key_path, host="127.0.0.1", port=0, quiet=True)
        thread = threading.Thread(target=wsgi_server.run, args=(app,), daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if getattr(wsgi_server, "srv", None) is not None:
                    break
                time.sleep(0.02)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(f"https://127.0.0.1:{wsgi_server.port}/config", context=ctx) as response:
                assert json.loads(response.read())["model"] == "m"
        finally:
            wsgi_server.srv.shutdown()


class TestConcurrency:
    """The page sat on "thinking" forever because the server took one connection at a time, and a
    browser that opens one and says nothing -- Safari does, routinely -- blocked everything behind it."""

    def test_a_silent_connection_does_not_block_real_requests(self, server):
        import socket

        base_url, *_ = server()
        port = int(base_url.rsplit(":", 1)[1])
        with socket.create_connection(("127.0.0.1", port)):  # connects, then never sends a byte
            with urllib.request.urlopen(f"{base_url}/config", timeout=5) as response:
                assert response.status == 200

    def test_a_silent_connection_does_not_block_https_either(self, certificate):
        import socket
        import ssl

        from baymax.phone import _SSLWSGIRefServer

        cert_path, key_path = certificate
        app = build_app(
            FakeBrain(), FakeTranscriber(), FakeSynthesizer(), model="m", host="h",
            https_url="https://127.0.0.1:0", cert_path=cert_path,
        )
        wsgi_server = _SSLWSGIRefServer(cert_path, key_path, host="127.0.0.1", port=0, quiet=True)
        threading.Thread(target=wsgi_server.run, args=(app,), daemon=True).start()
        try:
            for _ in range(100):
                if getattr(wsgi_server, "srv", None) is not None:
                    break
                time.sleep(0.02)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            # A raw TCP connection that never starts the TLS handshake -- the worse case, since the
            # handshake used to happen inside the accept loop itself.
            with socket.create_connection(("127.0.0.1", wsgi_server.port)):
                with urllib.request.urlopen(
                    f"https://127.0.0.1:{wsgi_server.port}/config", context=ctx, timeout=5
                ) as response:
                    assert response.status == 200
        finally:
            wsgi_server.srv.shutdown()

    def test_replies_still_take_turns_even_though_requests_no_longer_do(self, server):
        # Brain's history isn't safe to mutate from two threads at once, so the lock around it is
        # what keeps concurrent requests from corrupting a conversation.
        active = {"now": 0, "most": 0}
        lock = threading.Lock()

        class SlowBrain(FakeBrain):
            def reply(self, text):
                with lock:
                    active["now"] += 1
                    active["most"] = max(active["most"], active["now"])
                time.sleep(0.15)
                with lock:
                    active["now"] -= 1
                yield "Done."

        base_url, *_ = server(brain=SlowBrain())

        def ask():
            request = urllib.request.Request(f"{base_url}/say", data=b"text=hi", method="POST")
            urllib.request.urlopen(request, timeout=10).read()

        threads = [threading.Thread(target=ask) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert active["most"] == 1


class TestCertificate:
    def test_a_freshly_generated_certificate_covers_the_address_it_was_made_for(self, tmp_path):
        cert_path, key_path = ensure_certificate(tmp_path, "192.168.1.50")

        from cryptography import x509

        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert "192.168.1.50" in {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
        assert key_path.exists()

    def test_a_second_call_for_the_same_address_reuses_the_certificate(self, tmp_path):
        first_cert, first_key = ensure_certificate(tmp_path, "192.168.1.50")
        first_bytes = first_cert.read_bytes()

        second_cert, second_key = ensure_certificate(tmp_path, "192.168.1.50")

        assert second_cert.read_bytes() == first_bytes, "trusting the same certificate again shouldn't be required"

    def test_a_reconnect_landing_on_a_different_ip_gets_a_new_certificate(self, tmp_path):
        first_cert, _ = ensure_certificate(tmp_path, "192.168.1.50")
        first_bytes = first_cert.read_bytes()

        second_cert, _ = ensure_certificate(tmp_path, "192.168.1.99")

        assert second_cert.read_bytes() != first_bytes
        from cryptography import x509

        cert = x509.load_pem_x509_certificate(second_cert.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert "192.168.1.99" in {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}


class TestLanAddress:
    def test_returns_a_plausible_ip_address(self):
        address = lan_address()

        parts = address.split(".")
        assert len(parts) == 4
        assert all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)
