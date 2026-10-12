import json
import subprocess

import pytest

from baymax import tailscale


def _done(returncode=0, stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _status(state="Running", name="pc.tail1234.ts.net.", cert_domains=("pc.tail1234.ts.net",)):
    return json.dumps({"BackendState": state, "Self": {"DNSName": name}, "CertDomains": list(cert_domains)})


@pytest.fixture
def cli(monkeypatch):
    """A fake tailscale command: records what it was asked, and answers ``status`` and ``cert``."""
    calls: list[tuple[str, ...]] = []
    state = {"status": _done(stdout=_status()), "cert": _done(), "write_files": True}

    def run(*command):
        calls.append(command)
        if command[1] == "status":
            return state["status"]
        if state["write_files"]:
            cert_file, key_file = command[3], command[5]
            open(cert_file, "w").write("CERT")
            open(key_file, "w").write("KEY")
        return state["cert"]

    monkeypatch.setattr(tailscale, "find_cli", lambda: "tailscale")
    monkeypatch.setattr(tailscale, "_run", run)
    return {"calls": calls, "state": state}


def test_no_tailscale_installed_means_no_tailnet(monkeypatch, tmp_path):
    monkeypatch.setattr(tailscale, "find_cli", lambda: None)

    assert tailscale.certificate(tmp_path) is None


def test_a_signed_in_tailnet_gets_its_name_and_a_certificate(cli, tmp_path):
    tailnet = tailscale.certificate(tmp_path)

    assert tailnet.name == "pc.tail1234.ts.net", "the trailing dot Tailscale reports isn't part of the name"
    assert tailnet.cert.read_text() == "CERT"
    assert tailnet.key.read_text() == "KEY"
    cert_call = cli["calls"][-1]
    assert cert_call[-1] == "pc.tail1234.ts.net", "the certificate must be requested for that exact name"


def test_a_stopped_tailscale_means_no_tailnet(cli, tmp_path):
    cli["state"]["status"] = _done(stdout=_status(state="Stopped"))

    assert tailscale.certificate(tmp_path) is None


def test_https_certificates_not_enabled_means_no_tailnet(cli, tmp_path):
    cli["state"]["status"] = _done(stdout=_status(cert_domains=()))

    assert tailscale.certificate(tmp_path) is None


def test_a_refused_certificate_means_no_tailnet(cli, tmp_path):
    cli["state"]["cert"] = _done(returncode=1)

    assert tailscale.certificate(tmp_path) is None


def test_a_broken_cli_never_stops_baymax_starting(cli, tmp_path, monkeypatch):
    def broken(*command):
        raise subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(tailscale, "_run", broken)

    assert tailscale.certificate(tmp_path) is None


def test_unreadable_status_output_never_stops_baymax_starting(cli, tmp_path):
    cli["state"]["status"] = _done(stdout="not json at all")

    assert tailscale.certificate(tmp_path) is None
