from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _docker_available() -> bool:
    return shutil.which("docker") is not None


@pytest.mark.skipif(not _docker_available(), reason="docker is not available")
def test_docker_container_health_and_dependencies() -> None:
    image_tag = f"document-intelligence-test:{uuid.uuid4().hex[:8]}"
    host_port = _free_port()
    container_id = ""

    try:
        build = subprocess.run(
            ["docker", "build", "-f", "docker/Dockerfile", "-t", image_tag, "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert build.returncode == 0, build.stderr

        run = subprocess.run(
            ["docker", "run", "-d", "-p", f"{host_port}:8000", image_tag],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stderr
        container_id = run.stdout.strip()

        dependency_check = subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "python",
                "-c",
                "import cv2, fastapi, fitz, PIL, pytesseract; print('ok')",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert dependency_check.returncode == 0, dependency_check.stderr

        health_url = f"http://127.0.0.1:{host_port}/health"
        deadline = time.time() + 120
        last_error = ""
        payload: dict[str, object] | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    break
            except Exception as exc:  # pragma: no cover - depends on Docker startup timing
                last_error = str(exc)
                time.sleep(2)

        assert payload is not None, last_error
        assert payload["status"] in {"ok", "degraded"}
        assert "model_loaded" in payload
        assert "ocr_loaded" in payload
    finally:
        if container_id:
            subprocess.run(["docker", "rm", "-f", container_id], cwd=REPO_ROOT, capture_output=True, text=True)
        subprocess.run(["docker", "rmi", "-f", image_tag], cwd=REPO_ROOT, capture_output=True, text=True)
