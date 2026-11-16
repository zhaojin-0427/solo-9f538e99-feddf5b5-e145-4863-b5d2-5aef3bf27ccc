"""End-to-end concurrency against a real uvicorn server.

TestClient buffers through an in-process transport; this module starts the app
on a socket so a slow client genuinely reads the body in chunks while other
requests are served, which is the behaviour the retirement design relies on.
"""

from __future__ import annotations

import hashlib
import socket
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn

from app.config import Settings
from app.main import create_app
from tests.conftest import make_geotiff, submit_payload


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def live_server(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        if time.time() > deadline:  # pragma: no cover - startup failure
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}", settings
    finally:
        server.should_exit = True
        thread.join(timeout=30)


def _submit(http: httpx.Client, *, expected_version: int, a, b, day: str = "2026-01-01"):
    data, files = submit_payload(
        block_a=make_geotiff(a),
        block_b=make_geotiff(b, x0=0.01),
        expected_version=expected_version,
        day=day,
    )
    return http.post("/api/v1/mosaics", data=data, files=files)


def test_slow_real_download_completes_after_retirement_and_others_proceed(live_server):
    base, settings = live_server
    with httpx.Client(base_url=base, timeout=60.0) as http:
        # v1 and v2 on day one; v3 and v4 on day two.
        assert _submit(http, expected_version=0, a=np.array([[1, 2], [3, 4]]), b=np.array([[5, 6], [7, 8]])).status_code == 201
        assert _submit(http, expected_version=1, a=np.full((2, 2), 21, np.int16), b=np.full((2, 2), 22, np.int16)).status_code == 201
        assert _submit(http, expected_version=2, a=np.zeros((2, 2), np.int16), b=np.ones((2, 2), np.int16), day="2026-01-02").status_code == 201
        assert _submit(http, expected_version=3, a=np.full((2, 2), 31, np.int16), b=np.full((2, 2), 32, np.int16), day="2026-01-02").status_code == 201
        expected = http.get("/api/v1/mosaics/1/download").content
        first_chunk = threading.Event()
        release = threading.Event()
        received = bytearray()

        def slow_read():
            with http.stream("GET", "/api/v1/mosaics/1/download") as response:
                assert response.status_code == 200
                iterator = response.iter_raw(8)
                received.extend(next(iterator))
                first_chunk.set()
                assert release.wait(timeout=30), "download was never released"
                for chunk in iterator:
                    received.extend(chunk)

        reader = threading.Thread(target=slow_read)
        reader.start()
        assert first_chunk.wait(timeout=30), "download never started"

        try:
            # While the body is stalled: retire the other day's old version and
            # publish a new one. Neither may wait on the open download.
            retired = http.post("/api/v1/mosaics/3/retire")
            assert retired.status_code == 200, retired.text
            assert retired.json()["reclaim_status"] == "reclaimed"
            assert _submit(http, expected_version=4, a=np.ones((2, 2), np.int16), b=np.zeros((2, 2), np.int16), day="2026-01-03").status_code == 201

            # The stalled version's own retirement is independent as well.
            assert not (settings.versions_dir / "v3.tif").exists()
            retired_v1 = http.post("/api/v1/mosaics/1/retire")
            assert retired_v1.status_code == 200, retired_v1.text
            assert not (settings.versions_dir / "v1.tif").exists()
            assert http.get("/api/v1/mosaics/1/download").status_code == 410
        finally:
            release.set()
        reader.join(timeout=60)
        assert not reader.is_alive()

    assert bytes(received) == expected


def test_real_download_after_restart_of_server(live_server):
    base, settings = live_server
    with httpx.Client(base_url=base, timeout=30.0) as http:
        assert _submit(http, expected_version=0, a=np.array([[3]]), b=np.array([[4]])).status_code == 201
        before = http.get("/api/v1/mosaics/1/download").content
        assert _submit(http, expected_version=1, a=np.array([[7]]), b=np.array([[8]])).status_code == 201
        assert http.post("/api/v1/mosaics/1/retire").status_code == 200
    # Same process, application restarted on the same data dir.
    app = create_app(settings)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert time.time() < deadline
        time.sleep(0.02)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0) as http:
            metadata = http.get("/api/v1/mosaics/1").json()["version"]
            assert metadata["retired"] is True
            assert metadata["reclaim_status"] == "reclaimed"
            assert metadata["image"]["sha256"] == hashlib.sha256(before).hexdigest()
            assert http.get("/api/v1/mosaics/1/download").status_code == 410
            assert http.get("/api/v1/mosaics/2/download").status_code == 200
    finally:
        server.should_exit = True
        thread.join(timeout=30)
