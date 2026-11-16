"""Request body limits apply before multipart parsing and the disk spool."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _stream(total: int, chunk: int = 64 * 1024):
    sent = 0
    while sent < total:
        size = min(chunk, total - sent)
        yield b"A" * size
        sent += size


def test_chunked_body_over_limit_is_rejected_without_content_length(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "d", max_upload_bytes=1000)
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/mosaics",
            content=_stream(3 * 1024 * 1024),
            headers={"content-type": "multipart/form-data; boundary=abc"},
        )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_too_large"


def test_declared_length_over_limit_is_rejected(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "d2", max_upload_bytes=1000)
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/mosaics",
            content=b"A" * 10,
            headers={
                "content-type": "multipart/form-data; boundary=abc",
                "content-length": "9999999",
            },
        )
    assert response.status_code == 413


def test_normal_body_passes_through_unchanged(client):
    from tests.conftest import make_geotiff, submit_payload

    data, files = submit_payload(
        block_a=make_geotiff([[1]]),
        block_b=make_geotiff([[2]], x0=0.01),
        expected_version=0,
    )
    response = client.post("/api/v1/mosaics", data=data, files=files)
    assert response.status_code == 201, response.text
