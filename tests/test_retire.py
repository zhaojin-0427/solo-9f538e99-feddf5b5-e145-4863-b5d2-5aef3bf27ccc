"""Retirement: supersede rule, metadata retention, 410, slow downloads, recovery."""

from __future__ import annotations

import hashlib
import threading

import numpy as np
from fastapi.testclient import TestClient

from app import store
from app.main import create_app
from tests.conftest import make_geotiff, read_band, submit_payload

A_TILE = np.array([[1, 2], [3, 4]], dtype=np.int16)
B_TILE = np.array([[5, 6], [7, 8]], dtype=np.int16)

# Retired version 1 produced by A_TILE/B_TILE, used by several tests.
V1_PIXELS = [[1, 5, 6], [3, 7, 8]]


def _submit(client, *, expected_version: int, day: str | None = None, a=A_TILE, b=B_TILE):
    kwargs = {"day": day} if day is not None else {}
    data, files = submit_payload(
        block_a=make_geotiff(a),
        block_b=make_geotiff(b, x0=0.01),
        expected_version=expected_version,
        **kwargs,
    )
    return client.post("/api/v1/mosaics", data=data, files=files)


def _publish_two(client, *, day: str | None = None):
    assert _submit(client, expected_version=0, day=day).status_code == 201
    second = _submit(
        client,
        expected_version=1,
        day=day,
        a=np.full((2, 2), 11, dtype=np.int16),
        b=np.full((2, 2), 12, dtype=np.int16),
    )
    assert second.status_code == 201, second.text
    return second


def test_retire_requires_a_newer_same_day_version(client):
    response = _submit(client, expected_version=0)
    assert response.status_code == 201
    old_bytes = client.get("/api/v1/mosaics/1/download").content

    rejected = client.post("/api/v1/mosaics/1/retire")
    assert rejected.status_code == 409
    detail = rejected.json()["detail"]
    assert detail["code"] == "not_superseded"
    assert detail["version"] == 1
    assert detail["day"] == "2026-01-01"
    assert detail["latest_version"] == 1

    # Nothing changed: same metadata, same bytes, still current.
    body = client.get("/api/v1/mosaics/1").json()["version"]
    assert body["retired"] is False
    assert body["reclaim_status"] is None
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 1
    assert client.get("/api/v1/mosaics/1/download").content == old_bytes
    assert (client.app.state.settings.versions_dir / "v1.tif").is_file()


def test_newer_version_on_another_day_does_not_allow_retirement(client):
    assert _submit(client, expected_version=0, day="2026-01-01").status_code == 201
    assert _submit(client, expected_version=1, day="2026-01-02").status_code == 201

    rejected = client.post("/api/v1/mosaics/1/retire")
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "not_superseded"
    assert client.get("/api/v1/mosaics/1/download").status_code == 200


def test_retire_keeps_metadata_and_sha256_and_returns_410(client):
    _publish_two(client)
    before = client.get("/api/v1/mosaics/1").json()["version"]
    digest = before["image"]["sha256"]
    assert before["retired"] is False
    assert before["downloadable"] is True

    response = client.post(
        "/api/v1/mosaics/1/retire",
        json={"retired_by": "ops", "reason": "superseded by v2"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["retired"] is True
    assert body["reused"] is False
    assert body["reclaim_status"] == "reclaimed"
    assert body["retirement"]["retired_by"] == "ops"
    assert body["retirement"]["reason"] == "superseded by v2"
    assert body["retirement"]["retired_at"].endswith("Z")

    after = client.get("/api/v1/mosaics/1").json()["version"]
    assert after["retired"] is True
    assert after["reclaim_status"] == "reclaimed"
    assert after["downloadable"] is False
    assert after["image"]["sha256"] == digest
    assert after["image"]["size_bytes"] == before["image"]["size_bytes"]
    assert after["image"]["width"] == before["image"]["width"]
    assert after["blocks"] == before["blocks"]
    assert after["published_at"] == before["published_at"]
    assert after["retirement"]["retired_at"] == body["retirement"]["retired_at"]
    assert (client.app.state.settings.versions_dir / "v1.tif").exists() is False

    gone = client.get("/api/v1/mosaics/1/download")
    assert gone.status_code == 410
    assert gone.json()["detail"]["code"] == "version_retired"
    assert gone.headers["cache-control"] == "no-store"

    # The newer version is untouched, remains current and downloadable.
    assert client.get("/api/v1/mosaics/2/download").status_code == 200
    assert client.get("/api/v1/mosaics/2").json()["version"]["retired"] is False
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 2


def test_retire_is_idempotent_and_reuses_the_record(client):
    _publish_two(client)
    first = client.post("/api/v1/mosaics/1/retire")
    assert first.status_code == 200
    second = client.post("/api/v1/mosaics/1/retire")
    assert second.status_code == 200
    assert second.json()["reused"] is True
    assert second.json()["retirement"] == first.json()["retirement"]
    assert second.json()["reclaim_status"] == "reclaimed"

    # A repeated request never rewrites the original record.
    third = client.post("/api/v1/mosaics/1/retire", json={"retired_by": "someone-else"})
    assert third.status_code == 200
    assert third.json()["reused"] is True
    assert third.json()["retirement"]["retired_by"] is None


def test_retire_unknown_version_is_404(client):
    response = client.post("/api/v1/mosaics/99/retire")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "version_not_found"


def test_listing_includes_retired_versions_and_can_filter_them(client):
    _publish_two(client)
    assert client.post("/api/v1/mosaics/1/retire").status_code == 200

    everything = client.get("/api/v1/mosaics").json()
    assert [item["version"] for item in everything["items"]] == [2, 1]
    assert [item["retired"] for item in everything["items"]] == [False, True]

    active = client.get("/api/v1/mosaics", params={"include_retired": False}).json()
    assert [item["version"] for item in active["items"]] == [2]
    assert active["total"] == 1


def test_download_admitted_before_retirement_reads_all_original_bytes(client, settings):
    _publish_two(client)
    expected = client.get("/api/v1/mosaics/1/download").content

    with client.stream("GET", "/api/v1/mosaics/1/download") as response:
        assert response.status_code == 200
        assert response.headers["etag"] == f'"{hashlib.sha256(expected).hexdigest()}"'
        # The response is admitted but its body has not been consumed yet.
        retire = client.post("/api/v1/mosaics/1/retire")
        assert retire.status_code == 200
        assert retire.json()["reclaim_status"] == "reclaimed"
        assert not (settings.versions_dir / "v1.tif").exists()

        body = b"".join(response.iter_bytes())

    assert body == expected
    values, _ = read_band(body)
    assert values.tolist() == V1_PIXELS
    # New downloads are refused from now on.
    assert client.get("/api/v1/mosaics/1/download").status_code == 410


def test_slow_download_does_not_block_publishing_or_retiring_another_version(client):
    assert _submit(client, expected_version=0, day="2026-01-01").status_code == 201
    assert _submit(client, expected_version=1, day="2026-01-02").status_code == 201
    assert _submit(client, expected_version=2, day="2026-01-02").status_code == 201
    original = client.get("/api/v1/mosaics/1/download").content

    with client.stream("GET", "/api/v1/mosaics/1/download") as slow:
        # The body is stalled at the first chunk, yet other writes proceed.
        retired = client.post("/api/v1/mosaics/2/retire")
        assert retired.status_code == 200, retired.text
        assert retired.json()["reclaim_status"] == "reclaimed"

        published = _submit(client, expected_version=3, day="2026-01-03")
        assert published.status_code == 201, published.text

        body = b"".join(slow.iter_bytes())

    assert body == original
    assert client.get("/api/v1/mosaics/2/download").status_code == 410
    assert client.get("/api/v1/mosaics/4/download").status_code == 200
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 4


def test_retirement_registration_failure_leaves_version_downloadable(client, monkeypatch):
    _publish_two(client)
    before = client.get("/api/v1/mosaics/1/download").content

    def failing_insert(*args, **kwargs):
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr(store, "_insert_retirement", failing_insert)
    response = client.post("/api/v1/mosaics/1/retire")

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "retire_failed"

    metadata = client.get("/api/v1/mosaics/1").json()["version"]
    assert metadata["retired"] is False
    assert metadata["reclaim_status"] is None
    assert client.get("/api/v1/mosaics/1/download").content == before
    assert (client.app.state.settings.versions_dir / "v1.tif").is_file()

    # Retrying after the failure works.
    monkeypatch.undo()
    retried = client.post("/api/v1/mosaics/1/retire")
    assert retried.status_code == 200
    assert retried.json()["reused"] is False


def test_unlink_failure_keeps_version_unavailable_and_pending_then_recovers_on_restart(
    settings, restartable_client, monkeypatch
):
    first = restartable_client()
    _publish_two(first)
    real_unlink = store._unlink_file

    def failing_unlink(path):
        raise PermissionError("simulated volume failure")

    monkeypatch.setattr(store, "_unlink_file", failing_unlink)
    response = first.post("/api/v1/mosaics/1/retire")
    monkeypatch.setattr(store, "_unlink_file", real_unlink)

    assert response.status_code == 200, response.text
    assert response.json()["reclaim_status"] == "pending"
    assert response.json()["metadata"]["retirement"]["reclaim_error"]
    # Registration succeeded: it is retired, so no new downloads even though
    # the bytes are still on disk.
    assert first.get("/api/v1/mosaics/1/download").status_code == 410
    assert first.get("/api/v1/mosaics/1").json()["version"]["reclaim_status"] == "pending"
    assert (settings.versions_dir / "v1.tif").is_file()
    # The other version is untouched.
    assert first.get("/api/v1/mosaics/2/download").status_code == 200
    first.__exit__(None, None, None)

    restarted = restartable_client()
    metadata = restarted.get("/api/v1/mosaics/1").json()["version"]
    assert metadata["retired"] is True
    assert metadata["reclaim_status"] == "reclaimed"
    assert metadata["retirement"]["reclaim_error"] is None
    assert not (settings.versions_dir / "v1.tif").exists()
    assert restarted.get("/api/v1/mosaics/1/download").status_code == 410
    assert restarted.get("/api/v1/mosaics/2/download").status_code == 200
    # Recovery is idempotent across another restart.
    restarted.__exit__(None, None, None)
    again = restartable_client()
    assert again.get("/api/v1/mosaics/1").json()["version"]["reclaim_status"] == "reclaimed"
    assert again.get("/api/v1/mosaics/2/download").status_code == 200


def test_restart_with_pending_reclaim_never_deletes_other_versions(settings):
    with TestClient(create_app(settings)) as client:
        _publish_two(client)

    # Simulate a crash after the retirement commit but before the unlink.
    conn = store.connect(settings)
    try:
        conn.execute(
            "INSERT INTO retirements (version, day, retired_at, reclaim_status) "
            "VALUES (1, '2026-01-01', '2026-01-01T00:00:00.000000Z', 'pending')"
        )
    finally:
        conn.close()
    assert (settings.versions_dir / "v1.tif").is_file()
    v2_bytes = (settings.versions_dir / "v2.tif").read_bytes()

    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/mosaics/1").json()["version"]["reclaim_status"] == "reclaimed"
        assert not (settings.versions_dir / "v1.tif").exists()
        assert (settings.versions_dir / "v2.tif").read_bytes() == v2_bytes
        assert client.get("/api/v1/mosaics/2/download").status_code == 200
        assert client.get("/api/v1/mosaics/1/download").status_code == 410


def test_publishing_and_downloading_active_versions_is_unchanged(client):
    _publish_two(client)
    current = client.get("/api/v1/mosaics/current").json()
    assert current["current_version"] == 2
    assert current["version"]["retired"] is False
    assert current["version"]["reclaim_status"] is None
    download = client.get("/api/v1/download/current")
    assert download.status_code == 200
    values, _ = read_band(download.content)
    assert values.tolist() == [[11, 12, 12], [11, 12, 12]]


def test_head_and_range_requests_on_active_version(client):
    assert _submit(client, expected_version=0).status_code == 201
    full = client.get("/api/v1/mosaics/1/download").content

    head = client.head("/api/v1/mosaics/1/download")
    assert head.status_code == 200
    assert head.headers["content-length"] == str(len(full))
    assert head.content == b""

    partial = client.get("/api/v1/mosaics/1/download", headers={"Range": "bytes=0-9"})
    assert partial.status_code == 206
    assert partial.content == full[:10]
    assert partial.headers["content-range"] == f"bytes 0-9/{len(full)}"

    unsat = client.get("/api/v1/mosaics/1/download", headers={"Range": "bytes=999999-"})
    assert unsat.status_code == 416
    assert unsat.headers["content-range"] == f"bytes */{len(full)}"


def test_concurrent_retirement_requests_produce_one_record(client):
    _publish_two(client)
    barrier = threading.Barrier(2)

    def attempt():
        barrier.wait()
        return client.post("/api/v1/mosaics/1/retire")

    with __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=2) as pool:
        responses = [future.result() for future in [pool.submit(attempt) for _ in range(2)]]

    assert sorted(r.status_code for r in responses) == [200, 200]
    assert sorted(r.json()["reused"] for r in responses) == [False, True]
    assert responses[0].json()["retirement"] == responses[1].json()["retirement"]


def test_retirement_reports_success_when_commit_landed_despite_an_error(client, monkeypatch):
    _publish_two(client)
    real_commit = store._commit

    def commit_then_raise(conn):
        real_commit(conn)
        raise RuntimeError("connection dropped after COMMIT")

    monkeypatch.setattr(store, "_commit", commit_then_raise)
    response = client.post("/api/v1/mosaics/1/retire")
    monkeypatch.setattr(store, "_commit", real_commit)

    # The write is durable, so the API must not claim the version is unchanged.
    assert response.status_code == 200, response.text
    assert response.json()["retired"] is True
    assert client.get("/api/v1/mosaics/1/download").status_code == 410
    assert client.get("/api/v1/mosaics/1").json()["version"]["retired"] is True


def test_retire_chain_of_superseded_versions(client):
    assert _submit(client, expected_version=0).status_code == 201
    assert _submit(client, expected_version=1).status_code == 201
    assert _submit(client, expected_version=2).status_code == 201

    # v1 and v2 are both superseded by v3; v3 is the newest for the day.
    assert client.post("/api/v1/mosaics/1/retire").status_code == 200
    assert client.post("/api/v1/mosaics/2/retire").status_code == 200
    assert client.post("/api/v1/mosaics/3/retire").status_code == 409
    listing = client.get("/api/v1/mosaics").json()
    assert [(i["version"], i["retired"]) for i in listing["items"]] == [
        (3, False),
        (2, True),
        (1, True),
    ]
    assert client.get("/api/v1/mosaics/3/download").status_code == 200


def test_existing_data_volume_without_retirement_tables_is_upgraded(settings):
    """A volume written by the pre-retirement schema keeps working."""
    import hashlib
    import json
    import sqlite3

    settings.ensure_directories()
    image = b"legacy-image-bytes"
    (settings.versions_dir / "v1.tif").write_bytes(b"older-legacy")
    (settings.versions_dir / "v2.tif").write_bytes(image)
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE versions (
                version INTEGER PRIMARY KEY, day TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('ready')),
                published_at TEXT NOT NULL, width INTEGER NOT NULL,
                height INTEGER NOT NULL, crs_wkt TEXT NOT NULL,
                crs_label TEXT NOT NULL, transform TEXT NOT NULL,
                bounds TEXT NOT NULL, res_x REAL NOT NULL, res_y REAL NOT NULL,
                dtype TEXT NOT NULL, nodata INTEGER NOT NULL,
                sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                block_meta TEXT NOT NULL);
            CREATE INDEX versions_day_idx ON versions(day);
            CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        conn.execute("INSERT INTO state VALUES ('current_version', '2')")
        for version, payload in ((1, b"older-legacy"), (2, image)):
            conn.execute(
                "INSERT INTO versions VALUES (?, '2025-12-31', 'ready', ?, 1, 1, "
                "'WKT', 'EPSG:4326', ?, ?, 0.01, 0.01, 'int16', -32768, ?, ?, ?)",
                (
                    version,
                    f"2025-12-31T2{version}:00:00.000000Z",
                    json.dumps([0.01, 0, 0, 0, -0.01, 0.02]),
                    json.dumps([0, 0, 0.02, 0.02]),
                    hashlib.sha256(payload).hexdigest(),
                    len(payload),
                    "[]",
                ),
            )
        conn.commit()
    finally:
        conn.close()

    with TestClient(create_app(settings)) as client:
        # Untouched versions behave exactly as before.
        assert client.get("/healthz").json()["current_version"] == 2
        current = client.get("/api/v1/mosaics/2").json()["version"]
        assert current["retired"] is False
        assert current["downloadable"] is True
        download = client.get("/api/v1/mosaics/2/download")
        assert download.status_code == 200
        assert download.content == image

        # Retirement works against the upgraded volume.
        retired = client.post("/api/v1/mosaics/1/retire")
        assert retired.status_code == 200, retired.text
        assert retired.json()["reclaim_status"] == "reclaimed"
        assert client.get("/api/v1/mosaics/1/download").status_code == 410
        assert client.get("/api/v1/mosaics/2/download").content == image

        # Publishing still works and becomes current.
        published = _submit(client, expected_version=2)
        assert published.status_code == 201, published.text
        assert client.get("/api/v1/mosaics/current").json()["current_version"] == 3


def test_openapi_documents_retire_endpoint(client):
    schema = client.get("/openapi.json").json()
    path = schema["paths"]["/api/v1/mosaics/{version}/retire"]
    assert "post" in path
    assert "200" in path["post"]["responses"]
    assert "409" in path["post"]["responses"]
    assert client.get("/docs").status_code == 200
