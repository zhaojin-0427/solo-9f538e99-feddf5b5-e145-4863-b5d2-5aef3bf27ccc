"""API behaviour: publishing, version pinning, conflicts, durability, validation."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app import rasters, store
from app.rasters import NODATA
from tests.conftest import make_geotiff, read_band, submit_payload

A_TILE = np.array([[1, 2], [3, 4]], dtype=np.int16)
B_TILE = np.array([[NODATA, 2], [NODATA, 4]], dtype=np.int16)


def _submit(client, *, expected_version: int, a=A_TILE, b=B_TILE, **kwargs):
    data, files = submit_payload(
        block_a=make_geotiff(a),
        block_b=make_geotiff(b, x0=0.01),
        expected_version=expected_version,
        **kwargs,
    )
    return client.post("/api/v1/mosaics", data=data, files=files)


def test_docs_and_health_available(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "current_version": 0}
    assert client.get("/", follow_redirects=False).status_code in (307, 308)


def test_submit_publishes_version_and_is_downloadable(client):
    response = _submit(client, expected_version=0)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["version"] == 1
    assert body["day"] == "2026-01-01"
    assert body["status"] == "ready"
    assert body["image"]["width"] == 3
    assert body["image"]["height"] == 2
    assert body["image"]["nodata"] == NODATA
    assert body["image"]["dtype"] == "int16"
    assert body["image"]["crs"] == "EPSG:4326"
    assert body["download_url"] == "/api/v1/mosaics/1/download"
    assert {block["block_no"] for block in body["blocks"]} == {1, 2}

    current = client.get("/api/v1/mosaics/current")
    assert current.status_code == 200
    assert current.json()["current_version"] == 1
    assert current.json()["version"]["version"] == 1

    download = client.get("/api/v1/mosaics/1/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "image/tiff"
    assert download.headers["x-mosaic-version"] == "1"
    assert download.headers["cache-control"].endswith("immutable")
    assert "rainmosaic-2026-01-01-v1.tif" in download.headers["content-disposition"]

    values, meta = read_band(download.content)
    assert values.tolist() == [[1, 2, 2], [3, 4, 4]]
    assert meta["dtype"] == "int16"
    assert meta["nodata"] == NODATA
    assert meta["crs"] == "EPSG:4326"
    assert meta["width"] == 3 and meta["height"] == 2

    same = client.get("/api/v1/download/current")
    assert same.status_code == 200
    assert same.content == download.content


def test_new_version_becomes_current_and_old_version_stays_pinned(client):
    assert _submit(client, expected_version=0).status_code == 201
    old = client.get("/api/v1/mosaics/1/download").content

    new_a = np.full((2, 2), 7, dtype=np.int16)
    new_b = np.full((2, 2), 8, dtype=np.int16)
    response = _submit(client, expected_version=1, a=new_a, b=new_b)
    assert response.status_code == 201, response.text
    assert response.json()["version"] == 2

    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 2
    # Version 1 is immutable and still returns the original pixels.
    assert client.get("/api/v1/mosaics/1/download").content == old
    values, _ = read_band(client.get("/api/v1/mosaics/1/download").content)
    assert values.tolist() == [[1, 2, 2], [3, 4, 4]]
    values2, _ = read_band(client.get("/api/v1/mosaics/2/download").content)
    assert values2.tolist() == [[7, 8, 8], [7, 8, 8]]


def test_stale_expected_version_is_rejected(client):
    assert _submit(client, expected_version=0).status_code == 201
    stale = _submit(client, expected_version=0)
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert detail["code"] == "version_conflict"
    assert detail["expected_version"] == 0
    assert detail["current_version"] == 1
    assert detail["current"]["version"] == 1
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 1


def test_future_expected_version_is_rejected(client):
    assert _submit(client, expected_version=0).status_code == 201
    response = _submit(client, expected_version=5)
    assert response.status_code == 409
    assert response.json()["detail"]["current_version"] == 1


def test_concurrent_submissions_publish_exactly_one_version(client):
    barrier = threading.Barrier(2)

    def attempt():
        data, files = submit_payload(
            block_a=make_geotiff(A_TILE),
            block_b=make_geotiff(B_TILE, x0=0.01),
            expected_version=0,
        )
        barrier.wait()
        return client.post("/api/v1/mosaics", data=data, files=files)

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = [future.result() for future in [pool.submit(attempt) for _ in range(2)]]

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [201, 409], [r.text for r in responses]
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 1
    listing = client.get("/api/v1/mosaics").json()
    assert listing["total"] == 1


def test_failed_generation_keeps_old_version(client, monkeypatch):
    assert _submit(client, expected_version=0).status_code == 201
    old = client.get("/api/v1/mosaics/1/download").content

    real_write = rasters.write_mosaic

    def broken_write(left, right, destination):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(rasters, "write_mosaic", broken_write)
    failed = _submit(client, expected_version=1)
    assert failed.status_code == 500
    assert failed.json()["detail"]["code"] == "publish_failed"

    monkeypatch.setattr(rasters, "write_mosaic", real_write)
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 1
    assert client.get("/api/v1/mosaics/1/download").content == old
    assert client.get("/api/v1/mosaics").json()["total"] == 1
    assert client.get("/api/v1/mosaics/2").status_code == 404
    # The failed attempt consumed no version number.
    assert _submit(client, expected_version=1).status_code == 201


def test_readers_keep_downloading_old_version_during_publish(client, monkeypatch):
    assert _submit(client, expected_version=0).status_code == 201
    old = client.get("/api/v1/mosaics/1/download").content

    started = threading.Event()
    release = threading.Event()
    real_write = rasters.write_mosaic

    def slow_write(left, right, destination):
        started.set()
        assert release.wait(timeout=30), "publish never released"
        return real_write(left, right, destination)

    monkeypatch.setattr(rasters, "write_mosaic", slow_write)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_submit, client, expected_version=1)
        assert started.wait(timeout=10), "publish did not start"
        try:
            during_current = client.get("/api/v1/mosaics/current")
            during_download = client.get("/api/v1/mosaics/1/download")
            during_current_download = client.get("/api/v1/download/current")
            assert during_current.status_code == 200
            assert during_current.json()["current_version"] == 1
            assert during_download.content == old
            assert during_current_download.content == old
        finally:
            release.set()
        result = pending.result(timeout=60)

    assert result.status_code == 201
    monkeypatch.setattr(rasters, "write_mosaic", real_write)
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 2
    assert client.get("/api/v1/mosaics/1/download").content == old


def test_versions_survive_restart(restartable_client):
    first = restartable_client()
    assert _submit(first, expected_version=0).status_code == 201
    old = first.get("/api/v1/mosaics/1/download").content
    second_new = _submit(first, expected_version=1)
    assert second_new.status_code == 201
    first.__exit__(None, None, None)

    restarted = restartable_client()
    current = restarted.get("/api/v1/mosaics/current")
    assert current.status_code == 200
    assert current.json()["current_version"] == 2
    assert restarted.get("/api/v1/mosaics/1/download").content == old
    new_values, _ = read_band(restarted.get("/api/v1/mosaics/2/download").content)
    assert new_values.shape == (2, 3)
    listing = restarted.get("/api/v1/mosaics").json()
    assert [item["version"] for item in listing["items"]] == [2, 1]


def test_list_filters_by_day_and_paginates(client):
    assert _submit(client, expected_version=0).status_code == 201
    day_two = _submit(
        client,
        expected_version=1,
        a=np.full((2, 2), 5, dtype=np.int16),
        b=np.full((2, 2), 6, dtype=np.int16),
        day="2026-01-02",
        a_time="2026-01-02T01:00:00Z",
        b_time="2026-01-02T02:00:00Z",
    )
    assert day_two.status_code == 201, day_two.text

    all_items = client.get("/api/v1/mosaics").json()
    assert all_items["total"] == 2
    assert all_items["items"][0]["version"] == 2

    filtered = client.get("/api/v1/mosaics", params={"day": "2026-01-01"}).json()
    assert filtered["total"] == 1
    assert filtered["items"][0]["day"] == "2026-01-01"

    page = client.get("/api/v1/mosaics", params={"limit": 1, "offset": 1}).json()
    assert page["total"] == 2
    assert [item["version"] for item in page["items"]] == [1]

    assert client.get("/api/v1/mosaics", params={"day": "2026-01-03"}).json()["total"] == 0


@pytest.mark.parametrize(
    "mutate, expected_code",
    [
        (lambda d: d.__setitem__("day", "01-01-2026"), "invalid_day"),
        (lambda d: d.__setitem__("block_a_captured_at", "2026-01-01T01:00:00"), "invalid_captured_at"),
        (lambda d: d.__setitem__("block_a_captured_at", "not-a-dateZ"), "invalid_captured_at"),
        (lambda d: d.__setitem__("block_b_no", "1"), "duplicate_block_no"),
        (lambda d: d.__setitem__("expected_version", "-1"), "invalid_expected_version"),
        (lambda d: d.__setitem__("block_a_no", "-3"), "invalid_block_no"),
    ],
)
def test_submission_parameter_validation(client, mutate, expected_code):
    data, files = submit_payload(
        block_a=make_geotiff(A_TILE),
        block_b=make_geotiff(B_TILE, x0=0.01),
        expected_version=0,
    )
    mutate(data)
    response = client.post("/api/v1/mosaics", data=data, files=files)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == expected_code


@pytest.mark.parametrize(
    "a_bytes, b_bytes, field",
    [
        (make_geotiff(A_TILE, crs="EPSG:3857"), make_geotiff(B_TILE, x0=0.01), "crs"),
        (make_geotiff(A_TILE, res=0.02), make_geotiff(B_TILE, x0=0.01), "resolution"),
        (make_geotiff(A_TILE), make_geotiff(B_TILE, x0=0.005), "grid_alignment"),
        (make_geotiff(A_TILE, dtype="uint16", nodata=0), make_geotiff(B_TILE, x0=0.01), None),
        (make_geotiff(A_TILE, nodata=-9999), make_geotiff(B_TILE, x0=0.01), None),
        (make_geotiff(np.zeros((65, 2), dtype=np.int16)), make_geotiff(B_TILE, x0=0.01), "dimensions"),
        (b"not a tiff", make_geotiff(B_TILE, x0=0.01), None),
    ],
)
def test_raster_validation_rejects_bad_blocks(client, a_bytes, b_bytes, field):
    data, files = submit_payload(block_a=a_bytes, block_b=b_bytes, expected_version=0)
    files["block_a"] = ("block_a.tif", a_bytes, "image/tiff")
    response = client.post("/api/v1/mosaics", data=data, files=files)
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    if field is not None:
        assert detail.get("field") == field
        if detail.get("slot") is not None:
            assert detail["slot"] == "a"
    assert client.get("/api/v1/mosaics/current").status_code == 404
    assert client.get("/api/v1/mosaics").json()["total"] == 0


def test_missing_upload_is_rejected(client):
    response = client.post(
        "/api/v1/mosaics",
        data={
            "day": "2026-01-01",
            "expected_version": "0",
            "block_a_no": "1",
            "block_a_captured_at": "2026-01-01T01:00:00Z",
            "block_b_no": "2",
            "block_b_captured_at": "2026-01-01T02:00:00Z",
        },
    )
    assert response.status_code == 422


def test_oversized_upload_is_rejected(client, settings):
    from app.config import Settings
    from app.main import create_app
    from fastapi.testclient import TestClient

    small = Settings(data_dir=settings.data_dir, max_upload_bytes=100)
    app = create_app(small)
    with TestClient(app) as small_client:
        data, files = submit_payload(
            block_a=make_geotiff(A_TILE),
            block_b=make_geotiff(B_TILE, x0=0.01),
            expected_version=0,
        )
        response = small_client.post("/api/v1/mosaics", data=data, files=files)
    assert response.status_code == 413
    # Either the whole-body guard or the per-file check may fire first.
    assert response.json()["detail"]["code"] in ("file_too_large", "request_too_large")


def test_unknown_versions_return_404(client):
    assert client.get("/api/v1/mosaics/current").status_code == 404
    assert client.get("/api/v1/mosaics/99").status_code == 404
    assert client.get("/api/v1/mosaics/99/download").status_code == 404
    assert client.get("/api/v1/mosaics/0/download").status_code == 422


def test_missing_image_on_disk_returns_410(client, settings):
    assert _submit(client, expected_version=0).status_code == 201
    (settings.versions_dir / "v1.tif").unlink()
    response = client.get("/api/v1/mosaics/1/download")
    assert response.status_code == 410
    assert response.json()["detail"]["code"] == "image_missing"


def test_corrupt_pointer_is_recovered_on_startup(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    for _ in range(2):
        with TestClient(create_app(settings)) as client:
            pass
    conn = store.connect(settings)
    try:
        conn.execute("UPDATE state SET value = '42' WHERE key = 'current_version'")
    finally:
        conn.close()
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").json()["current_version"] == 0
        assert client.get("/api/v1/mosaics/current").status_code == 404


def test_optional_day_is_derived_from_capture_time(client):
    response = _submit(client, expected_version=0, include_day=False)
    assert response.status_code == 201
    assert response.json()["day"] == "2026-01-01"


def test_capture_instants_may_straddle_utc_midnight(client):
    # A daily total read at 23:55Z and again at 00:05Z the next day still
    # belongs to the labelled day.
    response = _submit(
        client,
        expected_version=0,
        day="2026-01-01",
        a_time="2026-01-01T23:55:00Z",
        b_time="2026-01-02T00:05:00Z",
    )
    assert response.status_code == 201, response.text
    assert response.json()["day"] == "2026-01-01"
    assert response.json()["blocks"][0]["captured_at"].startswith("2026-01-01T23:55")


def test_explicit_day_is_authoritative(client):
    response = _submit(
        client,
        expected_version=0,
        day="2026-02-14",
        a_time="2026-02-14T00:10:00Z",
        b_time="2026-02-14T00:20:00Z",
    )
    assert response.status_code == 201
    assert response.json()["day"] == "2026-02-14"


def test_equivalent_crs_is_accepted(client):
    data, files = submit_payload(
        block_a=make_geotiff(A_TILE),
        block_b=make_geotiff(B_TILE, x0=0.01, crs="+proj=longlat +datum=WGS84 +no_defs"),
        expected_version=0,
    )
    response = client.post("/api/v1/mosaics", data=data, files=files)
    assert response.status_code == 201, response.text
