"""Shared test fixtures: GeoTIFF block builders and a live API client."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.config import Settings
from app.main import create_app

NODATA = -32768


def make_geotiff(
    values,
    *,
    x0: float = 0.0,
    y0: float = 0.02,
    res: float = 0.01,
    crs: str = "EPSG:4326",
    dtype: str = "int16",
    nodata: int | None = NODATA,
    count: int = 1,
    band: int = 1,
) -> bytes:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 2:
        height, width = array.shape
        if count == 1:
            to_write, band_index = array, 1
        else:  # replicate a 2-D pattern across the requested number of bands
            to_write, band_index = np.repeat(array[np.newaxis, ...], count, axis=0), None
    else:
        height, width = array.shape[1], array.shape[2]
        to_write, band_index = array, None
    buffer = io.BytesIO()
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=from_origin(x0, y0, res, res),
        nodata=nodata,
    ) as dst:
        if band_index is not None:
            dst.write(to_write, band)
        else:
            dst.write(to_write)
    return buffer.getvalue()


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def submit_payload(
    *,
    block_a: bytes,
    block_b: bytes,
    expected_version: int,
    day: str = "2026-01-01",
    a_no: int = 1,
    a_time: str = "2026-01-01T01:00:00Z",
    b_no: int = 2,
    b_time: str = "2026-01-01T02:00:00Z",
    include_day: bool = True,
) -> tuple[dict, dict]:
    data = {
        "block_a_no": str(a_no),
        "block_a_captured_at": a_time,
        "block_b_no": str(b_no),
        "block_b_captured_at": b_time,
        "expected_version": str(expected_version),
    }
    if include_day:
        data["day"] = day
    files = {
        "block_a": ("block_a.tif", block_a, "image/tiff"),
        "block_b": ("block_b.tif", block_b, "image/tiff"),
    }
    return data, files


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture()
def client(settings: Settings):
    from fastapi.testclient import TestClient

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def restartable_client(settings: Settings):
    """Factory that keeps the same data dir but restarts the application."""

    from fastapi.testclient import TestClient

    clients = []

    def factory():
        app = create_app(settings)
        test_client = TestClient(app)
        test_client.__enter__()
        clients.append(test_client)
        return test_client

    yield factory
    for test_client in clients:
        test_client.__exit__(None, None, None)


def read_band(content: bytes) -> tuple[np.ndarray, dict]:
    with rasterio.MemoryFile(content) as mem, mem.open() as ds:
        return ds.read(1), {
            "width": ds.width,
            "height": ds.height,
            "transform": tuple(ds.transform)[:6],
            "crs": ds.crs.to_string(),
            "nodata": ds.nodata,
            "dtype": ds.dtypes[0],
        }
