"""Mosaic rules: union extent, overlap priority, NoData fallback, gaps."""

from __future__ import annotations

import numpy as np

from app.rasters import NODATA, RasterInputError, build_mosaic, load_block, validate_pair
from tests.conftest import make_geotiff, utc


def _loads(raw: bytes, *, slot: str, block_no: int, captured_at: str):
    return load_block(
        raw,
        slot=slot,
        block_no=block_no,
        captured_at=utc(captured_at),
        original_filename=f"{slot}.tif",
    )


def _mosaic(a_values, b_values, *, a_time, b_time, a_no=1, b_no=2, a_offset=(0.0, 0.02), b_offset=(0.01, 0.02)):
    left = _loads(make_geotiff(a_values, x0=a_offset[0], y0=a_offset[1]), slot="a", block_no=a_no, captured_at=a_time)
    right = _loads(make_geotiff(b_values, x0=b_offset[0], y0=b_offset[1]), slot="b", block_no=b_no, captured_at=b_time)
    validate_pair(left, right)
    return build_mosaic(left, right)


def test_overlap_prefers_later_valid_pixel_and_falls_back_on_nodata():
    # A covers columns 0-1, B covers columns 1-2; only column 1 overlaps.
    a = np.array([[10, 20], [30, 40]], dtype=np.int16)
    b = np.array([[NODATA, 2], [NODATA, 4]], dtype=np.int16)
    canvas, layout = _mosaic(
        a, b, a_time="2026-01-01T01:00:00Z", b_time="2026-01-01T05:00:00Z"
    )
    # Later block B wins in overlap, but its NoData pixels fall back to A.
    assert canvas.tolist() == [[10, 20, 2], [30, 40, 4]]
    assert (layout.width, layout.height) == (3, 2)
    assert layout.bounds == (0.0, 0.0, 0.03, 0.02)


def test_overlap_uses_later_block_values_not_slot_order():
    # Both blocks cover the same one-pixel area; acquisition time decides.
    a = np.array([[1]], dtype=np.int16)
    b = np.array([[9]], dtype=np.int16)
    canvas, _ = _mosaic(
        a,
        b,
        a_time="2026-01-01T23:00:00Z",
        b_time="2026-01-01T01:00:00Z",
        b_offset=(0.0, 0.02),
    )
    assert canvas.tolist() == [[1]]

    canvas, _ = _mosaic(
        a,
        b,
        a_time="2026-01-01T01:00:00Z",
        b_time="2026-01-01T23:00:00Z",
        b_offset=(0.0, 0.02),
    )
    assert canvas.tolist() == [[9]]


def test_later_block_nodata_over_tiny_overlap_falls_back():
    # A is a 2x1 row covering columns 0-1; B is a 1x1 cell at column 1.
    a = np.array([[1, NODATA]], dtype=np.int16)
    b = np.array([[9]], dtype=np.int16)
    canvas, _ = _mosaic(
        a,
        b,
        a_time="2026-01-01T01:00:00Z",
        b_time="2026-01-01T23:00:00Z",
    )
    assert canvas.tolist() == [[1, 9]]


def test_equal_timestamps_break_tie_by_block_number():
    a = np.array([[5]], dtype=np.int16)
    b = np.array([[7]], dtype=np.int16)
    canvas, _ = _mosaic(
        a,
        b,
        a_time="2026-01-01T10:00:00Z",
        b_time="2026-01-01T10:00:00Z",
        a_no=3,
        b_no=8,
        b_offset=(0.0, 0.02),
    )
    assert canvas.tolist() == [[7]]

    canvas, _ = _mosaic(
        a,
        b,
        a_time="2026-01-01T10:00:00Z",
        b_time="2026-01-01T10:00:00Z",
        a_no=9,
        b_no=2,
        b_offset=(0.0, 0.02),
    )
    assert canvas.tolist() == [[5]]


def test_uncovered_pixels_stay_nodata():
    a = np.array([[1]], dtype=np.int16)
    b = np.array([[2]], dtype=np.int16)
    canvas, layout = _mosaic(
        a,
        b,
        a_time="2026-01-01T01:00:00Z",
        b_time="2026-01-01T02:00:00Z",
        b_offset=(0.02, 0.02),
    )
    assert canvas.tolist() == [[1, NODATA, 2]]
    assert layout.width == 3


def test_disjoint_blocks_produce_union_with_gap():
    a = np.full((2, 2), 1, dtype=np.int16)
    b = np.full((2, 2), 2, dtype=np.int16)
    canvas, layout = _mosaic(
        a,
        b,
        a_time="2026-01-01T01:00:00Z",
        b_time="2026-01-01T02:00:00Z",
        b_offset=(0.04, 0.02),
    )
    assert layout.width == 6
    assert canvas[0].tolist() == [1, 1, NODATA, NODATA, 2, 2]


def test_rejects_different_resolution():
    left = _loads(make_geotiff([[1]], res=0.01), slot="a", block_no=1, captured_at="2026-01-01T01:00:00Z")
    right = _loads(make_geotiff([[2]], res=0.02, x0=0.01), slot="b", block_no=2, captured_at="2026-01-01T02:00:00Z")
    try:
        validate_pair(left, right)
    except RasterInputError as exc:
        assert exc.field == "resolution"
    else:  # pragma: no cover
        raise AssertionError("expected a resolution mismatch")


def test_rejects_unaligned_grid():
    left = _loads(make_geotiff([[1]], res=0.01), slot="a", block_no=1, captured_at="2026-01-01T01:00:00Z")
    right = _loads(
        make_geotiff([[2]], x0=0.005, y0=0.02),
        slot="b",
        block_no=2,
        captured_at="2026-01-01T02:00:00Z",
    )
    try:
        validate_pair(left, right)
    except RasterInputError as exc:
        assert exc.field == "grid_alignment"
    else:  # pragma: no cover
        raise AssertionError("expected a grid alignment error")


def test_rejects_different_crs():
    left = _loads(make_geotiff([[1]], crs="EPSG:4326"), slot="a", block_no=1, captured_at="2026-01-01T01:00:00Z")
    right = _loads(make_geotiff([[2]], crs="EPSG:3857"), slot="b", block_no=2, captured_at="2026-01-01T02:00:00Z")
    try:
        validate_pair(left, right)
    except RasterInputError as exc:
        assert exc.field == "crs"
    else:  # pragma: no cover
        raise AssertionError("expected a CRS mismatch")


def test_accepts_equivalent_crs_spelling():
    left = _loads(make_geotiff([[1]], crs="EPSG:4326"), slot="a", block_no=1, captured_at="2026-01-01T01:00:00Z")
    right = _loads(
        make_geotiff([[2]], x0=0.01, crs="+proj=longlat +datum=WGS84 +no_defs"),
        slot="b",
        block_no=2,
        captured_at="2026-01-01T02:00:00Z",
    )
    validate_pair(left, right)


def test_rejects_same_block_number():
    left = _loads(make_geotiff([[1]]), slot="a", block_no=4, captured_at="2026-01-01T01:00:00Z")
    right = _loads(make_geotiff([[2]], x0=0.01), slot="b", block_no=4, captured_at="2026-01-01T02:00:00Z")
    try:
        validate_pair(left, right)
    except RasterInputError as exc:
        assert exc.field == "block_no"
    else:  # pragma: no cover
        raise AssertionError("expected a duplicate block number error")


def test_rejects_bad_dtype_nodata_and_bands():
    for kwargs, field_message in (
        ({"dtype": "uint16", "nodata": 0}, "int16"),
        ({"nodata": -9999}, "NoData"),
        ({"count": 2}, "one band"),
    ):
        raw = make_geotiff([[1]], **kwargs)
        try:
            _loads(raw, slot="a", block_no=1, captured_at="2026-01-01T01:00:00Z")
        except RasterInputError as exc:
            assert field_message.lower() in str(exc).lower()
        else:  # pragma: no cover
            raise AssertionError(f"expected rejection for {kwargs}")


def test_rejects_oversized_block():
    raw = make_geotiff(np.zeros((65, 2), dtype=np.int16))
    try:
        _loads(raw, slot="a", block_no=1, captured_at="2026-01-01T01:00:00Z")
    except RasterInputError as exc:
        assert exc.field == "dimensions"
    else:  # pragma: no cover
        raise AssertionError("expected a size rejection")


def test_rejects_rotated_grid():
    # Build a block with a rotated transform.
    import io

    import rasterio
    from rasterio.transform import Affine

    out = io.BytesIO()
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        width=1,
        height=1,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=Affine(0.01, 0.001, 0.0, 0.0, -0.01, 0.02),
        nodata=NODATA,
    ) as ds:
        ds.write(np.array([[1]], dtype=np.int16), 1)
    try:
        _loads(out.getvalue(), slot="a", block_no=1, captured_at="2026-01-01T01:00:00Z")
    except RasterInputError as exc:
        assert "rotated" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a rotation rejection")


def test_sixty_four_pixel_blocks_are_allowed():
    a = np.zeros((64, 64), dtype=np.int16)
    b = np.ones((64, 64), dtype=np.int16)
    canvas, layout = _mosaic(
        a, b, a_time="2026-01-01T01:00:00Z", b_time="2026-01-01T02:00:00Z"
    )
    assert (layout.width, layout.height) == (65, 64)
    assert canvas[0, 0] == 0 and canvas[0, 1] == 1
