"""Raster validation and mosaic construction.

The service only accepts two single-band, int16 GeoTIFF blocks that sit on
exactly the same CRS/resolution pixel grid.  Both blocks are painted into the
union window; higher-priority pixels (later UTC acquisition time, then larger
block number) overwrite lower-priority pixels, but only where the
higher-priority block has a valid value.  This yields the required
"later valid pixel, otherwise fall back to the other block" rule.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine

try:  # pyproj gives semantic CRS equality (independent of WKT spelling).
    import pyproj
except ImportError:  # pragma: no cover - normal deployments install pyproj
    pyproj = None


NODATA = -32768
DTYPE = "int16"
MAX_BLOCK_DIMENSION = 64
# Whole-pixel offsets are compared in pixel units.  A thousandth of a pixel is
# far below any real misalignment but safely above double-precision noise from
# large projected coordinates.
ALIGN_TOLERANCE_PIXELS = 1e-3
# Relative tolerance when comparing the two declared pixel sizes.
RESOLUTION_REL_TOLERANCE = 1e-7


class RasterInputError(Exception):
    """A client error in one of the uploaded raster blocks."""

    def __init__(self, message: str, *, slot: str | None = None, field: str | None = None):
        super().__init__(message)
        self.message = message
        self.slot = slot
        self.field = field

    def as_detail(self) -> dict:
        detail: dict[str, object] = {"code": "invalid_raster", "message": self.message}
        if self.slot is not None:
            detail["slot"] = self.slot
        if self.field is not None:
            detail["field"] = self.field
        return detail


@dataclass(frozen=True)
class SourceBlock:
    slot: str
    block_no: int
    captured_at: datetime
    original_filename: str
    sha256: str
    data: np.ndarray
    transform: Affine
    crs: CRS
    width: int
    height: int

    @property
    def res_x(self) -> float:
        return float(self.transform.a)

    @property
    def res_y(self) -> float:
        return float(-self.transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.transform.c),
            float(self.transform.f - self.height * self.res_y),
            float(self.transform.c + self.width * self.res_x),
            float(self.transform.f),
        )

    def metadata(self) -> dict:
        return {
            "slot": self.slot,
            "block_no": self.block_no,
            "captured_at": iso_utc(self.captured_at),
            "original_filename": self.original_filename,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "bounds": list(self.bounds),
        }


@dataclass(frozen=True)
class MosaicLayout:
    width: int
    height: int
    transform: Affine
    crs: CRS
    bounds: tuple[float, float, float, float]
    placements: tuple[tuple[SourceBlock, int, int], ...]

    @property
    def res_x(self) -> float:
        return float(self.transform.a)

    @property
    def res_y(self) -> float:
        return float(-self.transform.e)


@dataclass(frozen=True)
class MosaicInfo:
    width: int
    height: int
    transform: Affine
    crs: CRS
    bounds: tuple[float, float, float, float]
    res_x: float
    res_y: float


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RasterInputError(
            "captured_at must include a UTC offset (for example 2026-01-01T12:00:00Z)",
            field="captured_at",
        )
    return value.astimezone(timezone.utc)


def load_block(
    raw: bytes,
    *,
    slot: str,
    block_no: int,
    captured_at: datetime,
    original_filename: str,
) -> SourceBlock:
    """Validate and materialise one uploaded GeoTIFF block."""

    captured = _utc(captured_at)
    safe_name = os.path.basename((original_filename or "").replace("\\", "/"))[:255]
    sha256 = hashlib.sha256(raw).hexdigest()

    try:
        with MemoryFile(raw) as mem, mem.open() as ds:
            if ds.driver != "GTiff":
                raise RasterInputError(
                    f"block must be a GeoTIFF (driver GTiff), got {ds.driver!r}", slot=slot
                )
            if ds.count != 1:
                raise RasterInputError(
                    f"block must have exactly one band, got {ds.count}", slot=slot
                )
            if ds.dtypes[0] != DTYPE:
                raise RasterInputError(
                    f"block pixel type must be int16, got {ds.dtypes[0]}", slot=slot
                )
            # -32768 is the contract's NoData value and must be declared so a
            # real -32768 reading can never be mistaken for a gap (and vice
            # versa).
            if ds.nodata is None or not math.isclose(
                float(ds.nodata), float(NODATA), rel_tol=0.0, abs_tol=1e-9
            ):
                raise RasterInputError(
                    f"block NoData must be declared as {NODATA}", slot=slot
                )
            if ds.crs is None:
                raise RasterInputError("block must declare a CRS", slot=slot)
            if ds.width < 1 or ds.height < 1:
                raise RasterInputError("block dimensions must be positive", slot=slot)
            if ds.width > MAX_BLOCK_DIMENSION or ds.height > MAX_BLOCK_DIMENSION:
                raise RasterInputError(
                    f"block must be at most {MAX_BLOCK_DIMENSION}x{MAX_BLOCK_DIMENSION} pixels; "
                    f"got {ds.width}x{ds.height}",
                    slot=slot,
                    field="dimensions",
                )

            transform = ds.transform
            coefficients = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
            if not all(math.isfinite(float(v)) for v in coefficients):
                raise RasterInputError("block affine transform is not finite", slot=slot)
            if not math.isclose(transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
                transform.d, 0.0, abs_tol=1e-12
            ):
                raise RasterInputError(
                    "rotated or sheared pixel grids are not supported", slot=slot
                )
            if transform.a <= 0 or transform.e >= 0:
                raise RasterInputError(
                    "block must use a north-up, positive-x pixel grid", slot=slot
                )

            data = ds.read(1)
            crs = ds.crs
            width = ds.width
            height = ds.height
    except RasterInputError:
        raise
    except Exception as exc:
        raise RasterInputError(f"file is not a readable GeoTIFF: {exc}", slot=slot) from exc

    if data.shape != (height, width):
        raise RasterInputError("unexpected raster window shape", slot=slot)

    return SourceBlock(
        slot=slot,
        block_no=block_no,
        captured_at=captured,
        original_filename=safe_name,
        sha256=sha256,
        data=data,
        transform=transform,
        crs=crs,
        width=width,
        height=height,
    )


def crs_equal(left: CRS, right: CRS) -> bool:
    if left == right:
        return True
    if pyproj is not None:
        try:
            return pyproj.CRS.from_user_input(left.to_wkt()).equals(
                pyproj.CRS.from_user_input(right.to_wkt()), ignore_axis_order=True
            )
        except Exception:
            return False
    try:  # pragma: no cover - pyproj is a normal dependency
        return left.to_wkt(version="WKT1_GDAL") == right.to_wkt(version="WKT1_GDAL")
    except Exception:
        return False


def validate_pair(left: SourceBlock, right: SourceBlock) -> None:
    if left.block_no == right.block_no:
        raise RasterInputError(
            f"the two blocks must have different block numbers; both are {left.block_no}",
            field="block_no",
        )
    if not crs_equal(left.crs, right.crs):
        raise RasterInputError("the two blocks must use the same CRS", field="crs")

    for axis, a, b in (
        ("x", left.res_x, right.res_x),
        ("y", left.res_y, right.res_y),
    ):
        if not math.isclose(a, b, rel_tol=RESOLUTION_REL_TOLERANCE, abs_tol=1e-12):
            raise RasterInputError(
                f"the two blocks must have the same {axis} resolution; got {a} and {b}",
                field="resolution",
            )

    # Both origins must differ by a whole number of pixels on each axis.
    for axis, oa, ob, res in (
        ("x", left.transform.c, right.transform.c, left.res_x),
        ("y", left.transform.f, right.transform.f, left.res_y),
    ):
        offset = (oa - ob) / res
        if abs(offset - round(offset)) > ALIGN_TOLERANCE_PIXELS:
            raise RasterInputError(
                f"the two pixel grids are not aligned on the {axis} axis",
                field="grid_alignment",
            )

    _layout(left, right)  # also validates the union geometry


def _layout(left: SourceBlock, right: SourceBlock) -> MosaicLayout:
    res_x = left.res_x
    res_y = left.res_y
    min_x = min(left.bounds[0], right.bounds[0])
    min_y = min(left.bounds[1], right.bounds[1])
    max_x = max(left.bounds[2], right.bounds[2])
    max_y = max(left.bounds[3], right.bounds[3])

    width_f = (max_x - min_x) / res_x
    height_f = (max_y - min_y) / res_y
    if width_f < 1 - ALIGN_TOLERANCE_PIXELS or height_f < 1 - ALIGN_TOLERANCE_PIXELS:
        raise RasterInputError("union of the two blocks is empty", field="bounds")
    if abs(width_f - round(width_f)) > ALIGN_TOLERANCE_PIXELS or abs(
        height_f - round(height_f)
    ) > ALIGN_TOLERANCE_PIXELS:
        raise RasterInputError(
            "the union of the two blocks does not align to whole pixels", field="grid_alignment"
        )

    width = int(round(width_f))
    height = int(round(height_f))
    transform = Affine(res_x, 0.0, min_x, 0.0, -res_y, max_y)

    placements: list[tuple[SourceBlock, int, int]] = []
    for block in (left, right):
        col_f = (block.transform.c - min_x) / res_x
        row_f = (max_y - block.transform.f) / res_y
        col = int(round(col_f))
        row = int(round(row_f))
        if abs(col_f - col) > ALIGN_TOLERANCE_PIXELS or abs(row_f - row) > ALIGN_TOLERANCE_PIXELS:
            raise RasterInputError("block does not sit on the output pixel grid", field="grid_alignment")
        if col < 0 or row < 0 or col + block.width > width or row + block.height > height:
            raise RasterInputError("block lies outside the union window", field="bounds")
        placements.append((block, row, col))

    return MosaicLayout(
        width=width,
        height=height,
        transform=transform,
        crs=left.crs,
        bounds=(min_x, min_y, max_x, max_y),
        placements=tuple(placements),
    )


def build_mosaic(left: SourceBlock, right: SourceBlock) -> tuple[np.ndarray, MosaicLayout]:
    layout = _layout(left, right)
    canvas = np.full((layout.height, layout.width), NODATA, dtype=np.int16)

    # Paint low priority first; a later, higher-priority block only overwrites
    # pixels where it actually has data, so NoData falls back to the other one.
    ordered = sorted(layout.placements, key=lambda item: (item[0].captured_at, item[0].block_no))
    for block, row, col in ordered:
        region = canvas[row : row + block.height, col : col + block.width]
        valid = block.data != NODATA
        region[valid] = block.data[valid]

    return canvas, layout


def write_mosaic(left: SourceBlock, right: SourceBlock, destination: Path) -> MosaicInfo:
    canvas, layout = build_mosaic(left, right)
    profile = {
        "driver": "GTiff",
        "width": layout.width,
        "height": layout.height,
        "count": 1,
        "dtype": DTYPE,
        "crs": layout.crs,
        "transform": layout.transform,
        "nodata": NODATA,
        "compress": "deflate",
        "zlevel": 6,
        "predictor": 2,
        "BIGTIFF": "NO",
    }
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(canvas, 1)
        dst.update_tags(
            AREA_OR_POINT="Area",
            RAINMOSAIC_ALGORITHM="later-valid-pixel",
            RAINMOSAIC_BLOCKS=",".join(
                str(b.block_no)
                for b, _, _ in sorted(
                    layout.placements, key=lambda item: (item[0].captured_at, item[0].block_no)
                )
            ),
        )

    return MosaicInfo(
        width=layout.width,
        height=layout.height,
        transform=layout.transform,
        crs=layout.crs,
        bounds=layout.bounds,
        res_x=layout.res_x,
        res_y=layout.res_y,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_mosaic(path: Path, expected: MosaicInfo) -> None:
    """Re-open a freshly written mosaic and confirm its structure."""

    with rasterio.open(path) as ds:
        if ds.driver != "GTiff" or ds.count != 1 or ds.dtypes[0] != DTYPE:
            raise RuntimeError("written mosaic has unexpected driver, band count or dtype")
        if (ds.width, ds.height) != (expected.width, expected.height):
            raise RuntimeError("written mosaic has unexpected dimensions")
        if ds.nodata is None or float(ds.nodata) != float(NODATA):
            raise RuntimeError("written mosaic has unexpected NoData value")
        if ds.crs != expected.crs:
            raise RuntimeError("written mosaic has unexpected CRS")
        if not ds.transform.almost_equals(expected.transform):
            raise RuntimeError("written mosaic has unexpected transform")
