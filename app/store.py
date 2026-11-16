"""SQLite metadata store and atomic version publishing.

Version publication follows this sequence, all while holding a SQLite
``BEGIN IMMEDIATE`` write transaction:

1. compare the caller's expected current version with the stored one;
2. allocate the next version number and render a complete GeoTIFF in a
   temporary file;
3. re-open and verify the rendered file, fsync it and atomically rename it to
   its immutable final path;
4. insert the version row and move the ``current_version`` pointer in the same
   transaction.

Consequently the current pointer only moves after a complete image exists on
disk.  A crash before commit leaves the previous version current and at worst
an unreferenced file, which startup reconciliation removes.  Versions are
never overwritten or deleted, so every published version stays downloadable.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import rasters
from .config import Settings
from .rasters import SourceBlock, iso_utc

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    version        INTEGER PRIMARY KEY,
    day            TEXT    NOT NULL,
    status         TEXT    NOT NULL CHECK (status IN ('ready')),
    published_at   TEXT    NOT NULL,
    width          INTEGER NOT NULL,
    height         INTEGER NOT NULL,
    crs_wkt        TEXT    NOT NULL,
    crs_label      TEXT    NOT NULL,
    transform      TEXT    NOT NULL,
    bounds         TEXT    NOT NULL,
    res_x          REAL    NOT NULL,
    res_y          REAL    NOT NULL,
    dtype          TEXT    NOT NULL,
    nodata         INTEGER NOT NULL,
    sha256         TEXT    NOT NULL,
    size_bytes     INTEGER NOT NULL,
    block_meta     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS versions_day_idx ON versions(day);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

CURRENT_STATE_KEY = "current_version"
ORPHAN_GRACE_SECONDS = 3600.0


class VersionConflict(Exception):
    """The caller's expected version is not the published current version."""

    def __init__(self, expected_version: int, current_version: int, current: dict | None):
        super().__init__(
            f"expected version {expected_version} but current version is {current_version}"
        )
        self.expected_version = expected_version
        self.current_version = current_version
        self.current = current


class PublishError(Exception):
    """The new version could not be rendered or registered."""


class PublishBusy(Exception):
    """Another publication held the write lock for longer than the timeout."""


@dataclass(frozen=True)
class DownloadRecord:
    version: int
    day: str
    path: Path
    filename: str
    sha256: str
    size_bytes: int


def connect(settings: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(settings.db_path),
        timeout=settings.sqlite_timeout_seconds,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(settings.sqlite_timeout_seconds * 1000)}")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_store(settings: Settings) -> None:
    """Create directories/schema, recover the pointer and clean leftovers."""

    settings.ensure_directories()
    conn = connect(settings)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO state (key, value) VALUES (?, ?)",
            (CURRENT_STATE_KEY, "0"),
        )
        stored = _read_pointer(conn)
        max_version = _max_version(conn)
        if stored not in (0,) and _fetch_row(conn, stored) is None:
            log.warning("current version %s is missing; recovering to %s", stored, max_version)
            _write_pointer(conn, max_version)
        elif stored > max_version:
            log.warning("current version %s exceeds stored history; recovering to %s", stored, max_version)
            _write_pointer(conn, max_version)
    finally:
        conn.close()

    _reconcile_files(settings)


def _reconcile_files(settings: Settings) -> None:
    now = time.time()
    for path in settings.tmp_dir.glob("*"):
        try:
            if path.is_file() and now - path.stat().st_mtime > ORPHAN_GRACE_SECONDS:
                path.unlink()
        except OSError:
            log.warning("could not remove stale temporary file %s", path, exc_info=True)

    conn = connect(settings)
    try:
        registered = {int(row["version"]) for row in conn.execute("SELECT version FROM versions")}
    finally:
        conn.close()

    for path in settings.versions_dir.glob("v*.tif"):
        try:
            version = int(path.stem[1:])
        except ValueError:
            continue
        if version not in registered and now - path.stat().st_mtime > ORPHAN_GRACE_SECONDS:
            try:
                path.unlink()
                log.warning("removed orphaned mosaic file %s", path)
            except OSError:
                log.warning("could not remove orphaned mosaic %s", path, exc_info=True)


def _read_pointer(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM state WHERE key = ?", (CURRENT_STATE_KEY,)
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def _write_pointer(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT INTO state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (CURRENT_STATE_KEY, str(version)),
    )


def _max_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM versions").fetchone()
    return int(row["v"])


def _fetch_row(conn: sqlite3.Connection, version: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM versions WHERE version = ?", (version,)).fetchone()


def _row_to_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "version": int(row["version"]),
        "day": row["day"],
        "status": row["status"],
        "published_at": row["published_at"],
        "image": {
            "width": int(row["width"]),
            "height": int(row["height"]),
            "dtype": row["dtype"],
            "nodata": int(row["nodata"]),
            "crs": row["crs_label"],
            "crs_wkt": row["crs_wkt"],
            "transform": json.loads(row["transform"]),
            "bounds": json.loads(row["bounds"]),
            "res_x": float(row["res_x"]),
            "res_y": float(row["res_y"]),
            "sha256": row["sha256"],
            "size_bytes": int(row["size_bytes"]),
        },
        "blocks": json.loads(row["block_meta"]),
        "download_url": f"/api/v1/mosaics/{int(row['version'])}/download",
    }


def _current_payload(conn: sqlite3.Connection) -> tuple[int, dict[str, Any] | None]:
    version = _read_pointer(conn)
    if version <= 0:
        return 0, None
    row = _fetch_row(conn, version)
    if row is None:
        return 0, None
    return version, _row_to_payload(row)


def get_current_version(settings: Settings) -> int:
    conn = connect(settings)
    try:
        return _read_pointer(conn)
    finally:
        conn.close()


def get_current(settings: Settings) -> tuple[int, dict[str, Any] | None]:
    conn = connect(settings)
    try:
        return _current_payload(conn)
    finally:
        conn.close()


def get_version(settings: Settings, version: int) -> dict[str, Any] | None:
    conn = connect(settings)
    try:
        row = _fetch_row(conn, version)
        return _row_to_payload(row) if row is not None else None
    finally:
        conn.close()


def list_versions(
    settings: Settings, *, day: str | None = None, limit: int = 20, offset: int = 0
) -> tuple[list[dict[str, Any]], int]:
    conn = connect(settings)
    try:
        if day is None:
            total = conn.execute("SELECT COUNT(*) AS n FROM versions").fetchone()["n"]
            rows = conn.execute(
                "SELECT * FROM versions ORDER BY version DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM versions WHERE day = ?", (day,)
            ).fetchone()["n"]
            rows = conn.execute(
                "SELECT * FROM versions WHERE day = ? ORDER BY version DESC LIMIT ? OFFSET ?",
                (day, limit, offset),
            ).fetchall()
        return [_row_to_payload(row) for row in rows], int(total)
    finally:
        conn.close()


def get_download(settings: Settings, version: int) -> DownloadRecord | None:
    conn = connect(settings)
    try:
        row = _fetch_row(conn, version)
        if row is None:
            return None
        return DownloadRecord(
            version=int(row["version"]),
            day=row["day"],
            path=settings.versions_dir / f"v{int(row['version'])}.tif",
            filename=f"rainmosaic-{row['day']}-v{int(row['version'])}.tif",
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
        )
    finally:
        conn.close()


def get_current_download(settings: Settings) -> DownloadRecord | None:
    conn = connect(settings)
    try:
        version, _ = _current_payload(conn)
        if version <= 0:
            return None
        row = _fetch_row(conn, version)
        if row is None:
            return None
        return DownloadRecord(
            version=int(row["version"]),
            day=row["day"],
            path=settings.versions_dir / f"v{int(row['version'])}.tif",
            filename=f"rainmosaic-{row['day']}-v{int(row['version'])}.tif",
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
        )
    finally:
        conn.close()


def publish(
    settings: Settings,
    *,
    day: str,
    expected_version: int,
    left: SourceBlock,
    right: SourceBlock,
) -> dict[str, Any]:
    """Render and publish a new version atomically."""

    temp_path = settings.tmp_dir / f"publish-{uuid.uuid4().hex}.tif"
    final_path: Path | None = None
    new_version: int | None = None
    conn = connect(settings)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise PublishBusy() from exc
            raise
        current_version, current_payload = _current_payload(conn)
        if current_version != expected_version:
            raise VersionConflict(expected_version, current_version, current_payload)

        new_version = max(current_version, _max_version(conn)) + 1

        # Heavy work happens while the write lock is held.  Readers in WAL
        # mode keep seeing and downloading the old current version.
        info = rasters.write_mosaic(left, right, temp_path)
        rasters.verify_mosaic(temp_path, info)
        _fsync_file(temp_path)
        sha256 = rasters.sha256_file(temp_path)
        size_bytes = temp_path.stat().st_size

        final_path = settings.versions_dir / f"v{new_version}.tif"
        os.replace(temp_path, final_path)
        _fsync_dir(settings.versions_dir)

        published_at = iso_utc(datetime.now(timezone.utc))
        block_meta = [left.metadata(), right.metadata()]
        conn.execute(
            """
            INSERT INTO versions (
                version, day, status, published_at, width, height, crs_wkt,
                crs_label, transform, bounds, res_x, res_y, dtype, nodata,
                sha256, size_bytes, block_meta
            ) VALUES (?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_version,
                day,
                published_at,
                info.width,
                info.height,
                info.crs.to_wkt(),
                info.crs.to_string() or info.crs.to_wkt(),
                json.dumps(list(info.transform)[:6]),
                json.dumps(list(info.bounds)),
                info.res_x,
                info.res_y,
                rasters.DTYPE,
                rasters.NODATA,
                sha256,
                size_bytes,
                json.dumps(block_meta),
            ),
        )
        _write_pointer(conn, new_version)
        row = _fetch_row(conn, new_version)
        assert row is not None
        conn.execute("COMMIT")
        log.info(
            "published version %s for %s (%sx%s, %s bytes)",
            new_version,
            day,
            info.width,
            info.height,
            size_bytes,
        )
        payload = _row_to_payload(row)
        return payload
    except VersionConflict:
        _rollback(conn)
        raise
    except PublishBusy:
        _rollback(conn)
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        _rollback(conn)
        temp_path.unlink(missing_ok=True)
        if final_path is not None and new_version is not None:
            # Only remove the renamed file if the version really is not in the
            # database (guards against an ambiguous commit outcome).
            try:
                row = _fetch_row(conn, new_version)
            except sqlite3.Error:
                row = None
            if row is None:
                final_path.unlink(missing_ok=True)
        log.exception("publishing mosaic failed")
        raise PublishError("mosaic generation or registration failed") from exc
    finally:
        conn.close()


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            log.warning("rollback failed", exc_info=True)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
