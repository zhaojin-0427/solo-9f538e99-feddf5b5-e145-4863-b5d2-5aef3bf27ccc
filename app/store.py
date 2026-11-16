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
never overwritten, so every published version keeps its metadata and digest.

Retirement is a separate, additive record.  A version may be retired only once
a newer version for the same day has been published.  Registering the
retirement is the point of no return: after that commit new downloads are
refused.  The directory entry can then be unlinked immediately, because an
already-admitted download holds an open descriptor and still reads every
original byte.  A failed registration changes nothing; a failed unlink leaves
the version retired with ``reclaim_status = 'pending'`` and startup
reconciliation retries the removal.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, replace
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

CREATE TABLE IF NOT EXISTS retirements (
    version         INTEGER PRIMARY KEY REFERENCES versions(version),
    day             TEXT    NOT NULL,
    retired_at      TEXT    NOT NULL,
    retired_by      TEXT,
    reason          TEXT,
    reclaim_status  TEXT    NOT NULL
                    CHECK (reclaim_status IN ('pending', 'reclaimed')),
    reclaimed_at    TEXT,
    reclaim_error   TEXT
);
CREATE INDEX IF NOT EXISTS retirements_day_idx ON retirements(day);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

CURRENT_STATE_KEY = "current_version"
ORPHAN_GRACE_SECONDS = 3600.0

# Serialises download admission against file removal.  Admission performs the
# retirement lookup and the ``open`` while holding it; retirement performs the
# unlink while holding it.  Either admission observes the retirement and is
# refused, or it holds the descriptor before the unlink and can still read every
# original byte.  The lock covers only the lookup/open (or the unlink), never a
# response body, so slow downloads cannot delay publishing or another version's
# retirement.
_reclaim_lock = threading.Lock()


class VersionNotFound(Exception):
    """The requested version number does not exist."""

    def __init__(self, version: int):
        super().__init__(f"version {version} does not exist")
        self.version = version


class VersionRetired(Exception):
    """The version has been retired; new downloads are refused."""

    def __init__(self, version: int, retirement: dict[str, Any]):
        super().__init__(f"version {version} has been retired")
        self.version = version
        self.retirement = retirement


class ImageMissing(Exception):
    """The version is registered but its image file is not available."""

    def __init__(self, version: int):
        super().__init__(f"image for version {version} is unavailable on disk")
        self.version = version


class NotSuperseded(Exception):
    """The version has no newer published version on the same day."""

    def __init__(self, version: int, day: str, latest_version: int):
        super().__init__(
            f"version {version} has no newer published version on {day}"
        )
        self.version = version
        self.day = day
        self.latest_version = latest_version


class RetireError(Exception):
    """The retirement could not be registered; nothing was changed."""


class RetireBusy(Exception):
    """Another write transaction held the lock for longer than the timeout."""


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
    fd: int = -1


@dataclass(frozen=True)
class RetireResult:
    version: int
    payload: dict[str, Any]
    reused: bool


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
    reclaim_pending(settings)


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


def _record_reclaim_error(settings: Settings, version: int, error: str) -> None:
    try:
        conn = connect(settings)
        try:
            conn.execute(
                "UPDATE retirements SET reclaim_error = ? WHERE version = ?",
                (error[:500], version),
            )
        finally:
            conn.close()
    except sqlite3.Error:
        # Bookkeeping only; the retirement itself is already durable and the
        # file stays pending for the next startup attempt.
        log.warning("could not record reclaim error for version %s", version, exc_info=True)


def _mark_reclaimed(settings: Settings, version: int) -> None:
    """Settle a pending reclaim.  The first completion wins, so a repeat or a
    concurrent request never rewrites the recorded ``reclaimed_at``."""

    try:
        conn = connect(settings)
        try:
            conn.execute(
                "UPDATE retirements SET reclaim_status = 'reclaimed', reclaimed_at = ?, "
                "reclaim_error = NULL WHERE version = ? AND reclaim_status = 'pending'",
                (iso_utc(datetime.now(timezone.utc)), version),
            )
        finally:
            conn.close()
    except sqlite3.Error:
        # The file is already gone; leave the row pending so startup retries
        # (which settles it via the FileNotFoundError path).
        log.warning("could not mark version %s reclaimed", version, exc_info=True)


def version_file(settings: Settings, version: int) -> Path:
    """The one file owned by the given version registration."""
    return settings.versions_dir / f"v{version}.tif"


def _unlink_file(path: Path) -> None:
    """Remove one version file.  Kept separate so failure paths are testable."""
    path.unlink()


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


def _retirement_from_row(row: sqlite3.Row) -> dict[str, Any] | None:
    """Retirement details when the joined row carries one, else None."""

    try:
        retired_at = row["retired_at"]
    except IndexError:  # a plain versions row without the join
        return None
    if retired_at is None:
        return None
    return {
        "retired_at": retired_at,
        "retired_by": row["retired_by"],
        "reason": row["reason"],
        "reclaim_status": row["reclaim_status"],
        "reclaimed_at": row["reclaimed_at"],
        "reclaim_error": row["reclaim_error"],
    }


def _row_to_payload(row: sqlite3.Row) -> dict[str, Any]:
    retirement = _retirement_from_row(row)
    payload: dict[str, Any] = {
        "version": int(row["version"]),
        "day": row["day"],
        "status": row["status"],
        "retired": retirement is not None,
        "reclaim_status": retirement["reclaim_status"] if retirement else None,
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
        "downloadable": retirement is None,
    }
    if retirement is not None:
        payload["retirement"] = retirement
    return payload


VERSION_SELECT = (
    "SELECT v.*, r.retired_at, r.retired_by, r.reason, r.reclaim_status, "
    "r.reclaimed_at, r.reclaim_error "
    "FROM versions v LEFT JOIN retirements r ON r.version = v.version "
)


def _fetch_joined_row(conn: sqlite3.Connection, version: int) -> sqlite3.Row | None:
    return conn.execute(VERSION_SELECT + "WHERE v.version = ?", (version,)).fetchone()


def _current_payload(conn: sqlite3.Connection) -> tuple[int, dict[str, Any] | None]:
    version = _read_pointer(conn)
    if version <= 0:
        return 0, None
    row = _fetch_joined_row(conn, version)
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
        row = _fetch_joined_row(conn, version)
        return _row_to_payload(row) if row is not None else None
    finally:
        conn.close()


def list_versions(
    settings: Settings,
    *,
    day: str | None = None,
    limit: int = 20,
    offset: int = 0,
    include_retired: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    conn = connect(settings)
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if day is not None:
            clauses.append("v.day = ?")
            params.append(day)
        if not include_retired:
            clauses.append("r.version IS NULL")
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""

        total = conn.execute(
            "SELECT COUNT(*) AS n FROM versions v "
            "LEFT JOIN retirements r ON r.version = v.version " + where,
            params,
        ).fetchone()["n"]
        rows = conn.execute(
            VERSION_SELECT
            + where
            + "ORDER BY v.version DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return [_row_to_payload(row) for row in rows], int(total)
    finally:
        conn.close()


def _download_record(settings: Settings, row: sqlite3.Row) -> DownloadRecord:
    version = int(row["version"])
    return DownloadRecord(
        version=version,
        day=row["day"],
        path=version_file(settings, version),
        filename=f"rainmosaic-{row['day']}-v{version}.tif",
        sha256=row["sha256"],
        size_bytes=int(row["size_bytes"]),
    )


def open_download(settings: Settings, version: int) -> DownloadRecord:
    """Admit one download and pin the image bytes for its lifetime.

    The meta check, the retirement check and the ``open`` happen under the
    reclaim lock, as does the unlink performed by a retirement.  Admission
    therefore either observes the retirement and is refused, or holds the
    descriptor before the unlink and can still read every original byte.
    The lock covers only that short sequence, never a response body, so a slow
    download cannot delay a publication or another version's retirement.  The
    returned record owns the descriptor; the caller *must* call
    :func:`release_download` afterwards.
    """

    with _reclaim_lock:
        conn = connect(settings)
        try:
            row = _fetch_joined_row(conn, version)
            if row is None:
                raise VersionNotFound(version)
            retirement = _retirement_from_row(row)
            if retirement is not None:
                raise VersionRetired(version, retirement)
            path = version_file(settings, version)
            try:
                fd = os.open(path, os.O_RDONLY)
            except FileNotFoundError as exc:
                # The path vanished while the version is still active.  Treat a
                # concurrent retirement as retired, otherwise it is missing.
                conn2 = connect(settings)
                try:
                    fresh = _fetch_joined_row(conn2, version)
                finally:
                    conn2.close()
                retirement = _retirement_from_row(fresh) if fresh is not None else None
                if retirement is not None:
                    raise VersionRetired(version, retirement) from exc
                raise ImageMissing(version) from exc
            except OSError as exc:
                log.error("cannot open image for version %s: %s", version, exc)
                raise ImageMissing(version) from exc
        finally:
            conn.close()
        return replace(_download_record(settings, row), fd=fd)


def open_current_download(settings: Settings) -> DownloadRecord | None:
    """Admit a download of whatever version is current right now.

    Resolution and admission share one lock acquisition, so the pointer cannot
    move between choosing the version and pinning its bytes.
    """

    with _reclaim_lock:
        conn = connect(settings)
        try:
            version, _ = _current_payload(conn)
            if version <= 0:
                return None
            row = _fetch_joined_row(conn, version)
            if row is None:
                return None
            retirement = _retirement_from_row(row)
            if retirement is not None:
                # Defensive: a well-formed store never points at a retired
                # version, but never serve reclaimed bytes as "current".
                raise VersionRetired(version, retirement)
            path = version_file(settings, version)
            try:
                fd = os.open(path, os.O_RDONLY)
            except FileNotFoundError as exc:
                raise ImageMissing(version) from exc
            except OSError as exc:
                log.error("cannot open image for current version %s: %s", version, exc)
                raise ImageMissing(version) from exc
        finally:
            conn.close()
        return replace(_download_record(settings, row), fd=fd)


def release_download(record: DownloadRecord) -> None:
    """Close the descriptor pinned by :func:`open_download`."""
    if record.fd < 0:
        return
    try:
        os.close(record.fd)
    except OSError:  # pragma: no cover - already closed/aborted
        log.warning("could not close download descriptor for v%s", record.version, exc_info=True)


def _insert_retirement(
    conn: sqlite3.Connection,
    *,
    version: int,
    day: str,
    retired_at: str,
    retired_by: str | None,
    reason: str | None,
) -> None:
    """Insert the retirement row; isolated so failures are observable/testable."""
    conn.execute(
        "INSERT INTO retirements (version, day, retired_at, retired_by, reason, "
        "reclaim_status) VALUES (?, ?, ?, ?, ?, 'pending')",
        (version, day, retired_at, retired_by, reason),
    )


def retire_version(
    settings: Settings,
    version: int,
    *,
    retired_by: str | None = None,
    reason: str | None = None,
) -> RetireResult:
    """Register the retirement of ``version`` and reclaim its file if safe.

    The ``BEGIN IMMEDIATE`` transaction makes "only retire when a newer
    same-day version exists" a single atomic decision; a rejected request
    writes nothing.  The registration commit is the point of no return: from
    then on new downloads answer 410.  The file is then unlinked under the
    reclaim lock, which cannot race an admitted download because admission
    opens its descriptor under the same lock.  Unlinking never touches another
    version's file, and an unlink failure leaves a pending reclaim for startup
    reconciliation instead of failing the request.
    """

    conn = connect(settings)
    retired_payload: dict[str, Any] | None = None
    path: Path | None = None
    reused = False
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise RetireBusy() from exc
            raise
        row = _fetch_joined_row(conn, version)
        if row is None:
            raise VersionNotFound(version)
        day = row["day"]
        existing = _retirement_from_row(row)
        if existing is not None:
            # Idempotent: repeat requests return the same retirement record.
            reused = True
            retired_payload = _row_to_payload(row)
            _commit(conn)
        else:
            latest = conn.execute(
                "SELECT MAX(version) AS v FROM versions WHERE day = ?", (day,)
            ).fetchone()["v"]
            latest_version = int(latest) if latest is not None else version
            if latest_version <= version:
                raise NotSuperseded(version, day, latest_version)

            retired_at = iso_utc(datetime.now(timezone.utc))
            _insert_retirement(
                conn,
                version=version,
                day=day,
                retired_at=retired_at,
                retired_by=retired_by,
                reason=reason,
            )
            _commit(conn)
            updated = _fetch_joined_row(conn, version)
            assert updated is not None
            retired_payload = _row_to_payload(updated)
        path = version_file(settings, version)
    except (VersionNotFound, NotSuperseded):
        _rollback(conn)
        raise
    except RetireBusy:
        _rollback(conn)
        raise
    except Exception as exc:
        _rollback(conn)
        # A failed COMMIT may still have landed.  Never report "unchanged"
        # while the retirement is in fact registered: re-read and, when the row
        # exists, treat the request as succeeded.
        try:
            fallback = get_version(settings, version)
        except sqlite3.Error:
            fallback = None
        if fallback is not None and fallback.get("retired"):
            log.warning(
                "retirement of version %s raised %s after it may have committed; "
                "reporting the registered record",
                version,
                exc,
                exc_info=True,
            )
            return RetireResult(
                version=version,
                payload=_reclaim_file(
                    settings, version, version_file(settings, version), fallback
                ),
                reused=True,
            )
        log.exception("retiring version %s failed", version)
        raise RetireError("retirement could not be registered") from exc
    finally:
        conn.close()

    assert retired_payload is not None
    if reused:
        # A previous attempt may have crashed between commit and unlink.  Retry
        # the reclaim for pending records only; a completed one stays as is.
        if retired_payload["retirement"]["reclaim_status"] == "pending" and path is not None:
            retired_payload = _reclaim_file(settings, version, path, retired_payload)
        return RetireResult(version=version, payload=retired_payload, reused=True)

    assert path is not None
    retired_payload = _reclaim_file(settings, version, path, retired_payload)
    return RetireResult(version=version, payload=retired_payload, reused=False)


def _reclaim_file(
    settings: Settings,
    version: int,
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Remove the retired version's own file and update the reclaim state."""

    failure: str | None = None
    with _reclaim_lock:
        try:
            _unlink_file(path)
        except FileNotFoundError:
            # Already gone (a previous attempt or external cleanup): settled.
            pass
        except OSError as exc:
            failure = str(exc)
            log.warning(
                "version %s is retired but %s could not be removed; "
                "it stays pending for startup reconciliation",
                version,
                path,
                exc_info=True,
            )
    if failure is not None:
        # The registration is already committed, so the version must stay
        # non-downloadable; record the pending reclaim for a later retry.
        _record_reclaim_error(settings, version, failure)
    else:
        _mark_reclaimed(settings, version)
    return _reload_payload(settings, version, payload)


def _reload_payload(settings: Settings, version: int, fallback: dict[str, Any]) -> dict[str, Any]:
    """Re-read the version payload; fall back to the in-hand copy on error."""
    try:
        refreshed = get_version(settings, version)
    except sqlite3.Error:
        log.warning("could not reload retired version %s", version, exc_info=True)
        return fallback
    return refreshed if refreshed is not None else fallback


def reclaim_pending(settings: Settings) -> int:
    """Retry removal of retired files that are still pending (startup path)."""

    conn = connect(settings)
    try:
        versions = [
            int(row["version"])
            for row in conn.execute(
                "SELECT version FROM retirements WHERE reclaim_status = 'pending' "
                "ORDER BY version"
            )
        ]
    finally:
        conn.close()
    reclaimed = 0
    for version in versions:
        failure: str | None = None
        path = version_file(settings, version)
        with _reclaim_lock:
            try:
                _unlink_file(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                failure = str(exc)
                log.warning("could not reclaim retired version %s", version, exc_info=True)
        if failure is not None:
            _record_reclaim_error(settings, version, failure)
            continue
        _mark_reclaimed(settings, version)
        reclaimed += 1
    return reclaimed


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


def _commit(conn: sqlite3.Connection) -> None:
    conn.execute("COMMIT")


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
