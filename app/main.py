"""FastAPI application: submit two daily rainfall blocks, browse, download."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from email.utils import formatdate
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, Path as PathParam, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import __version__, store
from .config import Settings, load_settings
from .limits import BodyLimitMiddleware
from .rasters import RasterInputError, load_block, validate_pair

log = logging.getLogger(__name__)


class RetireRequest(BaseModel):
    """Operator note recorded with a retirement."""

    retired_by: str | None = Field(
        default=None, max_length=255, description="Who requested the retirement"
    )
    reason: str | None = Field(
        default=None, max_length=2000, description="Why the version is being reclaimed"
    )



def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def parse_day(raw: str) -> str:
    value = (raw or "").strip()
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_day",
                "message": "day must be a calendar date formatted YYYY-MM-DD",
            },
        ) from exc
    if parsed.isoformat() != value:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_day",
                "message": "day must be a calendar date formatted YYYY-MM-DD",
            },
        )
    return value


def parse_captured_at(raw: str, slot: str) -> datetime:
    text = (raw or "").strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_captured_at",
                "slot": slot,
                "message": "captured_at must be an ISO-8601 timestamp with a UTC offset",
            },
        ) from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_captured_at",
                "slot": slot,
                "message": (
                    "captured_at must include a UTC offset "
                    "(for example 2026-01-01T12:00:00Z)"
                ),
            },
        )
    return value.astimezone(timezone.utc)


async def read_upload(upload: UploadFile, *, slot: str, limit: int) -> bytes:
    data = await upload.read(limit + 1)
    await upload.close()
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "file_too_large",
                "slot": slot,
                "message": f"each uploaded file must be at most {limit} bytes",
            },
        )
    if not data:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_file", "slot": slot, "message": "uploaded file is empty"},
        )
    return data


def error_response(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    detail: dict[str, Any] = {"code": code, "message": message}
    detail.update(extra)
    return JSONResponse(status_code=status, content={"detail": detail})


class PinnedFileResponse(Response):
    """Serve a GeoTIFF from an already-open descriptor.

    The descriptor was taken when the download was admitted, before any
    retirement could unlink the path.  Reading from the descriptor therefore
    yields exactly the bytes that were registered even if the directory entry
    disappears mid-transfer, and it never needs to re-open the path (a normal
    ``FileResponse`` does) after the file has been reclaimed.
    """

    chunk_size = 64 * 1024
    media_type = "image/tiff"

    def __init__(self, record: store.DownloadRecord, status_code: int = 200):
        super().__init__(media_type=self.media_type, status_code=status_code)
        self.record: store.DownloadRecord | None = record
        self.fd = record.fd
        try:
            stat_result = os.fstat(self.fd)
        except OSError:
            store.release_download(record)
            self.record = None
            raise
        self.version = record.version
        self.size = stat_result.st_size
        self.headers["ETag"] = f'"{record.sha256}"'
        self.headers["X-Mosaic-Version"] = str(record.version)
        self.headers["X-Mosaic-SHA256"] = record.sha256
        self.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        self.headers["Accept-Ranges"] = "bytes"
        self.headers["Last-Modified"] = formatdate(stat_result.st_mtime, usegmt=True)
        self.headers["Content-Disposition"] = _content_disposition(record.filename)

    async def read_at(self, offset: int, length: int) -> bytes:
        return await run_in_threadpool(os.pread, self.fd, length, offset)

    async def __call__(self, scope, receive, send) -> None:
        try:
            await self._serve(scope, receive, send)
        finally:
            self.close()

    def close(self) -> None:
        """Release the pinned descriptor exactly once."""
        record, self.record = self.record, None
        if record is not None:
            store.release_download(record)

    def __del__(self) -> None:  # pragma: no cover - safety net for aborted requests
        try:
            self.close()
        except Exception:
            pass

    async def _serve(self, scope, receive, send) -> None:
        method = scope["method"].upper()
        request_headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope["headers"]
        }
        offset, length, status = 0, self.size, 200
        range_header = request_headers.get("range")
        if_range = request_headers.get("if-range")
        if range_header and method in ("GET", "HEAD"):
            if not if_range or if_range == self.headers["etag"] or if_range == self.headers["last-modified"]:
                parsed = _parse_single_range(range_header, self.size)
                if parsed is False:
                    await PlainTextResponse(
                        "requested range not satisfiable",
                        status_code=416,
                        headers={"Content-Range": f"bytes */{self.size}"},
                    )(scope, receive, send)
                    return
                if parsed is not None:
                    offset, length = parsed
                    status = 206
                    self.headers["Content-Range"] = (
                        f"bytes {offset}-{offset + length - 1}/{self.size}"
                    )

        self.headers["Content-Length"] = str(length)
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": self.raw_headers,
            }
        )
        if method == "HEAD" or length == 0:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        remaining = length
        while remaining > 0:
            chunk = await self.read_at(offset, min(self.chunk_size, remaining))
            if not chunk:
                raise RuntimeError(f"image for version {self.version} ended unexpectedly")
            offset += len(chunk)
            remaining -= len(chunk)
            await send(
                {"type": "http.response.body", "body": chunk, "more_body": remaining > 0}
            )


def _content_disposition(filename: str) -> str:
    encoded = quote(filename)
    if encoded != filename:
        return f"attachment; filename*=utf-8''{encoded}"
    return f'attachment; filename="{filename}"'


def _parse_single_range(header: str, size: int) -> tuple[int, int] | None | bool:
    """Return (offset, length), None when the header should be ignored, or
    False when no byte range is satisfiable."""

    if size == 0:
        return None
    try:
        unit, spec = header.split("=", 1)
    except ValueError:
        return None
    if unit.strip().lower() != "bytes" or "," in spec:
        # Multiple ranges: permitted to ignore the header and send the entity.
        return None
    start_text, dash, end_text = spec.strip().partition("-")
    if not dash:
        return None
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                return False
            start = max(size - suffix, 0)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if start >= size:
        return False
    if end < start:
        return None
    end = min(end, size - 1)
    return start, end - start + 1


def _validate_version_number(value: int) -> int:
    if value < 0:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_expected_version",
                "message": "expected_version must be 0 or a positive version number",
            },
        )
    return value


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = settings if settings is not None else load_settings()
        store.init_store(resolved)
        app.state.settings = resolved
        # Two files plus multipart framing, with a little headroom.
        app.state.body_limit_bytes = 2 * resolved.max_upload_bytes + 1024 * 1024
        log.info("rain-mosaic service ready; data dir %s", resolved.data_dir)
        yield

    app = FastAPI(
        title="Rainfall Mosaic Service",
        version=__version__,
        summary=(
            "Combine two daily single-band rainfall GeoTIFF blocks "
            "into one research-area mosaic."
        ),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    static_dir = Path(__file__).parent / "static" / "swagger-ui"
    app.mount("/static/swagger-ui", StaticFiles(directory=static_dir), name="swagger-ui")

    # Rejects oversized bodies before the multipart parser spools them to disk.
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Flatten request-model errors into the same {code, message} shape."""
        problems = exc.errors()
        first = problems[0] if problems else {}
        location = ".".join(
            str(part) for part in first.get("loc", ()) if part not in ("body", "query", "path")
        )
        message = first.get("msg", "invalid request")
        if location:
            message = f"{location}: {message}"
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": message,
                    "errors": [
                        {
                            "field": ".".join(
                                str(part)
                                for part in problem.get("loc", ())
                                if part not in ("body", "query", "path")
                            ),
                            "message": problem.get("msg", ""),
                            "type": problem.get("type", ""),
                        }
                        for problem in problems
                    ],
                }
            },
        )

    @app.get("/docs", include_in_schema=False)
    def swagger_ui() -> HTMLResponse:
        # Assets are served locally so /docs also works without internet access.
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} — API documentation",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
            swagger_ui_parameters={"persistAuthorization": True, "displayRequestDuration": True},
        )

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/healthz", tags=["ops"], summary="Liveness, database and version check")
    def healthz(request: Request) -> dict:
        current = get_settings(request)
        conn = store.connect(current)
        try:
            conn.execute("SELECT 1").fetchone()
        except sqlite3.Error as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        finally:
            conn.close()
        return {"status": "ok", "current_version": store.get_current_version(current)}

    @app.post(
        "/api/v1/mosaics",
        status_code=201,
        tags=["mosaics"],
        summary="Submit two blocks for one day and publish a new mosaic version",
        description=(
            "Upload two single-band int16 GeoTIFF blocks that belong to the same observation "
            "day. Both must use "
            "the same CRS, the same resolution and an aligned pixel grid, with at most 64x64 "
            "pixels each, NoData = -32768.\n\n"
            "The union bounding rectangle of the two blocks is mosaicked. In overlapping "
            "pixels the block with the later UTC acquisition time wins; where that block is "
            "NoData the other block is used; when acquisition times are equal the larger "
            "block number wins. Pixels covered by neither block stay NoData.\n\n"
            "`day` labels the observation and is optional; when omitted it is taken from the "
            "earlier of the two UTC acquisition timestamps. Capture instants may straddle UTC "
            "midnight (daily totals are often read just after it).\n\n"
            "`expected_version` is the version the submitter last saw (0 when nothing is "
            "published). Publishing is compare-and-swap: concurrent submissions that started "
            "from the same expected version produce exactly one new version and the other "
            "receives 409. A version only becomes current after its image is completely "
            "written and registered, so a failed submission never disturbs the current "
            "version."
        ),
        responses={
            201: {"description": "New version published"},
            409: {"description": "Another submission published first"},
            413: {"description": "Uploaded file too large"},
            422: {"description": "Invalid parameters or raster blocks"},
        },
    )
    async def submit_mosaic(
        request: Request,
        block_a: Annotated[
            UploadFile,
            File(
                description="GeoTIFF for block A",
                json_schema_extra={"format": "binary"},
            ),
        ],
        block_b: Annotated[
            UploadFile,
            File(
                description="GeoTIFF for block B",
                json_schema_extra={"format": "binary"},
            ),
        ],
        block_a_no: Annotated[int, Form(description="Block number of block A")],
        block_a_captured_at: Annotated[
            str, Form(description="UTC acquisition time of block A, ISO-8601 with offset")
        ],
        block_b_no: Annotated[int, Form(description="Block number of block B")],
        block_b_captured_at: Annotated[
            str, Form(description="UTC acquisition time of block B, ISO-8601 with offset")
        ],
        expected_version: Annotated[
            int,
            Form(
                description=(
                    "Version the submitter last saw; 0 when no version exists yet"
                )
            ),
        ],
        day: Annotated[
            str | None,
            Form(description="UTC day of the observation, YYYY-MM-DD (derived if omitted)"),
        ] = None,
    ):
        current_settings = get_settings(request)
        _validate_version_number(expected_version)
        a_time = parse_captured_at(block_a_captured_at, "a")
        b_time = parse_captured_at(block_b_captured_at, "b")
        # The observation day labels the pair; when omitted it is taken from
        # the earlier of the two acquisition instants.  Capture instants may
        # straddle UTC midnight (daily totals are often read just after it).
        day_value = (
            parse_day(day)
            if day is not None
            else min(a_time, b_time).date().isoformat()
        )
        if block_a_no < 0 or block_b_no < 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_block_no",
                    "message": "block numbers must not be negative",
                },
            )
        if block_a_no == block_b_no:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "duplicate_block_no",
                    "message": (
                        "the two blocks must have different block numbers; "
                        f"both are {block_a_no}"
                    ),
                },
            )

        raw_a = await read_upload(block_a, slot="a", limit=current_settings.max_upload_bytes)
        raw_b = await read_upload(block_b, slot="b", limit=current_settings.max_upload_bytes)

        def work() -> dict[str, Any]:
            try:
                left = load_block(
                    raw_a,
                    slot="a",
                    block_no=block_a_no,
                    captured_at=a_time,
                    original_filename=block_a.filename or "block_a.tif",
                )
                right = load_block(
                    raw_b,
                    slot="b",
                    block_no=block_b_no,
                    captured_at=b_time,
                    original_filename=block_b.filename or "block_b.tif",
                )
                validate_pair(left, right)
            except RasterInputError as exc:
                raise HTTPException(status_code=422, detail=exc.as_detail()) from exc

            try:
                return store.publish(
                    current_settings,
                    day=day_value,
                    expected_version=expected_version,
                    left=left,
                    right=right,
                )
            except store.VersionConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "version_conflict",
                        "message": (
                            "another submission published first; "
                            f"current version is {exc.current_version}"
                        ),
                        "expected_version": exc.expected_version,
                        "current_version": exc.current_version,
                        "current": exc.current,
                    },
                ) from exc
            except store.PublishBusy as exc:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "publish_busy",
                        "message": "another publication is still in progress; retry shortly",
                    },
                ) from exc
            except store.PublishError as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "publish_failed",
                        "message": (
                            "mosaic could not be generated or registered; "
                            "the current version is unchanged"
                        ),
                    },
                ) from exc

        return await run_in_threadpool(work)

    @app.get(
        "/api/v1/mosaics",
        tags=["mosaics"],
        summary="List versions (newest first), retired ones included by default",
    )
    def list_mosaics(
        request: Request,
        day: Annotated[str | None, Query(description="Filter by UTC day, YYYY-MM-DD")] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
        include_retired: Annotated[
            bool,
            Query(description="Include retired versions (metadata is always retained)"),
        ] = True,
    ) -> dict:
        current_settings = get_settings(request)
        day_value = parse_day(day) if day is not None else None
        items, total = store.list_versions(
            current_settings,
            day=day_value,
            limit=limit,
            offset=offset,
            include_retired=include_retired,
        )
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.get(
        "/api/v1/mosaics/current",
        tags=["mosaics"],
        summary="Show the currently published version",
        responses={404: {"description": "Nothing published yet"}},
    )
    def current_mosaic(request: Request) -> JSONResponse:
        version, payload = store.get_current(get_settings(request))
        if payload is None:
            return error_response(
                404, "no_current_version", "no mosaic version has been published yet"
            )
        return JSONResponse({"current_version": version, "version": payload})

    @app.get(
        "/api/v1/mosaics/{version}",
        tags=["mosaics"],
        summary="Show one version by number",
        description=(
            "Returns the immutable metadata of a version, including its SHA-256. "
            "A retired version keeps its metadata and digest; its payload adds "
            "`retired: true`, the `retirement` record and `reclaim_status`, which is "
            "`pending` while the image is queued for removal and `reclaimed` once "
            "the file is gone."
        ),
        responses={404: {"description": "Unknown version"}},
    )
    def get_mosaic_version(
        request: Request,
        version: Annotated[int, PathParam(ge=1, description="Version number")],
    ) -> dict:
        payload = store.get_version(get_settings(request), version)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "version_not_found",
                    "message": f"version {version} does not exist",
                },
            )
        return {"version": payload}

    @app.post(
        "/api/v1/mosaics/{version}/retire",
        tags=["mosaics"],
        summary="Retire a version superseded by a newer version of the same day",
        description=(
            "Marks the version as retired and reclaims its image to free volume "
            "space. Retirement is allowed only when a higher published version "
            "exists for the same day; otherwise the request returns 409 and nothing "
            "changes. The response is 200 in both the fresh and the repeated case, "
            "with `reused: true` when the identical retirement record was already "
            "in place.\n\n"
            "Metadata, SHA-256 and byte size are kept. New downloads return 410 from "
            "the moment the retirement is registered; downloads that were already "
            "accepted keep reading the original bytes to completion. If the file "
            "cannot be removed after registration, the version stays retired and "
            "`reclaim_status` remains `pending`; startup reconciliation retries it. "
            "Only the file belonging to this version is ever removed."
        ),
        responses={
            200: {"description": "Version retired (or already retired)"},
            404: {"description": "Unknown version"},
            409: {"description": "No newer published version for this day"},
        },
    )
    def retire_mosaic_version(
        request: Request,
        version: Annotated[int, PathParam(ge=1, description="Version number")],
        body: Annotated[
            RetireRequest | None,
            Body(
                description=(
                    "Optional operator note recorded with the retirement; "
                    "send an empty object to omit it"
                )
            ),
        ] = RetireRequest(),
    ) -> dict:
        current_settings = get_settings(request)
        try:
            result = store.retire_version(
                current_settings,
                version,
                retired_by=body.retired_by if body is not None else None,
                reason=body.reason if body is not None else None,
            )
        except store.VersionNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "version_not_found",
                    "message": f"version {version} does not exist",
                },
            ) from exc
        except store.NotSuperseded as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": {
                        "code": "not_superseded",
                        "message": (
                            f"version {version} can only be retired after a newer "
                            f"version of {exc.day} has been published"
                        ),
                        "version": version,
                        "day": exc.day,
                        "latest_version": exc.latest_version,
                    }
                },
            )
        except store.RetireBusy as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "retire_busy",
                    "message": "another write is still in progress; retry shortly",
                },
            ) from exc
        except store.RetireError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "retire_failed",
                    "message": (
                        "retirement could not be registered; the version is "
                        "unchanged and still downloadable"
                    ),
                },
            ) from exc
        return {
            "version": result.version,
            "retired": True,
            "reused": result.reused,
            "reclaim_status": result.payload.get("reclaim_status"),
            "retirement": result.payload.get("retirement"),
            "metadata": result.payload,
        }

    @app.api_route(
        "/api/v1/mosaics/{version}/download",
        methods=["GET", "HEAD"],
        include_in_schema=False,
        response_class=PinnedFileResponse,
        responses={
            404: {"description": "Unknown version"},
            410: {"description": "Version retired or image missing"},
        },
    )
    @app.get(
        "/api/v1/mosaics/{version}/download",
        tags=["mosaics"],
        summary="Download the GeoTIFF for a fixed version",
        description=(
            "Returns exactly the bytes registered for the version, with its SHA-256 "
            "as a strong ETag. Returns 410 once the version has been retired, and "
            "410 `image_missing` when an active version's file is absent. Supports "
            "HEAD and a single byte range."
        ),
        response_class=PinnedFileResponse,
        responses={
            404: {"description": "Unknown version"},
            410: {"description": "Version retired or image missing"},
        },
    )
    def download_mosaic_version(
        request: Request,
        version: Annotated[int, PathParam(ge=1, description="Version number")],
    ) -> PinnedFileResponse:
        try:
            record = store.open_download(get_settings(request), version)
        except store.VersionNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "version_not_found",
                    "message": f"version {version} does not exist",
                },
            ) from exc
        except store.VersionRetired as exc:
            raise HTTPException(
                status_code=410,
                detail={
                    "code": "version_retired",
                    "message": (
                        f"version {version} was retired and its image reclaimed; "
                        "no new downloads are served"
                    ),
                    "version": version,
                    "retirement": exc.retirement,
                },
                headers={"Cache-Control": "no-store"},
            ) from exc
        except store.ImageMissing as exc:
            log.error("image for version %s is missing", version)
            raise HTTPException(
                status_code=410,
                detail={
                    "code": "image_missing",
                    "message": f"image for version {version} is unavailable on disk",
                },
            ) from exc
        return PinnedFileResponse(record)

    @app.api_route(
        "/api/v1/download/current",
        methods=["GET", "HEAD"],
        include_in_schema=False,
        response_class=PinnedFileResponse,
        responses={404: {"description": "Nothing published yet"}},
    )
    @app.get(
        "/api/v1/download/current",
        tags=["mosaics"],
        summary="Download the current mosaic",
        response_class=PinnedFileResponse,
        responses={404: {"description": "Nothing published yet"}},
    )
    def download_current(request: Request) -> PinnedFileResponse:
        try:
            record = store.open_current_download(get_settings(request))
        except store.VersionRetired as exc:
            raise HTTPException(
                status_code=410,
                detail={
                    "code": "version_retired",
                    "message": (
                        "the current pointer references a retired version; "
                        "publish or point to an active version"
                    ),
                    "version": exc.version,
                    "retirement": exc.retirement,
                },
                headers={"Cache-Control": "no-store"},
            ) from exc
        except store.ImageMissing as exc:
            log.error("image for current version is missing")
            raise HTTPException(
                status_code=410,
                detail={
                    "code": "image_missing",
                    "message": "the current version's image is unavailable on disk",
                },
            ) from exc
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "no_current_version",
                    "message": "no mosaic version has been published yet",
                },
            )
        return PinnedFileResponse(record)

    return app


app = create_app()
