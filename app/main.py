"""FastAPI application: submit two daily rainfall blocks, browse, download."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Path as PathParam, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import __version__, store
from .config import Settings, load_settings
from .limits import BodyLimitMiddleware
from .rasters import RasterInputError, load_block, validate_pair

log = logging.getLogger(__name__)


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
        summary="List published versions (newest first)",
    )
    def list_mosaics(
        request: Request,
        day: Annotated[str | None, Query(description="Filter by UTC day, YYYY-MM-DD")] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict:
        current_settings = get_settings(request)
        day_value = parse_day(day) if day is not None else None
        items, total = store.list_versions(
            current_settings, day=day_value, limit=limit, offset=offset
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

    @app.api_route(
        "/api/v1/mosaics/{version}/download",
        methods=["GET", "HEAD"],
        include_in_schema=False,
        response_class=FileResponse,
        responses={
            404: {"description": "Unknown version"},
            410: {"description": "Image file missing"},
        },
    )
    @app.get(
        "/api/v1/mosaics/{version}/download",
        tags=["mosaics"],
        summary="Download the GeoTIFF for a fixed version",
        response_class=FileResponse,
        responses={
            404: {"description": "Unknown version"},
            410: {"description": "Image file missing"},
        },
    )
    def download_mosaic_version(
        request: Request,
        version: Annotated[int, PathParam(ge=1, description="Version number")],
    ) -> FileResponse:
        record = store.get_download(get_settings(request), version)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "version_not_found",
                    "message": f"version {version} does not exist",
                },
            )
        return _file_response(record)

    @app.api_route(
        "/api/v1/download/current",
        methods=["GET", "HEAD"],
        include_in_schema=False,
        response_class=FileResponse,
        responses={404: {"description": "Nothing published yet"}},
    )
    @app.get(
        "/api/v1/download/current",
        tags=["mosaics"],
        summary="Download the current mosaic",
        response_class=FileResponse,
        responses={404: {"description": "Nothing published yet"}},
    )
    def download_current(request: Request) -> FileResponse:
        record = store.get_current_download(get_settings(request))
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "no_current_version",
                    "message": "no mosaic version has been published yet",
                },
            )
        return _file_response(record)

    return app


def _file_response(record: store.DownloadRecord) -> FileResponse:
    if not record.path.is_file():
        log.error("image for version %s is missing at %s", record.version, record.path)
        raise HTTPException(
            status_code=410,
            detail={
                "code": "image_missing",
                "message": f"image for version {record.version} is unavailable on disk",
            },
        )
    return FileResponse(
        path=record.path,
        media_type="image/tiff",
        filename=record.filename,
        headers={
            "ETag": f'"{record.sha256}"',
            "X-Mosaic-Version": str(record.version),
            "X-Mosaic-SHA256": record.sha256,
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


app = create_app()
