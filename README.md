# Rainfall Mosaic Service

Combines two same-day, adjacent single-band rainfall GeoTIFF blocks into one
research-area mosaic that models can read, with versioned publication over a
FastAPI API, Rasterio for raster work and SQLite for metadata.

## Run with Docker Compose

```bash
docker compose up --build -d
# Swagger UI:   http://localhost:8000/docs
# Health:       http://localhost:8000/healthz
docker compose logs -f api
```

Published images and the SQLite database live in the named volume
`rainmosaic-data`, so versions and the current pointer survive
`docker compose restart`, `down` and container upgrades (`down -v` deletes
them). Change the host port with `RAINMOSAIC_PUBLISHED_PORT=9000 docker compose up -d`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/mosaics` | Submit two blocks; publishes a new version |
| GET | `/api/v1/mosaics` | List versions, newest first (`day`, `limit`, `offset`, `include_retired`) |
| GET | `/api/v1/mosaics/current` | Current version metadata |
| GET | `/api/v1/mosaics/{version}` | Metadata for a fixed version (retired ones included) |
| GET | `/api/v1/mosaics/{version}/download` | GeoTIFF for a fixed version |
| POST | `/api/v1/mosaics/{version}/retire` | Retire a superseded version and reclaim its image |
| GET | `/api/v1/download/current` | GeoTIFF for the current version |
| GET | `/healthz` | Liveness, database and current-version check |

`/docs` supports submitting (multipart form), browsing, downloading and
retiring.

### Submitting

`multipart/form-data` fields:

| Field | Meaning |
| --- | --- |
| `block_a`, `block_b` | The two GeoTIFF files |
| `block_a_no`, `block_b_no` | Distinct block numbers |
| `block_a_captured_at`, `block_b_captured_at` | UTC acquisition instants, ISO-8601 with offset |
| `expected_version` | Version the submitter last saw; `0` if none is published |
| `day` | Optional `YYYY-MM-DD`; taken from the earlier capture instant if omitted |

Raster contract, enforced per block:

* GeoTIFF, exactly one band, `int16`, NoData declared as `-32768`;
* at most 64×64 pixels;
* north-up grid (no rotation/shear);
* both blocks: same CRS (compared semantically, so an equivalent WKT/PROJ
  spelling is fine), same pixel size, origins offset by whole pixels.

```bash
curl -X POST http://localhost:8000/api/v1/mosaics \
  -F day=2026-01-01 \
  -F expected_version=0 \
  -F block_a_no=101 -F block_a_captured_at=2026-01-01T01:00:00Z \
  -F block_b_no=102 -F block_b_captured_at=2026-01-01T02:00:00Z \
  -F block_a=@block_a.tif -F block_b=@block_b.tif
```

Response `201` carries `version`, `day`, `image` (size, CRS, transform, bounds,
resolution, dtype, NoData, SHA-256, byte size), both block descriptors and a
`download_url`.

### Mosaic rule

The output covers the union bounding rectangle of the two blocks. Each
overlapping pixel takes the value of the block acquired later (UTC); where that
block is NoData the other block's value is used; equal instants are broken by
the larger block number. Pixels covered by neither block stay `-32768`. Output
is a single-band `int16` GeoTIFF with the block CRS/resolution and NoData
declared as `-32768`.

### Versioning and concurrency

`expected_version` makes publication a compare-and-swap. Submissions that
started from the same expected version result in exactly one new version; the
others get `409 version_conflict` with the winning version embedded. Raster
validation errors (`422`), missing/oversized uploads (`413`) and failed
generation (`500 publish_failed`) never create a version and never disturb the
current one.

A new version is rendered to a temporary file, verified by re-opening it,
fsynced and atomically renamed, and only then inserted into SQLite and pointed
to as current, all inside one `BEGIN IMMEDIATE` transaction. The current pointer
therefore never references a partial image; a crash simply leaves the previous
version current. Readers use WAL snapshots and keep downloading the old version
while a publication is in flight.

Every version keeps its own immutable file, so `GET /api/v1/mosaics/{version}/download`
always returns exactly the bytes registered under that version (with a stable
SHA-256 `ETag`), including after restarts.

### Retiring superseded versions

`POST /api/v1/mosaics/{version}/retire` frees the volume space held by an image
that a newer version of the same day has replaced. The optional body is
`{"retired_by": "...", "reason": "..."}`; both fields are just recorded.

```bash
curl -X POST http://localhost:8000/api/v1/mosaics/1/retire \
  -H 'content-type: application/json' \
  -d '{"retired_by": "ops", "reason": "replaced by v2"}'
```

* **Only a superseded version may be retired.** If no higher version for the
  same `day` is published, the request returns `409 not_superseded`, the
  winner's `latest_version` is reported, and nothing is written.
* **Metadata is kept.** `GET /api/v1/mosaics/{version}` still reports the full
  descriptor, including the original SHA-256 and byte size, and adds
  `retired: true`, `downloadable: false`, `reclaim_status` and a `retirement`
  record (`retired_at`, `retired_by`, `reason`, `reclaimed_at`,
  `reclaim_error`).
* **`reclaim_status`** is `pending` while the file is still queued for removal
  and `reclaimed` once it is gone. `GET /api/v1/mosaics` includes retired
  versions by default (`include_retired=false` hides them).
* **New downloads return `410 version_retired`** as soon as the retirement is
  registered. A download *admitted before* the retirement keeps reading the
  original bytes: the server streams from a descriptor opened at admission
  time, so unlinking the path cannot truncate it, and a slow download never
  holds a lock that publishing or another retirement needs.
* **Repeating the request is idempotent** and returns the same retirement
  record with `reused: true`.
* **Failure handling.** If the retirement cannot be registered, the request
  returns `500 retire_failed` and the version stays fully downloadable. If
  registration succeeds but the unlink fails, the version stays retired
  (`reclaim_status: pending`, `reclaim_error` set) and is never downloadable
  again; startup reconciliation removes the file and settles the record. Only
  `versions/v{version}.tif` for that exact version is ever removed, so other
  versions are untouched.

### Errors

Errors use `{"detail": {...}}` with `code` and `message`; raster errors also
carry `slot` (`a`/`b`) and, where applicable, `field` (`crs`, `resolution`,
`grid_alignment`, `dimensions`, `block_no`).

## Design notes

* SQLite runs in WAL mode with `synchronous=FULL`; state lives in
  `$RAINMOSAIC_DATA_DIR/rainmosaic.sqlite3`, images in `versions/`, scratch files
  in `tmp/`. Retirement records are additive rows; existing databases are
  upgraded in place on startup (the original `versions` rows are not modified).
* Startup validates the schema, repairs a dangling current pointer, removes
  old orphan files left by interrupted publications and retries any pending
  retired-file reclaim.
* Generation and file I/O run in a worker thread so other requests, including
  downloads, keep flowing.
* Download admission holds a short-lived lock while it checks the version and
  opens the image descriptor; file removal takes the same lock. Streaming the
  body holds no lock. On POSIX the open descriptor guarantees an admitted
  download reads every original byte even after its path is removed, while a
  retirement that got the lock first makes new downloads see `410`. A slow
  download therefore never delays publishing or another retirement.
* Only one API worker is used by default: SQLite plus a local disk volume make
  horizontal scaling a deployment concern rather than a code feature.

## Local development and tests

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
RAINMOSAIC_DATA_DIR=./data python main.py          # http://127.0.0.1:8000/docs
python -m pytest -q
```

The test suite covers the mosaic rules (overlap priority, NoData fallback, tie
break, gaps, alignment/CRS/size validation), publishing, version pinning,
conflicts, concurrent submissions, failed generation, downloads during
publication, restart persistence, listing/pagination, error handling, and
retirement (supersede gating, metadata/SHA retention, 410, in-flight and slow
downloads, idempotency, registration failure, unlink failure and restart
recovery).

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAINMOSAIC_DATA_DIR` | `/data` in the image, `./data` locally | Writable persistent directory |
| `RAINMOSAIC_MAX_UPLOAD_BYTES` | `2097152` | Per-file upload cap |
| `RAINMOSAIC_PORT` | `8000` | Listen port |
| `RAINMOSAIC_LOG_LEVEL` | `INFO` | Log level |
