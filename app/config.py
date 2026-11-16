"""Runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_upload_bytes: int = 2 * 1024 * 1024
    sqlite_timeout_seconds: float = 30.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rainmosaic.sqlite3"

    @property
    def versions_dir(self) -> Path:
        return self.data_dir / "versions"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("RAINMOSAIC_DATA_DIR", "./data")).resolve()
    raw_limit = os.environ.get("RAINMOSAIC_MAX_UPLOAD_BYTES")
    max_upload_bytes = int(raw_limit) if raw_limit else 2 * 1024 * 1024
    if max_upload_bytes < 1:
        raise ValueError("RAINMOSAIC_MAX_UPLOAD_BYTES must be positive")
    return Settings(data_dir=data_dir, max_upload_bytes=max_upload_bytes)
