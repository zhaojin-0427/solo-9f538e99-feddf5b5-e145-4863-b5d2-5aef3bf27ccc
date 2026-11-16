"""Entrypoint for the container image."""

import logging
import os

import uvicorn

logging.basicConfig(
    level=os.environ.get("RAINMOSAIC_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("RAINMOSAIC_HOST", "0.0.0.0"),
        port=int(os.environ.get("RAINMOSAIC_PORT", "8000")),
        proxy_headers=False,
        access_log=True,
        workers=1,
    )
