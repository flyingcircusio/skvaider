import asyncio
import logging
import sys
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from skvaider.inference.routers import models
from skvaider.inference.state import manager

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Inference host starting...")
    yield
    log.info("Shutting down...")

    # Stop all running models
    for name, model in list(manager.running_models.items()):
        log.info("Stopping model on shutdown", model=name)
        try:
            model.process.terminate()
            await model.process.wait()
        except Exception as e:
            log.error("Error stopping model", model=name, error=str(e))


app = FastAPI(lifespan=lifespan)

app.include_router(models.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    # Basic logging setup
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
