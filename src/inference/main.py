import asyncio
import logging
import sys
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from inference.routers import models
from inference.state import manager

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Inference host starting...")
    idle_checker = asyncio.create_task(manager.check_idle_models())
    yield
    log.info("Shutting down...")
    idle_checker.cancel()
    try:
        await idle_checker
    except asyncio.CancelledError:
        pass

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

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
