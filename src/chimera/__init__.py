from fastapi import Depends, FastAPI

from chimera.auth import BearerToken
import chimera.routers.openai

app = FastAPI(dependencies=[Depends(BearerToken)])


app.include_router(
    chimera.routers.openai.router,
    prefix="/openai",
)
