from fastapi import Depends, FastAPI

import skvaider.routers.openai
from skvaider.auth import BearerToken

app = FastAPI(dependencies=[Depends(BearerToken)])


app.include_router(
    skvaider.routers.openai.router,
    prefix="/openai",
)
