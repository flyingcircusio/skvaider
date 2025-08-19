from fastapi import Depends, FastAPI

from skvaider.auth import BearerToken
import skvaider.routers.openai

app = FastAPI(dependencies=[Depends(BearerToken)])


app.include_router(
    skvaider.routers.openai.router,
    prefix="/openai",
)
